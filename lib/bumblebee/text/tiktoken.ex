defmodule Bumblebee.Text.Tiktoken do
  @moduledoc false

  # Converts a tiktoken vocabulary into the tokenizer.json format of
  # the Rust tokenizers library, as transformers' TikTokenConverter does.
  #
  # A tiktoken file holds only the tokens and their merge ranks. The
  # pattern that splits text before merging lives in each model's Python
  # tokenizer code, so it is kept here per tokenizer type.

  @patterns %{
    kimi:
      Enum.join(
        [
          ~S"[\p{Han}]+",
          ~S"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
          ~S"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
          ~S"\p{N}{1,3}",
          ~S" ?[^\s\p{L}\p{N}]+[\r\n]*",
          ~S"\s*[\r\n]+",
          ~S"\s+(?!\S)",
          ~S"\s+"
        ],
        "|"
      )
  }

  @doc """
  Returns the tokenizer.json content for the tiktoken vocabulary at
  `path`, with the added tokens listed in `tokenizer_config`.
  """
  @spec tokenizer_json(Path.t(), atom(), map()) :: map()
  def tokenizer_json(path, type, tokenizer_config) do
    pattern =
      @patterns[type] ||
        raise ArgumentError,
              "converting a tiktoken vocabulary needs the split pattern of the tokenizer," <>
                " which is known only for the types #{inspect(Map.keys(@patterns))}, got: #{inspect(type)}"

    ranks = read_ranks(path)
    to_unicode = byte_to_unicode()
    to_string = fn bytes -> for <<byte <- bytes>>, into: "", do: Map.fetch!(to_unicode, byte) end

    vocab = Map.new(ranks, fn {bytes, rank} -> {to_string.(bytes), rank} end)

    merges =
      ranks
      |> Enum.sort_by(&elem(&1, 1))
      |> Enum.flat_map(fn {bytes, rank} -> token_merges(bytes, rank, ranks) end)
      |> Enum.sort_by(&elem(&1, 2))
      |> Enum.map(fn {left, right, _rank} -> [to_string.(left), to_string.(right)] end)

    configured =
      Map.new(tokenizer_config["added_tokens_decoder"] || %{}, fn {id, token} ->
        {String.to_integer(id), token}
      end)

    # The tokenizers library numbers added tokens consecutively after the
    # vocabulary, so the ids between configured tokens are filled with
    # reserved tokens, as the Python tokenizers of tiktoken models do
    added_tokens =
      if configured == %{} do
        []
      else
        for id <- map_size(ranks)..Enum.max(Map.keys(configured)) do
          token = Map.get(configured, id, %{"content" => "<|reserved_token_#{id}|>"})

          %{
            "id" => id,
            "content" => token["content"],
            "single_word" => Map.get(token, "single_word", false),
            "lstrip" => Map.get(token, "lstrip", false),
            "rstrip" => Map.get(token, "rstrip", false),
            "normalized" => false,
            "special" => Map.get(token, "special", true)
          }
        end
      end

    %{
      "version" => "1.0",
      "truncation" => nil,
      "padding" => nil,
      "added_tokens" => added_tokens,
      "normalizer" => nil,
      "pre_tokenizer" => %{
        "type" => "Sequence",
        "pretokenizers" => [
          %{
            "type" => "Split",
            "pattern" => %{"Regex" => pattern},
            "behavior" => "Isolated",
            "invert" => false
          },
          %{
            "type" => "ByteLevel",
            "add_prefix_space" => false,
            "trim_offsets" => true,
            "use_regex" => false
          }
        ]
      },
      "post_processor" => nil,
      "decoder" => %{
        "type" => "ByteLevel",
        "add_prefix_space" => true,
        "trim_offsets" => true,
        "use_regex" => true
      },
      "model" => %{
        "type" => "BPE",
        "dropout" => nil,
        "unk_token" => nil,
        "continuing_subword_prefix" => nil,
        "end_of_word_suffix" => nil,
        "fuse_unk" => false,
        "byte_fallback" => false,
        # tiktoken takes a whole piece as one token when it is in the
        # vocabulary, before trying merges
        "ignore_merges" => true,
        "vocab" => vocab,
        "merges" => merges
      }
    }
  end

  defp read_ranks(path) do
    path
    |> File.stream!()
    |> Enum.flat_map(fn line ->
      case String.split(line) do
        [token, rank] -> [{Base.decode64!(token), String.to_integer(rank)}]
        [] -> []
      end
    end)
    |> Map.new()
  end

  # Every split of a token into two tokens whose concatenation is in the
  # vocabulary is a merge with the token's rank, ordered by the ranks of
  # the two halves
  defp token_merges(bytes, _rank, _ranks) when byte_size(bytes) == 1, do: []

  defp token_merges(bytes, rank, ranks) do
    for index <- 1..(byte_size(bytes) - 1),
        <<left::binary-size(^index), right::binary>> = bytes,
        Map.has_key?(ranks, left) and Map.has_key?(ranks, right) do
      {left, right, rank}
    end
    |> Enum.sort_by(fn {left, right, _rank} -> {ranks[left], ranks[right]} end)
  end

  # The byte-to-character table of GPT-2's byte-level BPE, which maps
  # every byte to a printable character
  defp byte_to_unicode() do
    printable = Enum.concat([?!..?~, ?¡..?¬, ?®..?ÿ])
    others = Enum.reject(0..255, &(&1 in printable))

    Map.new(
      Enum.zip(printable, printable) ++
        Enum.with_index(others, fn byte, index -> {byte, 256 + index} end),
      fn {byte, codepoint} -> {byte, <<codepoint::utf8>>} end
    )
  end
end
