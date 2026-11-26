defmodule Bumblebee.Text.TiktokenTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Text.Tiktoken

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "tiktoken.model")

    lines =
      for {token, rank} <- [{"a", 0}, {"b", 1}, {"ab", 2}],
          do: "#{Base.encode64(token)} #{rank}\n"

    File.write!(path, lines)
    %{path: path}
  end

  test "fills the ids between the vocabulary and the configured tokens with reserved tokens",
       %{path: path} do
    config = %{"added_tokens_decoder" => %{"5" => %{"content" => "<|end|>"}}}

    json = Tiktoken.tokenizer_json(path, :kimi, config)

    assert Enum.map(json["added_tokens"], &{&1["id"], &1["content"]}) == [
             {3, "<|reserved_token_3|>"},
             {4, "<|reserved_token_4|>"},
             {5, "<|end|>"}
           ]
  end

  test "keeps configured tokens inside the vocabulary without reserved tokens", %{path: path} do
    config = %{"added_tokens_decoder" => %{"1" => %{"content" => "b"}}}

    json = Tiktoken.tokenizer_json(path, :kimi, config)

    assert Enum.map(json["added_tokens"], &{&1["id"], &1["content"]}) == [{1, "b"}]
  end
end
