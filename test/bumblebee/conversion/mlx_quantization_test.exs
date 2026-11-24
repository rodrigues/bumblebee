defmodule Bumblebee.Conversion.MLXQuantizationTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()
  @moduletag :emlx

  @backend {EMLX.Backend, device: :gpu}
  @repo_4bit {:hf, "bumblebee-testing/tiny-random-Qwen3ForCausalLM-mlx-4bit"}
  @repo_8bit {:hf, "bumblebee-testing/tiny-random-Qwen3ForCausalLM-mlx-8bit"}
  @repo_qwen3_5 {:hf, "bumblebee-testing/tiny-random-Qwen3_5ForConditionalGeneration-mlx-4bit"}

  defp predict(model, params) do
    inputs = %{"input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], backend: @backend)}

    model
    |> Axon.predict(params, inputs, compiler: EMLX)
    |> Map.fetch!(:logits)
    |> to_binary_backend()
  end

  test "keeps dense kernels quantized and dequantizes the embedding table" do
    assert {:ok, %{params: params}} = Bumblebee.load_model(@repo_4bit, backend: @backend)

    query = params.data["decoder.blocks.0.self_attention.query"]["kernel"]
    assert EMLX.Quantization.quantized?(query)
    assert Nx.shape(query) == {32, 512}
    assert Nx.type(query) == {:s, 4}

    head = params.data["language_modeling_head.output"]["kernel"]
    assert EMLX.Quantization.quantized?(head)
    assert Nx.shape(head) == {1024, 32}

    # The checkpoint leaves weights whose input size is not a multiple of the group size unquantized
    down = params.data["decoder.blocks.0.ffn.output"]["kernel"]
    refute EMLX.Quantization.quantized?(down)
    assert Nx.shape(down) == {37, 32}

    embedding = params.data["embedder.token_embedding"]["kernel"]
    refute EMLX.Quantization.quantized?(embedding)

    # Reference values from mlx.core.dequantize on the same table
    assert_all_close(
      to_binary_backend(embedding[[10..12, 1..3]]),
      Nx.tensor([
        [-0.0373, -0.0062, -0.0062],
        [-0.0065, -0.0647, 0.0194],
        [-0.0265, -0.0106, 0.0159]
      ])
    )
  end

  for {name, repo, count} <- [
        {"4-bit", @repo_4bit, 13},
        {"8-bit", @repo_8bit, 13},
        {"Qwen3.5 4-bit", @repo_qwen3_5, 35}
      ] do
    test "#{name} quantized kernels compute the same products as their dequantized weights" do
      assert {:ok, %{params: params}} =
               Bumblebee.load_model(unquote(Macro.escape(repo)), backend: @backend)

      kernels =
        for {layer, %{"kernel" => kernel}} <- params.data,
            EMLX.Quantization.quantized?(kernel),
            do: {layer, kernel}

      assert length(kernels) == unquote(count)

      for {layer, kernel} <- kernels do
        # The output head contracts the last axis of its kernel, every other layer the first
        axis = if layer == "language_modeling_head.output", do: 1, else: 0
        size = elem(Nx.shape(kernel), axis)
        input = Nx.iota({1, 8, size}, backend: Nx.BinaryBackend) |> Nx.divide(size) |> Nx.sin()

        # The reference runs on the binary backend, because MLX computes float32
        # products of several rows in TF32 by default on GPUs with neural
        # accelerators, such as the M5, up to 7.4e-4 from the exact result
        dense = kernel |> EMLX.Quantization.dequantize() |> to_binary_backend()
        expected = Nx.dot(input, [2], dense, [axis])

        product =
          input |> Nx.backend_copy(@backend) |> Nx.dot([2], kernel, [axis]) |> to_binary_backend()

        assert_all_close(product, expected, atol: 1.0e-5)
      end
    end
  end

  test "logits match mlx-lm" do
    assert {:ok, %{model: model, params: params}} =
             Bumblebee.load_model(@repo_4bit, backend: @backend)

    # Reference values from mlx-lm 0.32.2 in full float32. The tolerance covers
    # mlx-lm always using SiLU, while this checkpoint sets GELU, 3.5e-4 apart,
    # and MLX's default TF32 products on GPUs with neural accelerators
    assert_all_close(
      predict(model, params)[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1601, 0.0822, -0.0808], [0.1610, -0.0205, -0.0057], [0.2381, -0.0044, -0.0262]]
      ]),
      atol: 1.0e-3
    )
  end

  test "Qwen3.5 logits match mlx-lm" do
    assert {:ok, %{model: model, params: params}} =
             Bumblebee.load_model(@repo_qwen3_5, backend: @backend)

    assert EMLX.Quantization.quantized?(
             params.data["decoder.blocks.0.linear_attention.query_key_value"]["kernel"]
           )

    # Reference values from mlx-lm 0.32.2. Through the linear attention, MLX's
    # float32 products put these logits 2.6e-3 from an exact computation with
    # the same weights, and 3.1e-3 from mlx-lm, which runs it step by step
    assert_all_close(
      predict(model, params)[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.0165, -0.2746, 1.0480], [0.2671, 2.3714, 0.2874], [1.1254, -0.2586, -0.1098]]
      ]),
      atol: 5.0e-3
    )
  end

  test "raises without the EMLX backend" do
    assert_raise ArgumentError, ~r/loading MLX quantized weights requires the EMLX backend/, fn ->
      Bumblebee.load_model(@repo_4bit, backend: Nx.BinaryBackend)
    end
  end

  test "raises on a quantization mode other than affine" do
    quantization = %{"group_size" => 32, "bits" => 4, "mode" => "mxfp4"}

    assert_raise ArgumentError, ~r/MLX quantized weights in "mxfp4" mode are not supported/, fn ->
      Bumblebee.Conversion.MLXQuantization.validate!(quantization, @backend)
    end
  end
end
