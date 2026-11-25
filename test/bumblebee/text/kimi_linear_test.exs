defmodule Bumblebee.Text.KimiLinearTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  # 22 tokens span two chunks of the chunked delta rule
  defp inputs() do
    input_ids = Nx.iota({1, 22}) |> Nx.multiply(37) |> Nx.remainder(1000) |> Nx.add(10)
    attention_mask = Nx.concatenate([Nx.broadcast(1, {1, 20}), Nx.broadcast(0, {1, 2})], axis: 1)
    %{"input_ids" => input_ids, "attention_mask" => attention_mask}
  end

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-KimiLinearForCausalLM"},
               architecture: :base
             )

    assert %Bumblebee.Text.KimiLinear{architecture: :base} = spec

    assert spec.layer_types == [
             :linear_attention,
             :linear_attention,
             :linear_attention,
             :full_attention,
             :linear_attention
           ]

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 22, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-2.3836, -0.3281, -0.2326], [-0.5739, -0.6473, -1.1054], [-1.1508, 0.2755, -1.6886]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-KimiLinearForCausalLM"})

    assert %Bumblebee.Text.KimiLinear{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 22, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0169, -1.5670, -1.8869], [-2.0884, -0.5399, 2.0655], [-0.4413, -4.2156, -0.3226]]
      ])
    )

    # In the second chunk
    assert_all_close(
      outputs.logits[[.., 17..19, 1..3]],
      Nx.tensor([
        [[-2.1705, 0.2246, 3.6935], [0.5027, 0.3806, 0.9272], [1.2712, -0.3505, 3.0494]]
      ])
    )
  end

  test "left padding does not change the logits of the real tokens" do
    assert {:ok, %{model: model, params: params}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-KimiLinearForCausalLM"})

    %{"input_ids" => input_ids} = inputs()
    input_ids = input_ids[[.., 0..17]]

    padded = %{
      "input_ids" => Nx.concatenate([Nx.tensor([[0, 0, 0]]), input_ids], axis: 1),
      "attention_mask" =>
        Nx.concatenate([Nx.broadcast(0, {1, 3}), Nx.broadcast(1, {1, 18})], axis: 1)
    }

    unpadded = Axon.predict(model, params, %{"input_ids" => input_ids})
    padded = Axon.predict(model, params, padded)

    assert_all_close(padded.logits[[.., 3..-1//1]], unpadded.logits, atol: 1.0e-4)
  end

  test "generation" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-KimiLinearForCausalLM"})

    generation_config =
      Bumblebee.configure(Bumblebee.Text.GenerationConfig,
        max_new_tokens: 8,
        strategy: %{type: :greedy_search},
        eos_token_id: nil,
        pad_token_id: 0
      )

    %{"input_ids" => input_ids} = inputs()
    inputs = %{"input_ids" => input_ids[[.., 0..17]], "seed" => Nx.tensor([0])}

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[34, 979, 384, 635, 227, 793, 841, 152]]))
  end
end
