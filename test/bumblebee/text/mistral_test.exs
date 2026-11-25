defmodule Bumblebee.Text.MistralTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MistralModel"})

    assert %Bumblebee.Text.Mistral{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.9450, -1.3945, 0.7331], [-2.1118, -1.3091, -0.7834], [-1.7609, -1.3034, 1.0634]]
      ])
    )
  end

  test ":base with attention sliding window" do
    assert {:ok, spec} =
             Bumblebee.load_spec({:hf, "hf-internal-testing/tiny-random-MistralModel"})

    spec = Bumblebee.configure(spec, attention_window_size: 2)

    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MistralModel"},
               spec: spec
             )

    assert %Bumblebee.Text.Mistral{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.9450, -1.3945, 0.7331], [-1.4422, -1.4622, -0.9143], [-1.5628, -1.0444, 0.9262]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-MistralForSequenceClassification"}
             )

    assert %Bumblebee.Text.Mistral{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[0.0035, -0.0357]])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-MistralForCausalLM"})

    assert %Bumblebee.Text.Mistral{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 32000}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.1054, 0.0026, 0.0450], [0.1400, 0.1388, 0.0265], [0.0060, -0.1150, -0.1463]]
      ])
    )
  end

  test ":base with Ministral 3" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-tiny-v2/tiny-random-Ministral3ForCausalLM"},
               architecture: :base
             )

    assert %Bumblebee.Text.Mistral{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0132, 0.3972, -0.1655], [-0.4824, 1.5165, -0.6992], [0.6401, 0.5267, -0.0813]]
      ])
    )
  end

  test ":for_causal_language_modeling with Ministral 3" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-tiny-v2/tiny-random-Ministral3ForCausalLM"})

    assert %Bumblebee.Text.Mistral{
             architecture: :for_causal_language_modeling,
             attention_head_size: 16,
             query_scaling_beta: 0.1,
             query_scaling_positions: 16384
           } = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 99}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1105, -0.0606, 0.1705], [-0.1473, -0.1888, 0.1039], [-0.0970, -0.0620, 0.1247]]
      ])
    )
  end

  test "Ministral 3 query scaling past the original context length" do
    assert {:ok, %{model: model, params: params}} =
             Bumblebee.load_model({:hf, "hf-tiny-v2/tiny-random-Ministral3ForCausalLM"})

    # Random weights make attention nearly uniform, so sharper queries are
    # needed for the query scaling to change the logits
    params =
      update_in(params.data, fn data ->
        Map.new(data, fn
          {"decoder.blocks." <> _ = name, %{"kernel" => kernel} = layer} = entry ->
            if String.ends_with?(name, "self_attention.query"),
              do: {name, %{layer | "kernel" => Nx.multiply(kernel, 30.0)}},
              else: entry

          entry ->
            entry
        end)
      end)

    # The scaling starts at position 16384, the fifth of these
    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "position_ids" => Nx.iota({1, 10}) |> Nx.add(16380)
    }

    outputs = Axon.predict(model, params, inputs)

    assert_all_close(
      outputs.logits[[.., 3..5, 1..3]],
      Nx.tensor([
        [[-0.0836, -0.0517, 0.1327], [-0.1127, 0.1216, -0.0757], [-0.0838, -0.2484, -0.0081]]
      ])
    )
  end

  test "Ministral 3 query scaling with a short original context length" do
    repo = {:hf, "hf-tiny-v2/tiny-random-Ministral3ForCausalLM"}
    assert {:ok, spec} = Bumblebee.load_spec(repo)

    # With 8 original positions, the query scaling grows within 20 tokens
    spec =
      Bumblebee.configure(spec,
        query_scaling_positions: 8,
        rotary_embedding_scaling_strategy: %{
          spec.rotary_embedding_scaling_strategy
          | original_max_positions: 8
        }
      )

    assert {:ok, %{model: model, params: params}} = Bumblebee.load_model(repo, spec: spec)

    inputs = %{"input_ids" => Nx.iota({1, 20}) |> Nx.multiply(3) |> Nx.add(5)}
    outputs = Axon.predict(model, params, inputs)

    # The scaling changes these logits by about 1.2e-4, hence the tolerance
    assert_all_close(
      Nx.take(outputs.logits[[0, .., 1..3]], Nx.tensor([7, 8, 19])),
      Nx.tensor([
        [0.11402107, -0.01341134, 0.11898933],
        [0.13499461, -0.13389961, 0.18508185],
        [-0.12670033, 0.10041963, -0.26193604]
      ]),
      atol: 1.0e-6
    )
  end

  test "generation with Ministral 3" do
    repo = {:hf, "hf-tiny-v2/tiny-random-Ministral3ForCausalLM"}
    assert {:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model(repo)

    generation_config =
      Bumblebee.configure(Bumblebee.Text.GenerationConfig,
        max_new_tokens: 8,
        strategy: %{type: :greedy_search},
        eos_token_id: nil,
        pad_token_id: 0
      )

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80]]),
      "seed" => Nx.tensor([0])
    }

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[11, 90, 1, 95, 51, 88, 49, 4]]))
  end

  test ":for_causal_language_modeling from a multimodal Mistral 3 checkpoint" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration"}
             )

    # The multimodal config ties the output head, while its text config does not
    assert %Bumblebee.Text.Mistral{
             architecture: :for_causal_language_modeling,
             tie_word_embeddings: true
           } = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.0502, -0.0152, -0.0476], [-0.0307, -0.0575, -0.0609], [0.1117, -0.0575, 0.1333]]
      ])
    )
  end
end
