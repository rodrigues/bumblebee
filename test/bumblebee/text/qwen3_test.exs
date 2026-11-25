defmodule Bumblebee.Text.Qwen3Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3Model"})

    assert %Bumblebee.Text.Qwen3{architecture: :base} = spec
    assert spec.use_qk_norm == true

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0461, -1.5811, 1.5504], [0.1340, -1.3477, 0.8047], [-0.5821, -0.4164, 0.9769]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3ForCausalLM"})

    assert %Bumblebee.Text.Qwen3{architecture: :for_causal_language_modeling} = spec
    assert spec.use_qk_norm == true

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1457, 0.0313, -0.0651], [0.1718, -0.0265, -0.0186], [0.2281, -0.0124, -0.0147]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Qwen3ForSequenceClassification"}
             )

    assert %Bumblebee.Text.Qwen3{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(
      outputs.logits,
      Nx.tensor([[-0.1487, -0.0071]])
    )
  end

  describe "mixture of experts" do
    test ":base" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "optimum-internal-testing/tiny-random-qwen3_moe"},
                 architecture: :base
               )

      assert %Bumblebee.Text.Qwen3{architecture: :base} = spec

      inputs = %{
        "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.hidden_state) == {1, 10, 64}

      assert_all_close(
        outputs.hidden_state[[.., 1..3, 1..3]],
        Nx.tensor([
          [[0.1411, -0.1006, 0.0461], [-0.3383, -0.0336, 0.1454], [-0.4860, -0.2833, 0.0881]]
        ])
      )
    end

    test ":for_causal_language_modeling" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, "optimum-internal-testing/tiny-random-qwen3_moe"})

      assert %Bumblebee.Text.Qwen3{
               architecture: :for_causal_language_modeling,
               num_experts: 8,
               num_experts_per_token: 2,
               expert_block_step: 2
             } = spec

      inputs = %{
        "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
      }

      outputs = Axon.predict(model, params, inputs)

      assert Nx.shape(outputs.logits) == {1, 10, 151_936}

      assert_all_close(
        outputs.logits[[.., 1..3, 1..3]],
        Nx.tensor([
          [[0.0042, 0.8791, -4.3974], [-0.7148, 0.9233, -4.0622], [4.9574, 0.1915, 1.2065]]
        ])
      )
    end

    test ":for_causal_language_modeling with few tokens, which gathers the chosen experts" do
      assert {:ok, %{model: model, params: params}} =
               Bumblebee.load_model({:hf, "optimum-internal-testing/tiny-random-qwen3_moe"})

      outputs = Axon.predict(model, params, %{"input_ids" => Nx.tensor([[10, 20]])})

      assert_all_close(
        outputs.logits[[.., .., 1..3]],
        Nx.tensor([[[1.4816, -1.7118, -0.8247], [0.0042, 0.8791, -4.3974]]])
      )
    end

    test "generation" do
      repo = {:hf, "optimum-internal-testing/tiny-random-qwen3_moe"}
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

      assert_equal(
        token_ids,
        Nx.tensor([[119_063, 105_153, 9595, 63815, 41149, 81888, 70206, 57088]])
      )
    end
  end
end
