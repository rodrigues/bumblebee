defmodule Bumblebee.Text.Llama4Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  defp inputs() do
    %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }
  end

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Llama4ForCausalLM"},
               architecture: :base
             )

    assert %Bumblebee.Text.Llama4{architecture: :base} = spec

    assert spec.layer_types == [
             :chunked_attention,
             :full_attention,
             :chunked_attention,
             :full_attention
           ]

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.hidden_state) == {1, 12, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.5806, -1.0373, 1.4803], [-0.8202, -0.5094, 0.4957], [0.2250, -1.2834, -0.4858]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Llama4ForCausalLM"})

    assert %Bumblebee.Text.Llama4{architecture: :for_causal_language_modeling} = spec

    outputs = Axon.predict(model, params, inputs())

    assert Nx.shape(outputs.logits) == {1, 12, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.4061, -0.0815, 0.5487], [-1.5561, -1.1119, -2.9968], [0.5637, -1.5385, 0.6490]]
      ])
    )
  end

  test ":for_causal_language_modeling with a null attention chunk size" do
    repo = {:hf, "bumblebee-testing/tiny-random-Llama4ForCausalLM"}
    {:ok, spec} = Bumblebee.load_spec(repo)

    spec =
      Bumblebee.HuggingFace.Transformers.Config.load(spec, %{"attention_chunk_size" => nil})

    assert spec.attention_chunk_size == nil

    assert {:ok, %{model: model, params: params}} = Bumblebee.load_model(repo, spec: spec)

    assert {:ok, %{model: causal_model, params: causal_params}} =
             Bumblebee.load_model(repo, spec: Bumblebee.configure(spec, attention_chunk_size: 12))

    assert_all_close(
      Axon.predict(model, params, inputs()).logits,
      Axon.predict(causal_model, causal_params, inputs()).logits
    )
  end

  test ":for_causal_language_modeling with experts in every other block" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Llama4ForCausalLM-interleaved"}
             )

    assert spec.expert_block_indices == [1, 3]
    assert spec.num_experts_per_token == 2
    assert spec.use_query_key_norm == false

    outputs = Axon.predict(model, params, inputs())

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.3706, 2.0226, 1.3732], [-1.6455, 1.8125, 2.2911], [-2.2315, 0.8752, 1.2895]]
      ])
    )
  end

  test ":for_causal_language_modeling from a multimodal checkpoint" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "optimum-intel-internal-testing/tiny-random-llama4"})

    assert %Bumblebee.Text.Llama4{architecture: :for_causal_language_modeling} = spec
    assert spec.attention_temperature_tuning == true

    outputs = Axon.predict(model, params, inputs())

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0237, 0.1952, 0.0716], [-0.0035, 0.0981, 0.0855], [-0.0122, 0.0863, 0.0623]]
      ])
    )
  end

  test "query temperature past the largest float16 integer" do
    repo = {:hf, "bumblebee-testing/tiny-random-Llama4ForCausalLM"}
    offset = 65_520

    # EXLA fuses the cast of the positions with the division and computes
    # it in float32, so this runs on a backend that rounds every step. The
    # temperature depends only on the cache offset, so a short cache keeps
    # it fast
    logits = fn type ->
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        assert {:ok, %{model: model, params: params, spec: spec}} =
                 Bumblebee.load_model(repo, type: type)

        input_ids = Nx.tensor([[10, 20, 30, 40]])
        cache = Bumblebee.Text.Generation.init_cache(spec, 1, 8, %{})

        inputs = %{
          "input_ids" => input_ids,
          "attention_mask" => Nx.broadcast(1, input_ids),
          "position_ids" => Nx.iota({1, 4}) |> Nx.add(offset),
          "cache" => %{cache | offset: Nx.tensor(offset)}
        }

        outputs = Axon.predict(model, params, inputs, compiler: Nx.Defn.Evaluator)
        Nx.as_type(outputs.logits, :f32)
      end)
    end

    assert_all_close(logits.(:f16), logits.(:f32), atol: 5.0e-2)
  end

  # The cache crosses two chunk boundaries and two query temperature steps
  for {repo, expected} <- [
        {"bumblebee-testing/tiny-random-Llama4ForCausalLM",
         [667, 823, 113, 198, 420, 502, 484, 709]},
        {"bumblebee-testing/tiny-random-Llama4ForCausalLM-interleaved",
         [881, 881, 99, 671, 66, 262, 635, 774]}
      ] do
    test "generation with #{repo}" do
      assert {:ok, %{model: model, params: params, spec: spec}} =
               Bumblebee.load_model({:hf, unquote(repo)})

      generation_config =
        Bumblebee.configure(Bumblebee.Text.GenerationConfig,
          max_new_tokens: 8,
          strategy: %{type: :greedy_search},
          eos_token_id: nil,
          pad_token_id: 0
        )

      inputs = %{
        "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60]]),
        "seed" => Nx.tensor([0])
      }

      generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
      %{token_ids: token_ids} = generate.(params, inputs)

      assert_equal(token_ids, Nx.tensor([unquote(expected)]))
    end
  end
end
