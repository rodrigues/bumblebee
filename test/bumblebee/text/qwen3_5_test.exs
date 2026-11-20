defmodule Bumblebee.Text.Qwen3_5Test do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3_5ForCausalLM"},
               architecture: :base
             )

    assert %Bumblebee.Text.Qwen3_5{architecture: :base} = spec

    assert spec.block_types == [
             :linear_attention,
             :linear_attention,
             :linear_attention,
             :full_attention,
             :linear_attention
           ]

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[1.4133, 0.5996, -2.3100], [1.2465, 0.0291, 0.3573], [-1.9832, -0.8136, 0.5604]]
      ]),
      atol: 1.0e-4
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3_5ForCausalLM"})

    assert %Bumblebee.Text.Qwen3_5{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.4915, 2.3465, -0.1588], [1.1608, -0.5369, 0.9091], [0.7094, 1.0836, -1.3933]]
      ]),
      atol: 1.0e-4
    )
  end

  test ":for_causal_language_modeling from a multimodal checkpoint" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Qwen3_5ForConditionalGeneration"}
             )

    assert %Bumblebee.Text.Qwen3_5{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.7326, 1.1803, 0.4093], [1.0902, 0.5977, 0.2788], [-0.6475, 0.8144, 0.5186]]
      ]),
      atol: 1.0e-4
    )
  end

  test ":for_causal_language_modeling from an MLX checkpoint" do
    # The same weights as the multimodal checkpoint, converted with mlx-lm,
    # which renames them, moves the convolution kernel axis and adds the
    # norm shift to the weights
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Qwen3_5ForConditionalGeneration-mlx"}
             )

    assert %Bumblebee.Text.Qwen3_5{architecture: :for_causal_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.7326, 1.1803, 0.4093], [1.0902, 0.5977, 0.2788], [-0.6475, 0.8144, 0.5186]]
      ]),
      atol: 1.0e-4
    )
  end

  test "generation with cache" do
    repo = {:hf, "bumblebee-testing/tiny-random-Qwen3_5ForCausalLM"}
    assert {:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model(repo)
    {:ok, generation_config} = Bumblebee.load_generation_config(repo)

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 8,
        strategy: %{type: :greedy_search},
        eos_token_id: nil,
        pad_token_id: 0
      )

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
      "seed" => Nx.tensor([0])
    }

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate.(params, inputs)

    assert_equal(token_ids, Nx.tensor([[279, 537, 784, 485, 860, 775, 279, 973]]))
  end

  test "left padding gives the same logits as unpadded inputs" do
    assert {:ok, %{model: model, params: params}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3_5ForCausalLM"})

    unpadded =
      Axon.predict(model, params, %{
        "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60]])
      })

    # Position ids skip the padding, as in generation
    padded =
      Axon.predict(model, params, %{
        "input_ids" => Nx.tensor([[0, 0, 10, 20, 30, 40, 50, 60]]),
        "attention_mask" => Nx.tensor([[0, 0, 1, 1, 1, 1, 1, 1]]),
        "position_ids" => Nx.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])
      })

    assert_all_close(padded.logits[[.., 2..-1//1]], unpadded.logits, atol: 1.0e-5)
  end

  test "refuses the mixture-of-experts variant" do
    path = Path.join(System.tmp_dir!(), "bumblebee-qwen3_5-moe")
    File.mkdir_p!(path)

    File.write!(
      Path.join(path, "config.json"),
      ~s({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"})
    )

    assert_raise ArgumentError, ~r/"Qwen3_5MoeForConditionalGeneration"/, fn ->
      Bumblebee.load_spec({:local, path})
    end
  end
end
