defmodule Bumblebee.Text.Qwen3 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 151_936,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 262_144,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 2560,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 9728,
        doc: "the dimensionality of intermediate layers"
      ],
      attention_head_size: [
        default: 128,
        doc: """
        the size of the key, value, and query projection per attention head.
        """
      ],
      num_blocks: [
        default: 36,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 8,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 5_000_000,
        doc: "base for computing rotary embedding frequency"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      tie_word_embeddings: [
        default: true,
        doc: "whether to tie input and output embedding weights"
      ],
      use_qk_norm: [
        default: true,
        doc: "whether to use RMS normalization on query and key projections"
      ],
      num_experts: [
        default: nil,
        doc: """
        the number of experts in the mixture-of-experts blocks. When `nil` or 0, every
        block uses a dense feed-forward network
        """
      ],
      num_experts_per_token: [
        default: 8,
        doc: "the number of experts each token is routed to"
      ],
      expert_intermediate_size: [
        default: 768,
        doc: "the dimensionality of intermediate layers in each expert"
      ],
      expert_block_step: [
        default: 1,
        doc:
          "every this many blocks use a mixture-of-experts block, starting from the last of each step"
      ],
      dense_block_indices: [
        default: [],
        doc: "the indices of blocks that use a dense feed-forward network instead of experts"
      ],
      normalize_top_k: [
        default: false,
        doc: "whether to normalize the router probabilities of the chosen experts to sum to 1"
      ]
    ] ++
      Shared.common_options([:rotary_embedding_scaling_strategy, :num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 151_643)

  @moduledoc """
  Qwen3 model family.

  This module also covers Qwen3-MoE, where blocks replace the feed-forward
  network with a mixture of experts, configured by `:num_experts`.

  ## Architectures

    * `:base` - plain Qwen3 without any head on top

    * `:for_causal_language_modeling` - Qwen3 with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - Qwen3 with a sequence
      classification head. The head returns logits corresponding to
      possible classes

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]
  import Nx.Defn

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling,
      :for_sequence_classification
    ]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_key_value_heads: spec.num_key_value_heads,
      decoder_num_blocks: spec.num_blocks
    )
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output",
        use_bias: false
      )

    pooled_logits =
      Layers.if_present inputs["input_ids"] do
        Axon.layer(
          fn logits, input_ids, _opts ->
            indices =
              input_ids
              |> Nx.not_equal(spec.pad_token_id)
              |> Nx.sum(axes: [-1])
              |> Nx.subtract(1)
              |> Nx.as_type({:s, 64})

            Bumblebee.Utils.Nx.batched_take(logits, indices)
          end,
          [logits, inputs["input_ids"]]
        )
      else
        Layers.take_token(logits, axis: 1, index: -1)
      end

    Layers.output(%{
      logits: pooled_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(
        inputs["input_ids"],
        inputs["input_embeddings"],
        spec,
        name: "embedder"
      )

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    Layers.default input_embeddings do
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
  end

  defp decoder(
         hidden_state,
         position_ids,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]

    # Query and key norm, applied per head
    qk_norm =
      if spec.use_qk_norm do
        &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, channel_index: -1, name: &2)
      end

    Layers.Transformer.blocks(
      hidden_state,
      [
        num_blocks: spec.num_blocks,
        attention_mask: attention_mask,
        attention_head_mask: attention_head_mask,
        cache: cache,
        name: join(name, "blocks")
      ],
      fn hidden_state, block ->
        name = block.name

        shortcut = hidden_state

        {hidden_state, attention, self_attention_cache} =
          hidden_state
          |> Layers.rms_norm(
            epsilon: spec.layer_norm_epsilon,
            name: join(name, "self_attention_norm")
          )
          |> Layers.Transformer.self_attention(block,
            num_heads: spec.num_attention_heads,
            num_key_value_heads: spec.num_key_value_heads,
            hidden_size: spec.hidden_size,
            attention_head_size: spec.attention_head_size,
            causal: true,
            query_norm: qk_norm,
            key_norm: qk_norm,
            rotary_embedding: [
              position_ids: position_ids,
              max_positions: spec.max_positions,
              base: spec.rotary_embedding_base,
              scaling_strategy: spec.rotary_embedding_scaling_strategy
            ],
            query_use_bias: false,
            key_use_bias: false,
            value_use_bias: false,
            output_use_bias: false,
            kernel_initializer: kernel_initializer(spec),
            name: join(name, "self_attention")
          )

        hidden_state = Axon.add(hidden_state, shortcut)

        shortcut = hidden_state

        hidden_state =
          hidden_state
          |> Layers.rms_norm(epsilon: spec.layer_norm_epsilon, name: join(name, "output_norm"))
          |> ffn(block.index, spec, name: join(name, "ffn"))
          |> Axon.add(shortcut)

        %{
          hidden_state: hidden_state,
          attention: attention,
          self_attention_cache: self_attention_cache
        }
      end
    )
  end

  defp ffn(hidden_state, index, spec, opts) do
    name = opts[:name]

    if expert_block?(spec, index) do
      router_logits =
        Axon.dense(hidden_state, spec.num_experts,
          use_bias: false,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "router")
        )

      {weights, experts} =
        Axon.layer(&route/2, [router_logits],
          num_experts_per_token: spec.num_experts_per_token,
          normalize_top_k: spec.normalize_top_k,
          op_name: :route
        )
        |> Layers.unwrap_tuple(2)

      Layers.Transformer.mixture_of_experts(
        hidden_state,
        weights,
        experts,
        spec.expert_intermediate_size,
        spec.hidden_size,
        num_experts: spec.num_experts,
        activation: spec.activation,
        kernel_initializer: kernel_initializer(spec),
        name: name
      )
    else
      Layers.Transformer.gated_ffn(hidden_state, spec.intermediate_size, spec.hidden_size,
        activation: spec.activation,
        name: name
      )
    end
  end

  defnp route(router_logits, opts \\ []) do
    opts = keyword!(opts, [:num_experts_per_token, :normalize_top_k, mode: :inference])

    probabilities = Axon.Activations.softmax(Nx.as_type(router_logits, :f32), axis: -1)
    {weights, experts} = Nx.top_k(probabilities, k: opts[:num_experts_per_token])

    weights =
      case opts[:normalize_top_k] do
        true -> weights / Nx.sum(weights, axes: [-1], keep_axes: true)
        false -> weights
      end

    {weights, experts}
  end

  defp expert_block?(spec, index) do
    experts?(spec) and index not in spec.dense_block_indices and
      rem(index + 1, spec.expert_block_step) == 0
  end

  defp experts?(spec), do: spec.num_experts != nil and spec.num_experts > 0

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          num_experts: {"num_experts", number()},
          num_experts_per_token: {"num_experts_per_tok", number()},
          expert_intermediate_size: {"moe_intermediate_size", number()},
          expert_block_step: {"decoder_sparse_step", number()},
          dense_block_indices: {"mlp_only_layers", list(number())},
          normalize_top_k: {"norm_topk_prob", boolean()}
        ) ++
          Shared.rotary_embedding_options_from_transformers(data) ++
          Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      experts =
        if spec.num_experts != nil and spec.num_experts > 0 do
          stack = fn projection ->
            sources =
              for expert <- 0..(spec.num_experts - 1) do
                [{"model.layers.{n}.mlp.experts.#{expert}.#{projection}", "weight"}]
              end

            {sources, &Nx.stack/1}
          end

          %{
            "decoder.blocks.{n}.ffn.router" => "model.layers.{n}.mlp.gate",
            "decoder.blocks.{n}.ffn.experts" => %{
              "gate_kernel" => stack.("gate_proj"),
              "intermediate_kernel" => stack.("up_proj"),
              "output_kernel" => stack.("down_proj")
            }
          }
        else
          %{}
        end

      Map.merge(experts, %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention.query_norm" => "model.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" => "model.layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.post_attention_layernorm",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head"),
        "sequence_classification_head.output" => "score"
      })
    end
  end
end
