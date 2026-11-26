defmodule Bumblebee.Layers.LinearAttention do
  @moduledoc false

  # Building blocks of linear attention with a delta rule, where each
  # head keeps a state of size {key_head_size, value_head_size} that
  # every token decays, reads and corrects, in place of attending to
  # all previous tokens.

  import Nx.Defn

  alias Bumblebee.Layers

  @doc """
  Adds a causal depthwise convolution over the sequence.

  The convolution sees the inputs of the preceding tokens, which are
  kept in the cache under `:conv_state`, or zeros at the start of the
  sequence. Returns the output and the updated cache.

  ## Options

    * `:channels` - the number of input channels

    * `:kernel_size` - the number of tokens each output sees

    * `:kernel_initializer` - initializer for the kernel

    * `:name` - layer name

  """
  def causal_conv(input, cache, opts) do
    opts = Keyword.validate!(opts, [:channels, :kernel_size, :kernel_initializer, :name])

    kernel =
      Axon.param("kernel", {opts[:channels], opts[:kernel_size]},
        initializer: opts[:kernel_initializer]
      )

    Axon.layer(&causal_conv_impl/4, [input, kernel, Axon.optional(cache)],
      name: opts[:name],
      op_name: :causal_conv
    )
    |> Layers.unwrap_tuple(2)
  end

  deftransformp causal_conv_impl(input, kernel, cache, _opts \\ []) do
    {channels, kernel_size} = Nx.shape(kernel)
    {batch_size, sequence_length, ^channels} = Nx.shape(input)

    window =
      case cache do
        %Axon.None{} ->
          Nx.broadcast(
            Nx.tensor(0, type: Nx.type(input)),
            {batch_size, kernel_size - 1, channels}
          )

        %{conv_state: conv_state} ->
          Nx.as_type(conv_state, Nx.type(input))
      end

    full_input = Nx.concatenate([window, input], axis: 1)

    output =
      0..(kernel_size - 1)
      |> Enum.map(fn offset ->
        full_input
        |> Nx.slice_along_axis(offset, sequence_length, axis: 1)
        |> Nx.multiply(kernel[[.., offset]])
      end)
      |> Enum.reduce(&Nx.add/2)

    cache =
      case cache do
        %Axon.None{} ->
          cache

        %{conv_state: conv_state} = cache ->
          window = Nx.slice_along_axis(full_input, sequence_length, kernel_size - 1, axis: 1)
          %{cache | conv_state: Nx.as_type(window, Nx.type(conv_state))}
      end

    {output, cache}
  end

  @doc """
  Applies the delta rule to the given heads, starting from `state`.

  Query, key and value have the shape `{batch_size, sequence_length,
  num_heads, head_size}`, and beta `{batch_size, sequence_length,
  num_heads}`. The decay is a log decay, either one per head, with the
  shape of beta, or one per key channel, with the shape of the key. The
  query and key are L2-normalized here. Returns the output and the
  final state.

  Decoding a single token takes one step of the recurrence, while longer
  inputs are processed in chunks.

  ## Options

    * `:chunk_size` - the number of tokens solved together within a
      chunk. Pairwise decays per key channel take `chunk_size` times
      more memory than per head, so they call for shorter chunks

  """
  deftransform delta_rule(query, key, value, beta, decay, state, opts \\ []) do
    opts = Keyword.validate!(opts, [:chunk_size])
    chunk_size = opts[:chunk_size]
    sequence_length = Nx.axis_size(value, 1)

    # A decay per head acts as a decay shared by every key channel
    decay = if Nx.rank(decay) == 3, do: Nx.new_axis(decay, -1), else: decay

    if sequence_length == 1 do
      delta_rule_single(query, key, value, beta, decay, state)
    else
      padding = rem(chunk_size - rem(sequence_length, chunk_size), chunk_size)

      delta_rule_chunked(query, key, value, beta, decay, state,
        padding: padding,
        chunk_size: chunk_size
      )
    end
  end

  defnp delta_rule_single(query, key, value, beta, decay, state) do
    {query, key} = normalize_query_key(query, key)
    take = &Nx.squeeze(&1, axes: [1])

    {query, key, value, beta, decay} =
      {take.(query), take.(key), take.(value), take.(beta), take.(decay)}

    state = state * Nx.new_axis(Nx.exp(decay), -1)
    memory = Nx.sum(state * Nx.new_axis(key, -1), axes: [2])
    delta = (value - memory) * Nx.new_axis(beta, -1)
    state = state + Nx.new_axis(key, -1) * Nx.new_axis(delta, 2)
    output = Nx.sum(state * Nx.new_axis(query, -1), axes: [2])

    {Nx.new_axis(output, 1), state}
  end

  # Within a chunk, the delta rule updates are condensed into matrix
  # products by solving a unit lower triangular system, so the loop runs
  # once per chunk instead of once per token
  defnp delta_rule_chunked(query, key, value, beta, decay, state, opts \\ []) do
    opts = keyword!(opts, [:padding, :chunk_size])
    chunk_size = opts[:chunk_size]
    {query, key} = normalize_query_key(query, key)
    {batch_size, sequence_length, num_heads, value_head_size} = Nx.shape(value)

    query = to_chunks(query, opts[:padding], chunk_size)
    key = to_chunks(key, opts[:padding], chunk_size)
    value = to_chunks(value, opts[:padding], chunk_size)
    beta = to_chunks(beta, opts[:padding], chunk_size)
    decay = to_chunks(decay, opts[:padding], chunk_size)

    key_beta = key * Nx.new_axis(beta, -1)
    value_beta = value * Nx.new_axis(beta, -1)

    # The decay accumulated from the start of the chunk to each position,
    # and between each pair of positions
    cumulative_decay = Nx.cumulative_sum(decay, axis: 3)

    lower =
      Nx.greater_equal(
        Nx.iota({chunk_size, chunk_size, 1}, axis: 0),
        Nx.iota({chunk_size, chunk_size, 1}, axis: 1)
      )

    pairwise = Nx.new_axis(cumulative_decay, 4) - Nx.new_axis(cumulative_decay, 3)

    pairwise_decay =
      Nx.exp(Nx.select(Nx.broadcast(lower, pairwise), pairwise, Nx.Constants.neg_infinity(:f32)))

    system = decayed_products(key_beta, key, pairwise_decay)
    intra_chunk = decayed_products(query, key, pairwise_decay)

    strictly_lower =
      Nx.greater(
        Nx.iota({chunk_size, chunk_size}, axis: 0),
        Nx.iota({chunk_size, chunk_size}, axis: 1)
      )

    unit_lower =
      Nx.select(Nx.broadcast(strictly_lower, system), system, 0.0) + Nx.eye(chunk_size)

    decayed_key_beta = key_beta * Nx.exp(cumulative_decay)

    solution =
      Nx.LinAlg.triangular_solve(
        unit_lower,
        Nx.concatenate([value_beta, decayed_key_beta], axis: 4),
        lower: true
      )

    new_values = Nx.slice_along_axis(solution, 0, value_head_size, axis: 4)

    key_cumulative_decay =
      Nx.slice_along_axis(solution, value_head_size, Nx.axis_size(decayed_key_beta, 4), axis: 4)

    last_decay = Nx.slice_along_axis(cumulative_decay, chunk_size - 1, 1, axis: 3)
    query = query * Nx.exp(cumulative_decay)
    key = key * Nx.exp(last_decay - cumulative_decay)
    # Each key channel decays its row of the state
    chunk_decay = last_decay |> Nx.exp() |> Nx.transpose(axes: [0, 1, 2, 4, 3])

    {output, state} =
      chunk_scan(query, key, new_values, key_cumulative_decay, intra_chunk, chunk_decay, state)

    output =
      output
      |> Nx.reshape({batch_size, num_heads, :auto, value_head_size})
      |> Nx.slice_along_axis(0, sequence_length, axis: 2)
      |> Nx.transpose(axes: [0, 2, 1, 3])

    {output, state}
  end

  # The products of every pair of positions in a chunk, with the decay
  # between them. A decay per head factors out of the product, so it
  # stays a matrix product; a decay per key channel does not
  deftransformp decayed_products(left, right, pairwise_decay) do
    if Nx.axis_size(pairwise_decay, -1) == 1 do
      left
      |> Nx.dot([4], [0, 1, 2], right, [4], [0, 1, 2])
      |> Nx.multiply(Nx.squeeze(pairwise_decay, axes: [-1]))
    else
      left
      |> Nx.new_axis(4)
      |> Nx.multiply(Nx.new_axis(right, 3))
      |> Nx.multiply(pairwise_decay)
      |> Nx.sum(axes: [-1])
    end
  end

  # Moves the heads before the sequence, pads the sequence to whole chunks
  # and splits it into chunks
  deftransformp to_chunks(tensor, padding, chunk_size) do
    [batch_size, sequence_length, num_heads | rest] = tensor |> Nx.shape() |> Tuple.to_list()
    axes = [0, 2, 1 | Enum.drop(Nx.axes(tensor), 3)]

    padding_config = [
      {0, 0, 0},
      {0, 0, 0},
      {0, padding, 0} | Enum.map(rest, fn _ -> {0, 0, 0} end)
    ]

    num_chunks = div(sequence_length + padding, chunk_size)

    tensor
    |> Nx.transpose(axes: axes)
    |> Nx.pad(0.0, padding_config)
    |> Nx.reshape(List.to_tuple([batch_size, num_heads, num_chunks, chunk_size | rest]))
  end

  # The number of chunks is known when building the graph, so the loop is
  # unrolled into a single graph
  deftransformp chunk_scan(
                  query,
                  key,
                  new_values,
                  key_cumulative_decay,
                  intra_chunk,
                  chunk_decay,
                  state
                ) do
    num_chunks = Nx.axis_size(new_values, 2)

    {outputs, state} =
      Enum.map_reduce(0..(num_chunks - 1), state, fn index, state ->
        take = &(&1 |> Nx.slice_along_axis(index, 1, axis: 2) |> Nx.squeeze(axes: [2]))

        delta =
          Nx.subtract(
            take.(new_values),
            Nx.dot(take.(key_cumulative_decay), [3], [0, 1], state, [2], [0, 1])
          )

        output =
          Nx.add(
            Nx.dot(take.(query), [3], [0, 1], state, [2], [0, 1]),
            Nx.dot(take.(intra_chunk), [3], [0, 1], delta, [2], [0, 1])
          )

        state =
          Nx.add(
            Nx.multiply(state, take.(chunk_decay)),
            Nx.dot(take.(key), [2], [0, 1], delta, [2], [0, 1])
          )

        {output, state}
      end)

    {Nx.stack(outputs, axis: 2), state}
  end

  defnp normalize_query_key(query, key) do
    {l2_normalize(query) / Nx.sqrt(Nx.axis_size(query, -1)), l2_normalize(key)}
  end

  defnp l2_normalize(x) do
    x * Nx.rsqrt(Nx.sum(x * x, axes: [-1], keep_axes: true) + 1.0e-6)
  end
end
