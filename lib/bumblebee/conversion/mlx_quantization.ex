defmodule Bumblebee.Conversion.MLXQuantization do
  @moduledoc false

  # MLX quantized checkpoints store each quantized weight as packed
  # unsigned 32-bit words, next to a scale and a bias for every group
  # of values along its last axis. EMLX runs matrix multiplications
  # directly on that representation, so these weights are kept packed.

  @compile {:no_warn_undefined, [EMLX.Backend, EMLX.Quantization]}

  defmodule QuantizedTensor do
    @moduledoc false

    # A lazy quantized weight, read from the checkpoint only when used.
    defstruct [:weight, :scales, :biases, :shape, :bits, :group_size]
  end

  @doc """
  Validates the "quantization" entry of an MLX checkpoint configuration.
  """
  def validate!(quantization, backend) do
    mode = Map.get(quantization, "mode", "affine")

    unless mode == "affine" do
      raise ArgumentError,
            "MLX quantized weights in #{inspect(mode)} mode are not supported, only \"affine\" is"
    end

    unless emlx_backend?(backend) do
      raise ArgumentError,
            "loading MLX quantized weights requires the EMLX backend, " <>
              "pass backend: {EMLX.Backend, device: :gpu} or set it as the default backend"
    end

    :ok
  end

  defp emlx_backend?(backend) do
    module =
      case backend || Nx.default_backend() do
        {module, _opts} -> module
        module -> module
      end

    module == EMLX.Backend and Code.ensure_loaded?(EMLX.Quantization)
  end

  @doc """
  Replaces each quantized weight in the state dictionary, together
  with its scales and biases, by a single lazy quantized tensor.
  """
  def quantize_state(state, quantization) do
    prefixes =
      for key <- Map.keys(state), String.ends_with?(key, ".scales") do
        String.replace_suffix(key, ".scales", "")
      end

    Enum.reduce(prefixes, state, fn prefix, state ->
      {weight, state} = Map.pop!(state, prefix <> ".weight")
      {scales, state} = Map.pop!(state, prefix <> ".scales")
      {biases, state} = Map.pop!(state, prefix <> ".biases")

      %{"bits" => bits, "group_size" => group_size} = layer_quantization(quantization, prefix)
      {out_features, num_groups} = lazy_shape(scales)

      quantized = %QuantizedTensor{
        weight: weight,
        scales: scales,
        biases: biases,
        shape: {out_features, num_groups * group_size},
        bits: bits,
        group_size: group_size
      }

      Map.put(state, prefix <> ".weight", quantized)
    end)
  end

  # Mixed-precision conversions override the default per layer
  defp layer_quantization(quantization, prefix) do
    case quantization do
      %{^prefix => %{} = override} -> Map.merge(quantization, override)
      %{} -> quantization
    end
  end

  defp lazy_shape(lazy) do
    {template, _} =
      Nx.LazyContainer.traverse(lazy, nil, fn template, _fun, acc -> {template, acc} end)

    Nx.shape(template)
  end

  @doc """
  Reads a lazy quantized tensor into an EMLX quantized tensor.
  """
  def materialize(%QuantizedTensor{} = quantized) do
    [weight, scales, biases] =
      for lazy <- [quantized.weight, quantized.scales, quantized.biases] do
        lazy |> Nx.to_tensor() |> EMLX.Backend.from_nx()
      end

    EMLX.Quantization.quantized_tensor(weight, scales, biases, quantized.shape,
      type: {:s, quantized.bits},
      group_size: quantized.group_size
    )
  end

  @doc """
  Returns whether the given tensor is quantized.
  """
  def quantized?(%Nx.Tensor{data: %{__struct__: EMLX.Backend}} = tensor) do
    EMLX.Quantization.quantized?(tensor)
  end

  def quantized?(_tensor), do: false

  @doc """
  Converts a quantized tensor to a regular tensor.
  """
  def dequantize(tensor) do
    if quantized?(tensor), do: EMLX.Quantization.dequantize(tensor), else: tensor
  end

  defimpl Nx.LazyContainer, for: QuantizedTensor do
    def traverse(quantized, acc, fun) do
      template = Nx.template(quantized.shape, {:s, quantized.bits})
      fun.(template, fn -> Bumblebee.Conversion.MLXQuantization.materialize(quantized) end, acc)
    end
  end
end
