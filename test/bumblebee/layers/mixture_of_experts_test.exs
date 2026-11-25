defmodule Bumblebee.Layers.MixtureOfExpertsTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  # 4 experts, 2 per token, hidden size 4 and intermediate size 3
  defp kernels(key) do
    {gate, key} = Nx.Random.normal(key, shape: {4, 3, 4})
    {intermediate, key} = Nx.Random.normal(key, shape: {4, 3, 4})
    {output, _key} = Nx.Random.normal(key, shape: {4, 4, 3})
    {gate, intermediate, output}
  end

  defp run(x, router_logits, {gate, intermediate, output}) do
    Bumblebee.Layers.mixture_of_experts_impl(x, router_logits, gate, intermediate, output,
      num_experts_per_token: 2,
      activation: :silu,
      normalize_top_k: true
    )
  end

  # The output of a token computed with only its two experts, spelled out
  defp reference(x, router_logits, {gate, intermediate, output}, experts) do
    probabilities = router_logits |> Nx.take(Nx.tensor(experts)) |> Axon.Activations.softmax()

    experts
    |> Enum.with_index()
    |> Enum.map(fn {expert, i} ->
      hidden =
        Nx.multiply(
          Axon.Activations.silu(Nx.dot(gate[expert], x)),
          Nx.dot(intermediate[expert], x)
        )

      Nx.multiply(Nx.dot(output[expert], hidden), probabilities[i])
    end)
    |> Enum.reduce(&Nx.add/2)
  end

  # Gathering the chosen experts is used for up to 2 tokens, and computing every expert for more
  for num_tokens <- [2, 5] do
    test "the output of a token depends only on its top k experts, with #{num_tokens} tokens" do
      num_tokens = unquote(num_tokens)
      key = Nx.Random.key(0)
      {x, key} = Nx.Random.normal(key, shape: {1, num_tokens, 4})
      {router_logits, key} = Nx.Random.normal(key, shape: {1, num_tokens, 4})
      # The first token is routed to experts 1 and 3
      router_logits =
        Nx.put_slice(router_logits, [0, 0, 0], Nx.tensor([[[-5.0, 2.0, -5.0, 1.0]]]))

      kernels = kernels(key)

      output = run(x, router_logits, kernels)

      assert_all_close(
        output[[0, 0]],
        reference(x[[0, 0]], router_logits[[0, 0]], kernels, [1, 3])
      )

      # Changing the experts the first token is not routed to leaves its output unchanged
      {gate, intermediate, output_kernel} = kernels
      changed = Nx.multiply(Nx.tensor([10.0, 1.0, 10.0, 1.0]) |> Nx.reshape({4, 1, 1}), gate)
      changed_output = run(x, router_logits, {changed, intermediate, output_kernel})

      assert_all_close(changed_output[[0, 0]], output[[0, 0]])
    end
  end
end
