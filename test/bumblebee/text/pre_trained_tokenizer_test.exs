defmodule Bumblebee.Text.PreTrainedTokenizerTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  test ":deberta_v2" do
    assert {:ok, tokenizer} =
             Bumblebee.load_tokenizer({:hf, "hf-internal-testing/tiny-random-DebertaV2Model"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :deberta_v2} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[1, 4144, 4341, 19, 128_000, 65, 2], [1, 7063, 44, 2, 9881, 2, 0]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0]])
    )
  end

  test ":albert" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "albert/albert-base-v2"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :albert} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [2, 1289, 5123, 29, 4, 13, 9, 3],
        [2, 1301, 60, 3, 1623, 3, 0, 0]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0]
      ])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0]
      ])
    )
  end

  test ":bart" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/bart-base"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :bart} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [0, 34603, 3645, 19, 646, 32804, 530, 8174, 2],
        [0, 45641, 116, 2, 2, 33683, 2, 1, 1]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0]
      ])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
      ])
    )
  end

  test ":bert" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-uncased"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :bert} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [101, 3231, 6251, 2007, 103, 1012, 102],
        [101, 3160, 1029, 102, 3437, 102, 0]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0]
      ])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0]
      ])
    )
  end

  test ":camembert" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "almanach/camembert-base"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :camembert} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with <mask>."
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [5, 9115, 22625, 1466, 32004, 21, 9, 6]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1]
      ])
    )
  end

  test ":clip" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-base-patch32"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :clip} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "a photo of a cat",
        "a photo of a dog"
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [49406, 320, 1125, 539, 320, 2368, 49407],
        [49406, 320, 1125, 539, 320, 1929, 49407]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
      ])
    )
  end

  test ":code_gen" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "microsoft/phi-2"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :code_gen} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello everyobdy, how are you?"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[15496, 790, 672, 9892, 11, 703, 389, 345, 30]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
    )
  end

  test ":distilbert" do
    assert {:ok, tokenizer} =
             Bumblebee.load_tokenizer({:hf, "distilbert/distilbert-base-uncased"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :distilbert} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [101, 3231, 6251, 2007, 103, 1012, 102],
        [101, 3160, 1029, 102, 3437, 102, 0]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0]
      ])
    )
  end

  test ":gemma" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/gemma-7b-it"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :gemma} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello World"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[2, 4521, 3855]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1]])
    )
  end

  test ":gpt_neo_x" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "EleutherAI/gpt-neox-20b"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :gpt_neo_x} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello World"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[12092, 3645]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1]])
    )
  end

  test ":gpt2" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :gpt2} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello World"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [15496, 2159]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1]
      ])
    )
  end

  test ":layout_lm" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "microsoft/layoutlm-base-uncased"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :layout_lm} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with [MASK].",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[101, 3231, 6251, 2007, 103, 1012, 102], [101, 3160, 1029, 102, 3437, 102, 0]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0]])
    )
  end

  test ":llama" do
    assert {:ok, tokenizer} =
             Bumblebee.load_tokenizer({:hf, "hf-internal-testing/llama-tokenizer"},
               type: :llama
             )

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :llama} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello everyobdy, how are you?"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[1, 15043, 1432, 711, 4518, 29892, 920, 526, 366, 29973]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    )
  end

  test ":mbart" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/mbart-large-cc25"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :mbart} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello, my dog is cute <mask>"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [35378, 4, 759, 10269, 83, 99942, 250_026, 2, 250_004]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
      ])
    )
  end

  test ":nllb" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "facebook/nllb-200-distilled-600M"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :nllb} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello, my dog is cute <mask>"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [256_047, 94124, 248_079, 1537, 6658, 248, 95740, 256_203, 2]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
      ])
    )

    tokenizer =
      Bumblebee.configure(tokenizer, template_options: [language_token: "fra_Latn"])

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello, my dog is cute <mask>"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [256_057, 94124, 248_079, 1537, 6658, 248, 95740, 256_203, 2]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
      ])
    )
  end

  test ":mpnet" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "microsoft/mpnet-base"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :mpnet} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with <mask>.",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [0, 3235, 6255, 2011, 30526, 1016, 2],
        [0, 3164, 1033, 2, 2, 3441, 2]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
      ])
    )
  end

  test ":roberta" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "FacebookAI/roberta-base"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :roberta} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with <mask>.",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [0, 34603, 3645, 19, 50264, 4, 2],
        [0, 45641, 116, 2, 2, 33683, 2]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
      ])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
      ])
    )
  end

  test ":siglip" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google/siglip-base-patch16-224"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :siglip} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, ["a photo of a cat", "Two dogs playing in the snow."])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [262, 266, 1304, 267, 262, 266, 2783, 1],
        [395, 3014, 1376, 268, 260, 2468, 1, 1]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0]])
    )
  end

  test ":smollm3" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "HuggingFaceTB/SmolLM3-3B"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :smollm3} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with <mask>.",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [2323, 11914, 449, 366, 11508, 14611],
        [14924, 30, 16533, 128_012, 128_012, 128_012]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    )
  end

  test ":t5" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-t5/t5-small"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :t5} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, ["translate English to German: How old are you?"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[13959, 1566, 12, 2968, 10, 571, 625, 33, 25, 58, 1]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    )
  end

  test ":whisper" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/whisper-tiny"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :whisper} = tokenizer

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Hello world"])

    assert_equal(inputs["input_ids"], Nx.tensor([[50258, 50363, 15947, 1002, 50257]]))
    assert_equal(inputs["attention_mask"], Nx.tensor([[1, 1, 1, 1, 1]]))
  end

  test ":xlm_roberta" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "FacebookAI/xlm-roberta-base"})

    assert %Bumblebee.Text.PreTrainedTokenizer{type: :xlm_roberta} = tokenizer

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, [
        "Test sentence with <mask>.",
        {"Question?", "Answer"}
      ])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([[0, 8647, 149_357, 678, 250_001, 6, 5, 2], [0, 68185, 32, 2, 2, 130_373, 2, 1]])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0]])
    )

    assert_equal(
      inputs["token_type_ids"],
      Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    )
  end

  test "pads and truncates to :length" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-cased"})

    tokenizer = Bumblebee.configure(tokenizer, length: 6)

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["foo", "foo bar", "foo bar baz bang buzz"])

    assert_equal(
      inputs["input_ids"],
      Nx.tensor([
        [101, 175, 5658, 102, 0, 0],
        [101, 175, 5658, 2927, 102, 0],
        [101, 175, 5658, 2927, 171, 102]
      ])
    )

    assert_equal(
      inputs["attention_mask"],
      Nx.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    )
  end

  test "encoding with special tokens mask" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-cased"})

    tokenizer = Bumblebee.configure(tokenizer, return_special_tokens_mask: true)

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Test sentence with [MASK]."])

    assert_equal(inputs["special_tokens_mask"], Nx.tensor([[1, 0, 0, 0, 0, 0, 1]]))
  end

  test "encoding with offsets" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-cased"})

    tokenizer = Bumblebee.configure(tokenizer, return_offsets: true)

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["Test sentence with [MASK]."])

    assert_equal(inputs["start_offsets"], Nx.tensor([[0, 0, 5, 14, 19, 25, 0]]))
    assert_equal(inputs["end_offsets"], Nx.tensor([[0, 4, 13, 18, 25, 26, 0]]))
  end

  test "encoding with multiple lengths" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-cased"})

    tokenizer = Bumblebee.configure(tokenizer, length: [8, 16])

    inputs = Bumblebee.apply_tokenizer(tokenizer, "This is short.")

    assert {1, 8} = Nx.shape(inputs["input_ids"])

    inputs =
      Bumblebee.apply_tokenizer(tokenizer, "This is definitely much longer than the above.")

    assert {1, 16} = Nx.shape(inputs["input_ids"])
  end

  test "adds template tokens when the sequence is truncated" do
    assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "google-bert/bert-base-cased"})

    tokenizer = Bumblebee.configure(tokenizer, length: 5)

    inputs = Bumblebee.apply_tokenizer(tokenizer, ["This is a long test sentence."])

    assert_equal(inputs["input_ids"], Nx.tensor([[101, 1188, 1110, 170, 102]]))
    assert_equal(inputs["attention_mask"], Nx.tensor([[1, 1, 1, 1, 1]]))
  end

  describe "chat template" do
    test "renders the template from tokenizer_config.json" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/gemma-7b-it"})

      messages = [
        %{"role" => "user", "content" => "What is the capital of Brazil?"},
        %{"role" => "assistant", "content" => "Brasília."},
        %{"role" => "user", "content" => "And of France?"}
      ]

      assert Bumblebee.apply_chat_template(tokenizer, messages, add_generation_prompt: true) ==
               "<bos><start_of_turn>user\nWhat is the capital of Brazil?<end_of_turn>\n" <>
                 "<start_of_turn>model\nBrasília.<end_of_turn>\n" <>
                 "<start_of_turn>user\nAnd of France?<end_of_turn>\n<start_of_turn>model\n"

      messages = [%{"role" => "user", "content" => "Hi"}]

      assert Bumblebee.apply_chat_template(tokenizer, messages) ==
               "<bos><start_of_turn>user\nHi<end_of_turn>\n"
    end

    test "continues the final message" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/gemma-7b-it"})

      messages = [
        %{"role" => "user", "content" => "Write a haiku."},
        %{"role" => "assistant", "content" => "Processes awake,"}
      ]

      assert Bumblebee.apply_chat_template(tokenizer, messages, continue_final_message: true) ==
               "<bos><start_of_turn>user\nWrite a haiku.<end_of_turn>\n" <>
                 "<start_of_turn>model\nProcesses awake,"
    end

    test "encodes the prompt without adding special tokens twice" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/gemma-7b-it"})

      messages = [%{"role" => "user", "content" => "Hi"}]
      prompt = Bumblebee.apply_chat_template(tokenizer, messages, add_generation_prompt: true)

      tokenizer = Bumblebee.configure(tokenizer, add_special_tokens: false)
      inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)

      assert_equal(
        inputs["input_ids"],
        Nx.tensor([[2, 106, 1645, 108, 2151, 107, 108, 106, 2516, 108]])
      )
    end

    test "renders the template from chat_template.jinja" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "HuggingFaceTB/SmolLM3-3B"})

      assert tokenizer.chat_template =~ "{%- if enable_thinking is not defined -%}"

      messages = [%{"role" => "user", "content" => "What is the capital of Brazil?"}]

      prompt = Bumblebee.apply_chat_template(tokenizer, messages, add_generation_prompt: true)
      assert prompt =~ "Reasoning Mode: /think\n"

      assert String.ends_with?(
               prompt,
               "<|im_start|>user\nWhat is the capital of Brazil?<|im_end|>\n<|im_start|>assistant\n"
             )

      prompt =
        Bumblebee.apply_chat_template(tokenizer, messages,
          add_generation_prompt: true,
          variables: %{"enable_thinking" => false}
        )

      assert prompt =~ "Reasoning Mode: /no_think\n"
    end

    test "passes special tokens the tokenizer type does not track" do
      assert {:ok, tokenizer} =
               Bumblebee.load_tokenizer({:hf, "mistralai/Mistral-7B-Instruct-v0.3"})

      messages = [%{"role" => "user", "content" => "What is the capital of Brazil?"}]

      assert Bumblebee.apply_chat_template(tokenizer, messages) ==
               "<s>[INST] What is the capital of Brazil?[/INST]"
    end

    test "selects among named templates" do
      assert {:ok, tokenizer} =
               Bumblebee.load_tokenizer({:hf, "NousResearch/Hermes-2-Pro-Llama-3-8B"})

      assert %{"default" => _, "tool_use" => _} = tokenizer.chat_template

      messages = [%{"role" => "user", "content" => "Hi"}]

      tools = [
        %{
          "type" => "function",
          "function" => %{
            "name" => "get_weather",
            "description" => "Get the weather.",
            "parameters" => %{
              "type" => "object",
              "properties" => %{"city" => %{"type" => "string"}},
              "required" => ["city"]
            }
          }
        }
      ]

      default_prompt =
        "<|begin_of_text|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"

      assert Bumblebee.apply_chat_template(tokenizer, messages, add_generation_prompt: true) ==
               default_prompt

      prompt =
        Bumblebee.apply_chat_template(tokenizer, messages,
          add_generation_prompt: true,
          tools: tools
        )

      assert prompt =~ "<|im_start|>system\nYou are a function calling AI model."
      assert prompt =~ "\"name\": \"get_weather\""

      assert Bumblebee.apply_chat_template(tokenizer, messages,
               add_generation_prompt: true,
               tools: tools,
               chat_template: "default"
             ) == default_prompt

      assert_raise ArgumentError,
                   ~s/no chat template named "rag", available: "default", "tool_use"/,
                   fn ->
                     Bumblebee.apply_chat_template(tokenizer, messages, chat_template: "rag")
                   end
    end

    test "leaves out message keys set to nil" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/gemma-7b-it"})

      messages = [
        %{"role" => "user", "content" => "Hi"},
        %{"role" => "assistant", "content" => nil}
      ]

      assert Bumblebee.apply_chat_template(tokenizer, messages) ==
               "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n<end_of_turn>\n"
    end

    test "gives the reasoning under each of its names" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})

      messages = [
        %{"role" => "user", "content" => "Is 91 prime?"},
        %{"role" => "assistant", "content" => "No.", "thinking" => "91 = 7 * 13."},
        %{"role" => "user", "content" => "And 97?"},
        %{"role" => "assistant", "content" => "Yes.", "thinking" => "No divisors."}
      ]

      assert Bumblebee.apply_chat_template(tokenizer, messages) ==
               "<|im_start|>user\nIs 91 prime?<|im_end|>\n<|im_start|>assistant\nNo.<|im_end|>\n" <>
                 "<|im_start|>user\nAnd 97?<|im_end|>\n" <>
                 "<|im_start|>assistant\n<think>\nNo divisors.\n</think>\n\nYes.<|im_end|>\n"
    end

    test "raises when the repository has no chat template" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai-community/gpt2"})

      assert tokenizer.chat_template == nil

      assert_raise ArgumentError, ~r/the tokenizer has no chat template/, fn ->
        Bumblebee.apply_chat_template(tokenizer, [%{"role" => "user", "content" => "Hi"}])
      end
    end

    test "raises the template error" do
      assert {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "unsloth/gemma-7b-it"})

      messages = [
        %{"role" => "system", "content" => "Be terse."},
        %{"role" => "user", "content" => "Hi"}
      ]

      assert_raise Otera.Error, ~r/System role not supported/, fn ->
        Bumblebee.apply_chat_template(tokenizer, messages)
      end
    end
  end
end
