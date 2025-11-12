import os
from pathlib import Path

from datasets import load_dataset

def load_hf_token():
    """
    Pull HF_TOKEN from the repository-level .env so pushing to the Hub works
    without hardcoding secrets in source.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    hf_token = None
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not hf_token:
        raise ValueError("Set HF_TOKEN in the repo .env file before running this script.")
    os.environ.setdefault("HF_TOKEN", hf_token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    return hf_token


HF_TOKEN = load_hf_token()

dataset = load_dataset("ThomasTheMaker/Arc-Corpus-sample", split="train")  # load the dataset
dataset = dataset.select(range(20))  # keep only 20 rows for quick tokenizer runs

def format_prompts(examples):
    """
    Wrap raw dataset rows into the chat format expected by the tokenizer/model.
    """
    formatted = []
    for text in examples["text"]:
        content = (text or "").strip()
        if not content:
            content = "No content provided."
        conversation = (
            "<|user|>\n"
            "Share a helpful continuation for the following document.\n"
            "<|end|>\n"
            "<|bot|>\n"
            f"{content}\n"
            "<|end|>"
        )
        formatted.append(conversation)
    return {"text": formatted}

dataset = dataset.map(format_prompts, batched=True)

print(dataset["text"][2])  # sanity check

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

training_corpus = get_training_corpus()

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
)
from trl import SFTTrainer, SFTConfig

TINYLLAMA_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_VOCAB_SIZE = 128_256  # match tinyllama config

base_tokenizer = AutoTokenizer.from_pretrained(
    TINYLLAMA_TOKENIZER,
    use_fast=True,
)

# Re-train the TinyLlama tokenizer on the Arc corpus so it stays architecture-compatible.
tokenizer = base_tokenizer.train_new_from_iterator(
    training_corpus,
    vocab_size=TARGET_VOCAB_SIZE,
)

if not isinstance(tokenizer, PreTrainedTokenizerFast):
    raise ValueError("Expected a fast tokenizer to attach chat template metadata.")

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "additional_special_tokens": ["<|user|>", "<|bot|>", "<|end|>"],
}
tokenizer.add_special_tokens(special_tokens)

tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")

chat_template = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|bot|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
    "{{ eos_token }}"
)
tokenizer.chat_template = chat_template

if tokenizer.pad_token_id is None:
    raise ValueError("Tokenizer is missing a pad token ID after special token setup.")

effective_vocab_size = len(tokenizer)

print(
    tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Why is the sky blue?"},
            {"role": "assistant", "content": "Due to Rayleigh scattering."},
        ],
        tokenize=False,
    )
)

# Configure ~800M parameter TinyLlama variant (embedding + transformer blocks).
config = LlamaConfig(
    vocab_size=effective_vocab_size,
    hidden_size=1536,
    intermediate_size=4096,
    num_hidden_layers=22,
    num_attention_heads=12,
    num_key_value_heads=3,
    max_position_embeddings=4096,
    rms_norm_eps=1.0e-6,
    initializer_range=0.02,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=False,
)

model = LlamaForCausalLM(config)

sft_config = SFTConfig(
    output_dir="output",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    optim="sgd",
    dataset_text_field="text",
    max_length=512,
    push_to_hub=True,
    hub_model_id="ThomasTheMaker/Arc",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    processing_class=tokenizer,
    train_dataset=dataset,
)

trainer.train()

tokenizer.save_pretrained("tokenizers/tinyllama")

trainer.push_to_hub(
    commit_message="Initial TinyLlama SFT run",
    token=HF_TOKEN,
)
