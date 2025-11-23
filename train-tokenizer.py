#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a custom tokenizer from the datasets specified in config.yaml
"""
import os
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable must be set")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

datasets_config = config["pretrain"]["dataset"]

# Tokenizer training parameters
VOCAB_SIZE = 32000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>"]
OUTPUT_DIR = "custom-tokenizer"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[*] Training custom tokenizer")
print(f"   Vocab size: {VOCAB_SIZE:,}")
print(f"   Output directory: {OUTPUT_DIR}")
print()

# Initialize a BPE tokenizer (similar to LLaMA/GPT)
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# Set pre-tokenizer (splits on whitespace and punctuation)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Set decoder
tokenizer.decoder = decoders.ByteLevel()

# Configure trainer
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
)

# Create an iterator that yields text from all datasets
def text_iterator():
    """Yield text from all datasets in config"""
    for ds_config in datasets_config:
        ds_name = ds_config["name"]
        ds_rows = ds_config.get("rows", None)
        
        # Support "All" as a string value
        if isinstance(ds_rows, str) and ds_rows.lower() == "all":
            ds_rows = None
        
        print(f"[*] Loading {ds_name} (rows: {ds_rows if ds_rows is not None else 'all'})")
        
        stream = load_dataset(
            ds_name,
            split="train",
            streaming=True,
            token=HF_TOKEN,
        )
        
        if ds_rows is not None:
            stream = stream.take(ds_rows)
        
        count = 0
        for example in stream:
            text = (example.get("text") or "").strip()
            if text:
                yield text
                count += 1
                
                # Progress update every 10k examples
                if count % 10000 == 0:
                    print(f"   Processed {count:,} examples from {ds_name}")
        
        print(f"   Completed {ds_name}: {count:,} examples")

print("[*] Training tokenizer on datasets...")
print()

# Train the tokenizer
tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

# Configure post-processor for proper formatting
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# Save the tokenizer
tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
tokenizer.save(tokenizer_path)
print(f"\n[OK] Saved tokenizer to {tokenizer_path}")

# Convert to HuggingFace format
print("[*] Converting to HuggingFace format...")
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
)

# Save in HuggingFace format
hf_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[OK] Saved HuggingFace tokenizer to {OUTPUT_DIR}")

# Test the tokenizer
print("\n[*] Testing tokenizer...")
test_texts = [
    "Hello, world! This is a test.",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "int main() { printf(\"Hello\\n\"); return 0; }",
]

for text in test_texts:
    tokens = hf_tokenizer.tokenize(text)
    ids = hf_tokenizer.encode(text)
    print(f"\nText: {text[:50]}...")
    print(f"Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    print(f"IDs ({len(ids)}): {ids[:10]}{'...' if len(ids) > 10 else ''}")

print(f"\n[OK] Tokenizer training complete!")
print(f"   Vocabulary size: {hf_tokenizer.vocab_size:,}")
print(f"   To use this tokenizer, update config.yaml:")
print(f'   tokenizer: "{OUTPUT_DIR}"')
