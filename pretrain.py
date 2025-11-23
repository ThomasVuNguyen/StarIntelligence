# -*- coding: utf-8 -*-
import os
import math
import time
import random
import yaml
import shutil
import csv
from itertools import islice, chain

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable must be set")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

pretrain_config = config["pretrain"]
model_config = pretrain_config["model"]["config"]
datasets_config = pretrain_config["dataset"]
training_config = pretrain_config["training"]
model_name = pretrain_config["model"]["name"]

TOKENIZER_NAME = training_config["tokenizer"]
OUTPUT_DIR = os.path.join("output", model_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy config to output directory
shutil.copy("config.yaml", os.path.join(OUTPUT_DIR, "config.yaml"))
print(f"[*] Output directory: {OUTPUT_DIR}")

BLOCK_SIZE = model_config.get("max_position_embeddings", 2048)
BATCH_SIZE = training_config["batch_size"]
GRAD_ACCUM_STEPS = training_config["grad_accum_steps"]
NUM_EPOCHS = training_config["num_epochs"]
LEARNING_RATE = float(training_config["learning_rate"])
WEIGHT_DECAY = training_config["weight_decay"]
WARMUP_RATIO = training_config["warmup_ratio"]
GRAD_CLIP = training_config["grad_clip"]
LOG_EVERY = training_config["log_every"]
SAVE_EVERY = training_config["save_every"]
RANDOM_SEED = training_config["random_seed"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

print("[*] Loading tokenizer:", TOKENIZER_NAME)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}

to_add = {k: v for k, v in special_tokens.items() if getattr(tokenizer, k, None) is None}
if to_add:
    print("[+] Adding special tokens:", to_add)
    tokenizer.add_special_tokens(to_add)

pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

print(f"[OK] Tokenizer vocab size: {len(tokenizer)}")
print(f"   pad_id={pad_id}, bos_id={bos_id}, eos_id={eos_id}")
print()

def ensure_text(example):
    content = (example.get("text") or "").strip()
    if not content:
        content = "No content provided."
    return {"text": content}

# Load all datasets from config
print("[*] Loading datasets from config...")
all_streams = []
total_rows = 0

for ds_config in datasets_config:
    ds_name = ds_config["name"]
    ds_rows = ds_config.get("rows", None)
    
    # Support "All" as a string value to explicitly train on all rows
    if isinstance(ds_rows, str) and ds_rows.lower() == "all":
        ds_rows = None
    
    print(f"   Loading {ds_name} (rows: {ds_rows if ds_rows is not None else 'all'})")

    stream = load_dataset(
        ds_name,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    if ds_rows is not None:
        stream = stream.take(ds_rows)
        total_rows += ds_rows

    all_streams.append(stream.map(ensure_text))

def combined_stream(streams):
    for stream in streams:
        for item in stream:
            yield item

# Estimate tokens
print("[*] Estimating dataset size...")
sample_size = min(1000, total_rows if total_rows > 0 else 1000)
sample_tokens = 0

temp_streams = []
for ds_config in datasets_config:
    ds_name = ds_config["name"]
    ds_rows = ds_config.get("rows", None)
    
    # Support "All" as a string value
    if isinstance(ds_rows, str) and ds_rows.lower() == "all":
        ds_rows = None
    
    stream = load_dataset(ds_name, split="train", streaming=True, token=HF_TOKEN)
    if ds_rows is not None:
        stream = stream.take(ds_rows)
    temp_streams.append(stream.map(ensure_text))

temp_combined = combined_stream(temp_streams)
for i, ex in enumerate(islice(temp_combined, sample_size)):
    text = ex["text"]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    sample_tokens += len(ids) + 1

avg_tokens_per_doc = sample_tokens / max(sample_size, 1)
print(f"   Sampled {sample_size} documents, avg {avg_tokens_per_doc:.1f} tokens/doc")

TOKENS_PER_STEP = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS

if total_rows > 0:
    # Known dataset size - can estimate steps accurately
    estimated_tokens = int(total_rows * avg_tokens_per_doc)
    TOTAL_STEPS = max(1, (estimated_tokens * NUM_EPOCHS) // TOKENS_PER_STEP)
    print(f"   Using {total_rows:,} documents")
    print(f"   Estimated total tokens: {estimated_tokens:,}")
    print(f"[*] Training for approximately {TOTAL_STEPS:,} steps ({NUM_EPOCHS} epoch(s))")
    print(f"   Tokens per step: {TOKENS_PER_STEP:,}")
else:
    # Unknown dataset size - train until exhaustion
    # Use a large estimate for scheduler warmup calculation
    estimated_tokens = int(sample_size * avg_tokens_per_doc * 100)  # Conservative multiplier
    TOTAL_STEPS = max(1, (estimated_tokens * NUM_EPOCHS) // TOKENS_PER_STEP)
    print(f"   Dataset size unknown - will train until data exhaustion")
    print(f"   Estimated avg tokens/doc: {avg_tokens_per_doc:.1f}")
    print(f"[*] Training for {NUM_EPOCHS} epoch(s) until data exhaustion")
    print(f"   Tokens per step: {TOKENS_PER_STEP:,}")
    print(f"   Using conservative estimate of ~{TOTAL_STEPS:,} steps for scheduler")
print()

# Build model from config
llama_config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=model_config["hidden_size"],
    intermediate_size=model_config["intermediate_size"],
    num_hidden_layers=model_config["num_hidden_layers"],
    num_attention_heads=model_config["num_attention_heads"],
    num_key_value_heads=model_config.get("num_key_value_heads", model_config["num_attention_heads"] // 3),
    max_position_embeddings=model_config["max_position_embeddings"],
    rms_norm_eps=float(model_config["rms_norm_eps"]),
    initializer_range=model_config["initializer_range"],
    use_cache=model_config["use_cache"],
    pad_token_id=pad_id,
    bos_token_id=bos_id,
    eos_token_id=eos_id,
    tie_word_embeddings=False,
)

print("[*] Building model...")
model = LlamaForCausalLM(llama_config)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and (not use_bf16)

if use_bf16:
    dtype = torch.bfloat16
elif use_fp16:
    dtype = torch.float16
else:
    dtype = torch.float32

model = model.to(device, dtype=dtype)

print(
    f"[OK] Model ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
    f"dtype={dtype}, device={device}"
)
print()

def token_block_stream(hf_stream, tokenizer, block_size, eos_id):
    buffer = []

    for ex in hf_stream:
        text = ex["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eos_id)
        buffer.extend(ids)

        while len(buffer) >= block_size:
            block = buffer[:block_size]
            buffer = buffer[block_size:]
            yield torch.tensor(block, dtype=torch.long)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)

num_warmup_steps = int(TOTAL_STEPS * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=TOTAL_STEPS,
)

scaler = GradScaler(enabled=use_fp16)

print("[*] Starting pretraining...")
print(
    f"   BLOCK_SIZE={BLOCK_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
    f"GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}"
)
print(
    f"   Effective tokens/step = {BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS:,}"
)
print(f"   Learning rate: {LEARNING_RATE}, Warmup steps: {num_warmup_steps}")
if total_rows == 0:
    print(f"   Will train until data is exhausted (no step limit)")
print()

global_step = 0
micro_step = 0
running_loss = 0.0
start_time = time.time()
window_start_time = time.time()
window_start_step = 0

# Initialize CSV logging
csv_path = os.path.join(OUTPUT_DIR, "training_log.csv")
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["step", "loss", "learning_rate", "tokens_per_sec", "elapsed_time"])
csv_file.flush()

# Track all metrics for plotting
all_steps = []
all_losses = []
all_lrs = []
all_tok_per_sec = []

def multi_epoch_stream(datasets_config, num_epochs):
    for epoch in range(num_epochs):
        print(f"[*] Starting epoch {epoch + 1}/{num_epochs}")
        streams = []
        for ds_config in datasets_config:
            ds_name = ds_config["name"]
            ds_rows = ds_config.get("rows", None)
            
            # Support "All" as a string value
            if isinstance(ds_rows, str) and ds_rows.lower() == "all":
                ds_rows = None
            
            stream = load_dataset(ds_name, split="train", streaming=True, token=HF_TOKEN)
            if ds_rows is not None:
                stream = stream.take(ds_rows)
            streams.append(stream.map(ensure_text))

        row_count = 0
        for item in combined_stream(streams):
            yield item
            row_count += 1
        print(f"   Processed {row_count:,} rows in epoch {epoch + 1}")

multi_epoch_data = multi_epoch_stream(datasets_config, NUM_EPOCHS)
block_iter = token_block_stream(multi_epoch_data, tokenizer, BLOCK_SIZE, eos_id)

model.train()

# Always use TOTAL_STEPS for progress bar to show ETA
# When training to exhaustion, this is a conservative estimate
pbar = tqdm(total=TOTAL_STEPS, desc="Training", unit="step")

autocast_ctx = autocast(enabled=(use_bf16 or use_fp16), dtype=torch.bfloat16 if use_bf16 else torch.float16)
with autocast_ctx:
    while True:  # Train until data exhaustion
        blocks = []
        for _ in range(BATCH_SIZE):
            try:
                block = next(block_iter)
                blocks.append(block)
            except StopIteration:
                print(f"\n[OK] Dataset exhausted after {global_step} steps")
                break

        if len(blocks) < BATCH_SIZE:
            if len(blocks) > 0:
                print(f"   Completed training with partial batch of {len(blocks)} blocks")
            break

        input_ids = torch.stack(blocks).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / GRAD_ACCUM_STEPS

        if use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item()
        micro_step += 1

        if micro_step % GRAD_ACCUM_STEPS == 0:
            if use_fp16:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            pbar.update(1)

            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                current_lr = scheduler.get_last_lr()[0]

                window_elapsed = time.time() - window_start_time
                window_steps = global_step - window_start_step
                tok_per_step = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS
                window_tps = (tok_per_step * window_steps) / window_elapsed if window_elapsed > 0 else 0

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "tok/s": f"{int(window_tps):,}"
                })

                # Log to CSV
                elapsed_total = time.time() - start_time
                csv_writer.writerow([global_step, avg_loss, current_lr, window_tps, elapsed_total])
                csv_file.flush()
                
                # Track for plotting
                all_steps.append(global_step)
                all_losses.append(avg_loss)
                all_lrs.append(current_lr)
                all_tok_per_sec.append(window_tps)

                running_loss = 0.0
                window_start_time = time.time()
                window_start_step = global_step

            if global_step % SAVE_EVERY == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                print(f"\n[*] Saving checkpoint to {ckpt_dir}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

                torch.save({
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
                }, os.path.join(ckpt_dir, "training_state.pt"))
                
                # Generate training plots
                print("[*] Generating training plots...")
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Loss plot
                axes[0, 0].plot(all_steps, all_losses, linewidth=2, color='#2E86AB')
                axes[0, 0].set_xlabel('Step', fontsize=12)
                axes[0, 0].set_ylabel('Loss', fontsize=12)
                axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Learning rate plot
                axes[0, 1].plot(all_steps, all_lrs, linewidth=2, color='#A23B72')
                axes[0, 1].set_xlabel('Step', fontsize=12)
                axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
                axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                
                # Tokens per second plot
                axes[1, 0].plot(all_steps, all_tok_per_sec, linewidth=2, color='#F18F01')
                axes[1, 0].set_xlabel('Step', fontsize=12)
                axes[1, 0].set_ylabel('Tokens/sec', fontsize=12)
                axes[1, 0].set_title('Training Throughput', fontsize=14, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Loss (log scale) plot
                axes[1, 1].plot(all_steps, all_losses, linewidth=2, color='#2E86AB')
                axes[1, 1].set_xlabel('Step', fontsize=12)
                axes[1, 1].set_ylabel('Loss (log scale)', fontsize=12)
                axes[1, 1].set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(OUTPUT_DIR, f"training_plot_step_{global_step}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[OK] Saved training plot to {plot_path}")

pbar.close()
csv_file.close()

print("\n[OK] Training complete!")
print("[*] Saving final model...")

final_dir = os.path.join(OUTPUT_DIR, "final-model")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

torch.save({
    'global_step': global_step,
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
}, os.path.join(final_dir, "training_state.pt"))

# Generate final training plot
print("[*] Generating final training plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss plot
axes[0, 0].plot(all_steps, all_losses, linewidth=2, color='#2E86AB')
axes[0, 0].set_xlabel('Step', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Learning rate plot
axes[0, 1].plot(all_steps, all_lrs, linewidth=2, color='#A23B72')
axes[0, 1].set_xlabel('Step', fontsize=12)
axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Tokens per second plot
axes[1, 0].plot(all_steps, all_tok_per_sec, linewidth=2, color='#F18F01')
axes[1, 0].set_xlabel('Step', fontsize=12)
axes[1, 0].set_ylabel('Tokens/sec', fontsize=12)
axes[1, 0].set_title('Training Throughput', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Loss (log scale) plot
axes[1, 1].plot(all_steps, all_losses, linewidth=2, color='#2E86AB')
axes[1, 1].set_xlabel('Step', fontsize=12)
axes[1, 1].set_ylabel('Loss (log scale)', fontsize=12)
axes[1, 1].set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
final_plot_path = os.path.join(OUTPUT_DIR, "training_plot_final.png")
plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved final training plot to {final_plot_path}")
print(f"[OK] Training log saved to {csv_path}")

print("[OK] Done!")
