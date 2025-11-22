''' Goal

Currently we have the following datasets:
ThomasTheMaker/arc-stack-javascript
ThomasTheMaker/arc-stack-python
ThomasTheMaker/arc-stack-c
ThomasTheMaker/arc-stack-cpp

Make each ofthem into a new dataset with only 2 columns:
- 'text' which is the 'content' column
- 'size' which is the size column
- 'token_count' which is the token count of the 'text' column

Data should be streamed as the code runs, not after the code has finished running.

We should have 4 datasets in total at the end.

Print out ETA.

'''

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import sys
from multiprocessing import Pool, cpu_count
import os

BATCH_SIZE = 10000
NUM_WORKERS = cpu_count()

# Global tokenizer for multiprocessing workers
tokenizer = None

def init_worker():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def tokenize_example(example):
    global tokenizer
    text = example.get('content', '')
    size = example.get('size', len(text))
    token_count = len(tokenizer.encode(text, truncation=False))
    return {
        'text': text,
        'size': size,
        'token_count': token_count
    }

DATASETS = {
    "js": "ThomasTheMaker/arc-stack-javascript",
    "python": "ThomasTheMaker/arc-stack-python",
    "c": "ThomasTheMaker/arc-stack-c",
    "cpp": "ThomasTheMaker/arc-stack-cpp",
}

def process_dataset(dataset_name):
    print(f"\nProcessing {dataset_name} using {NUM_WORKERS} CPU cores...")

    # Load dataset in streaming mode
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    output_name = dataset_name.split('/')[-1] + "-processed.parquet"

    # Define schema for parquet
    schema = pa.schema([
        ('text', pa.string()),
        ('size', pa.int64()),
        ('token_count', pa.int64())
    ])

    writer = None
    batch_buffer = []
    total_rows = 0
    start_time = time.time()

    # Get approximate total for ETA (if available)
    try:
        total = dataset.info.splits['train'].num_examples
    except:
        total = None

    # Create process pool
    pool = Pool(processes=NUM_WORKERS, initializer=init_worker)

    pbar = tqdm(dataset, total=total, desc=dataset_name.split('/')[-1])

    for i, example in enumerate(pbar):
        batch_buffer.append(example)

        # Process batch in parallel
        if len(batch_buffer) >= BATCH_SIZE:
            processed_batch = pool.map(tokenize_example, batch_buffer)

            table = pa.Table.from_pylist(processed_batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_name, schema)
            writer.write_table(table)
            total_rows += len(processed_batch)
            batch_buffer = []

        # Update ETA every 1000 examples
        if (i + 1) % 1000 == 0 and total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            pbar.set_postfix({'ETA': f'{remaining/60:.1f}min'})

    # Process remaining data
    if batch_buffer:
        processed_batch = pool.map(tokenize_example, batch_buffer)
        table = pa.Table.from_pylist(processed_batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output_name, schema)
        writer.write_table(table)
        total_rows += len(processed_batch)

    pool.close()
    pool.join()

    if writer:
        writer.close()

    print(f"\nSaved {output_name} with {total_rows} examples")

    return output_name

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        print("Usage: python create_corpus.py <language>")
        print("Languages: c, cpp, js, python")
        sys.exit(1)

    lang = sys.argv[1]
    dataset_name = DATASETS[lang]

    print(f"Processing {dataset_name}...")
    result = process_dataset(dataset_name)
    print(f"\nDone: {result}")

if __name__ == "__main__":
    main()
