# arc-lm
Training a language model with chinchila 20:1 ratio

## Converting filtered JSON dumps to Parquet

The `data/convert.py` helper streams the large JSON array files under `data/` and writes
compressed Parquet shards that Hugging Face can index without exhausting memory.

```bash
pip install pyarrow tqdm
python data/convert.py --output-dir output/parquet
```

Command-line flags let you override the input files, target directory, maximum rows
per parquet shard, and the batch size used while streaming.
