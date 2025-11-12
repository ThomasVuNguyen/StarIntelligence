# Stream dataset ThomasTheMaker/arc-stack-c & ThomasTheMaker/arc-stack-cpp
# Check for the 'content' column for these following string matches cad, mesh, geometry, render, object, shape, vector, model, transform, 3d, solid
# Note that the match must be exact, not partial ('caddoium' should not match 'cad')
# Add in a new column to show the word matches as a list of strings.
# Make sure the data is added to json as the code runs, not after the code has finished running.
# This computer has lots of CPU cores, use them to go brrr and as fast as possible
# Output results to 2 json files.
# Ensure to show ETA.
from __future__ import annotations

import json
import os
import re
import sys
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Deque, Iterable, Iterator, List, MutableMapping, Sequence, Tuple

from datasets import get_dataset_config_info, load_dataset
from tqdm.auto import tqdm


KEYWORDS: Tuple[str, ...] = (
    "cad",
    "mesh",
    "geometry",
    "render",
    "object",
    "shape",
    "vector",
    "model",
    "transform",
    "3d",
    "solid",
)
KEYWORD_PATTERN = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)
KEYWORD_PATTERNS = {kw: re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in KEYWORDS}
DATASETS = ("ThomasTheMaker/arc-stack-c", "ThomasTheMaker/arc-stack-cpp")
DEFAULT_CONFIG = "default"
BATCH_SIZE = 4096
MAX_WORKERS = max(4, os.cpu_count() or 4)
PENDING_LIMIT = MAX_WORKERS * 4


def chunked(iterable: Iterable[MutableMapping[str, object]], size: int) -> Iterator[List[MutableMapping[str, object]]]:
    batch: List[MutableMapping[str, object]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_batch(batch: Sequence[MutableMapping[str, object]]) -> List[MutableMapping[str, object]]:
    matched_records: List[MutableMapping[str, object]] = []
    for record in batch:
        content = record.get("content")
        if not isinstance(content, str) or not KEYWORD_PATTERN.search(content):
            continue
        matches = [kw for kw, pattern in KEYWORD_PATTERNS.items() if pattern.search(content)]
        if not matches:
            continue
        enriched = dict(record)
        enriched["matches"] = matches
        matched_records.append(enriched)
    return matched_records


def resolve_split(dataset_name: str, config_name: str) -> Tuple[str, int | None]:
    info = get_dataset_config_info(dataset_name, config_name)
    if not info.splits:
        raise ValueError(f"No splits found for {dataset_name}:{config_name}")
    split_name = "train" if "train" in info.splits else next(iter(info.splits))
    total = info.splits[split_name].num_examples
    return split_name, total


def dataset_to_filename(dataset_name: str) -> Path:
    return Path(f"{dataset_name.split('/')[-1]}_matches.json")


def filter_dataset(dataset_name: str) -> Path:
    split, total = resolve_split(dataset_name, DEFAULT_CONFIG)
    iterable = load_dataset(dataset_name, DEFAULT_CONFIG, split=split, streaming=True)
    progress = tqdm(
        total=total,
        unit="rows",
        desc=f"{dataset_name}:{split}",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    output_path = dataset_to_filename(dataset_name)
    match_count = 0
    first_write = True
    pending: Deque[Future[List[MutableMapping[str, object]]]] = deque()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor, output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("[\n")

        def drain_ready(block: bool) -> None:
            nonlocal match_count, first_write
            wait_once = block
            while pending:
                future = pending[0]
                if not wait_once and not future.done():
                    break
                matches = future.result()
                pending.popleft()
                if not matches:
                    wait_once = False
                    continue
                for record in matches:
                    if first_write:
                        first_write = False
                    else:
                        output_file.write(",\n")
                    json.dump(record, output_file, ensure_ascii=False)
                    output_file.flush()
                    match_count += 1
                wait_once = False

        try:
            for batch in chunked(iterable, BATCH_SIZE):
                progress.update(len(batch))
                pending.append(executor.submit(process_batch, batch))
                drain_ready(block=False)
                while len(pending) > PENDING_LIMIT:
                    drain_ready(block=True)
            drain_ready(block=True)
        finally:
            closing = "]\n" if first_write else "\n]\n"
            output_file.write(closing)
            progress.close()

    print(f"{dataset_name}: matched {match_count} rows -> {output_path}")
    return output_path


def main() -> None:
    failures = []
    for dataset_name in DATASETS:
        try:
            filter_dataset(dataset_name)
        except Exception as exc:  # noqa: BLE001
            failures.append((dataset_name, exc))

    if failures:
        for dataset_name, exc in failures:
            print(f"Failed {dataset_name}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
