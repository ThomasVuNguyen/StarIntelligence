#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, MutableMapping, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("pyarrow is required. Install it with `pip install pyarrow`.") from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("tqdm is required. Install it with `pip install tqdm`.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS: Sequence[Path] = (
    SCRIPT_DIR / "arc-stack-c_matches.json",
    SCRIPT_DIR / "arc-stack-cpp_matches.json",
)


def iter_json_array(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> Iterator[MutableMapping[str, object]]:
    """Incrementally parse a JSON array file and yield each object."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as source:
        buffer = ""
        eof = False
        while not eof:
            chunk = source.read(chunk_bytes)
            if not chunk:
                eof = True
            buffer += chunk
            start = 0
            while True:
                while start < len(buffer) and buffer[start] in " \r\n\t[,":
                    start += 1
                if start >= len(buffer):
                    break
                if buffer[start] == "]":
                    return
                try:
                    record, next_pos = decoder.raw_decode(buffer, start)
                except json.JSONDecodeError:
                    break
                if not isinstance(record, MutableMapping):
                    raise ValueError(f"Expected dict entries in {path}, found {type(record)!r}")
                yield record
                start = next_pos
            buffer = buffer[start:]
        start = 0
        while start < len(buffer) and buffer[start] in " \r\n\t,":
            start += 1
        if start < len(buffer) and buffer[start] != "]":
            raise ValueError(f"Unexpected trailing content in {path}")


class ParquetSink:
    """Append-only Parquet writer that can roll over into multiple files."""

    def __init__(
        self,
        destination_dir: Path,
        base_name: str,
        max_rows_per_file: int | None,
        compression: str = "zstd",
    ) -> None:
        self.destination_dir = destination_dir
        self.base_name = base_name
        self.max_rows_per_file = max_rows_per_file
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._rows_in_current_file = 0
        self._file_index = 0

    def _next_path(self) -> Path:
        if self.max_rows_per_file:
            suffix = f"-{self._file_index:05d}"
            self._file_index += 1
        else:
            suffix = ""
        return self.destination_dir / f"{self.base_name}{suffix}.parquet"

    def _ensure_writer(self, schema: pa.Schema) -> None:
        if self._writer is None:
            output_path = self._next_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(output_path, schema, compression=self.compression)
            self._rows_in_current_file = 0

    def write_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self._writer is None:
            self._ensure_writer(table.schema)
        elif self.max_rows_per_file and self._rows_in_current_file + table.num_rows > self.max_rows_per_file:
            self.close()
            self._ensure_writer(table.schema)

        if self._writer is None:  # pragma: no cover - defensive
            raise RuntimeError("Writer not initialized")
        self._writer.write_table(table)
        self._rows_in_current_file += table.num_rows
        if self.max_rows_per_file and self._rows_in_current_file >= self.max_rows_per_file:
            self.close()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._rows_in_current_file = 0


def convert_file(
    input_path: Path,
    output_dir: Path,
    rows_per_file: int | None,
    rows_per_batch: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    stem = input_path.stem
    sink = ParquetSink(output_dir, stem, rows_per_file)
    buffer: List[MutableMapping[str, object]] = []
    iterator = iter_json_array(input_path)

    progress = tqdm(desc=f"{input_path.name} -> parquet", unit="rows", dynamic_ncols=True)
    try:
        for record in iterator:
            buffer.append(record)
            if len(buffer) >= rows_per_batch:
                table = pa.Table.from_pylist(buffer)
                sink.write_table(table)
                progress.update(table.num_rows)
                buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer)
            sink.write_table(table)
            progress.update(table.num_rows)
    finally:
        sink.close()
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON array dumps into Parquet shards.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="JSON files to convert (defaults to the arc-stack outputs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/parquet"),
        help="Directory to place parquet files (default: output/parquet).",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=1_000_000,
        help="Maximum number of rows per parquet file (0 disables sharding).",
    )
    parser.add_argument(
        "--rows-per-batch",
        type=int,
        default=50_000,
        help="Number of records to accumulate in memory before writing a parquet batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs: Iterable[Path] = args.inputs or DEFAULT_INPUTS
    rows_per_file = args.rows_per_file or None
    for input_path in inputs:
        convert_file(input_path, args.output_dir, rows_per_file, args.rows_per_batch)


if __name__ == "__main__":
    main()
