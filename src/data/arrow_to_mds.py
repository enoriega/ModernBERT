"""
Convert a local, pre-tokenized HuggingFace *arrow* dataset (saved via
`Dataset.save_to_disk` / `DatasetDict.save_to_disk`) into MDS format ready for
the training loop (`NoStreamingDataset` / `StreamingTextDataset` in
`src/text_data.py`).

The input dataset is expected to already contain tokenized `input_ids` plus a
single string id column named either `pmcid` or `pmid` (or one given via
`--id-column`). Output uses the pre-tokenized MDS layout `MDS_COLS_TOKENIZED`
(`input_ids` as ndarray + `id` as str).

We read the saved arrow shards directly with pyarrow instead of
`datasets.load_from_disk`. This (a) sidesteps `dataset_info.json` / embedded
schema metadata that fails to parse when the data was written by a newer
`datasets` version than the one installed (e.g. the `List` feature type), and
(b) lets us convert the shards in parallel: each arrow shard becomes one MDS
partition (written to its own subdirectory), and the per-partition indexes are
merged into a single `index.json` at the end. Iterating a shard never holds more
than one record batch (~1k rows) in memory, so conversion scales to arbitrarily
large datasets.

Example:
    python arrow_to_mds.py -i /path/to/arrow_ds -o /path/to/mds_out
    python arrow_to_mds.py -i ds -o out --compression zstd --num-workers 8
"""

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import tqdm
from streaming import MDSWriter
from streaming.base.util import merge_index

from data_utils import MDS_COLS_TOKENIZED

# Candidate id columns, in priority order, when --id-column is not given.
ID_COLUMN_CANDIDATES = ("pmcid", "pmid")

# We write only input_ids + id. attention_mask is omitted on purpose: the
# training loaders synthesize `np.ones_like(input_ids)` when it is absent, and
# unpadding relies on the true per-sequence lengths.
MDS_COLS_OUT = {k: v for k, v in MDS_COLS_TOKENIZED.items() if k != "attention_mask"}


def resolve_id_column(column_names, explicit):
    """Pick the id column: the explicit one if given, else the first candidate present."""
    if explicit is not None:
        if explicit not in column_names:
            raise ValueError(
                f"--id-column '{explicit}' not found. Available columns: {list(column_names)}"
            )
        return explicit
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in column_names:
            return candidate
    raise ValueError(
        f"Could not find an id column (looked for {ID_COLUMN_CANDIDATES}). "
        f"Available columns: {list(column_names)}. Pass one explicitly with --id-column."
    )


def _iter_record_batches(path: str):
    """Yield the record batches of one arrow file, trying both IPC framings.

    The file stays memory-mapped while its batches are consumed, so only the
    batch currently being processed is realized in Python.
    """
    with pa.memory_map(path, "r") as src:
        try:
            reader = pa.ipc.open_stream(src)
        except pa.lib.ArrowInvalid:
            reader = None
        if reader is not None:
            yield from reader
            return
    with pa.memory_map(path, "r") as src:
        reader = pa.ipc.open_file(src)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def discover_arrow_files(split_dir: str):
    """Return the ordered list of arrow shard files for a saved split directory."""
    state_path = os.path.join(split_dir, "state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            files = [os.path.join(split_dir, e["filename"]) for e in json.load(f)["_data_files"]]
    else:
        files = sorted(glob.glob(os.path.join(split_dir, "*.arrow")))
    if not files:
        raise ValueError(f"No arrow files found in {split_dir}")
    return files


def discover_splits(path: str):
    """Return an ordered dict {split_name_or_None: [arrow_files]} for a saved dataset."""
    dict_manifest = os.path.join(path, "dataset_dict.json")
    if os.path.isfile(dict_manifest):
        with open(dict_manifest) as f:
            split_names = json.load(f)["splits"]
        return {name: discover_arrow_files(os.path.join(path, name)) for name in split_names}
    return {None: discover_arrow_files(path)}


def convert_one_shard(arrow_file: str, out_subdir: str, id_col: str, dtype_str: str, compression, size_limit):
    """Convert one arrow shard into an MDS partition at out_subdir. Returns row count.

    Runs in a worker process, so all arguments are simple picklable values.
    """
    dtype = np.dtype(dtype_str)
    dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
    os.makedirs(out_subdir, exist_ok=True)
    written = 0
    with MDSWriter(out=out_subdir, columns=MDS_COLS_OUT, compression=compression, size_limit=size_limit) as writer:
        for batch in _iter_record_batches(arrow_file):
            columns = batch.to_pydict()
            input_ids_col = columns["input_ids"]
            id_values = columns[id_col]
            for i in range(batch.num_rows):
                # Build at full width first so we can raise a clear error instead of a
                # raw numpy OverflowError when a token id doesn't fit the target dtype.
                wide = np.asarray(input_ids_col[i], dtype=np.int64)
                if wide.size and (int(wide.max()) > dtype_max or int(wide.min()) < dtype_min):
                    raise ValueError(
                        f"Token id {int(wide.max())} does not fit dtype {dtype} "
                        f"(range [{dtype_min}, {dtype_max}]). Use a wider --dtype (int32 is the "
                        f"default; int16 is too small for a ~50k vocab)."
                    )
                writer.write({"input_ids": wide.astype(dtype), "id": str(id_values[i])})
                written += 1
    return written


def convert_split(arrow_files, out_dir: str, id_column, dtype_str: str, compression, size_limit, num_workers: int):
    """Convert one split's arrow shards to MDS at out_dir, in parallel, then merge indexes."""
    # Detect columns / id column once from the first shard's schema (no data read).
    first_batch = next(_iter_record_batches(arrow_files[0]))
    column_names = list(first_batch.schema.names)
    if "input_ids" not in column_names:
        raise ValueError(f"Dataset must contain an 'input_ids' column. Found: {column_names}")
    id_col = resolve_id_column(column_names, id_column)

    n_shards = len(arrow_files)
    if num_workers > n_shards:
        print(
            f"  note: --num-workers={num_workers} but split has only {n_shards} arrow shard(s); "
            f"parallelism is limited to {n_shards}."
        )
    os.makedirs(out_dir, exist_ok=True)

    # One MDS partition per arrow shard, written to its own subdirectory.
    tasks = [
        (f, os.path.join(out_dir, f"part_{i:05d}"), id_col, dtype_str, compression, size_limit)
        for i, f in enumerate(arrow_files)
    ]

    total_rows = 0
    workers = max(1, min(num_workers, n_shards))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(convert_one_shard, *t): t[0] for t in tasks}
        bar = tqdm.tqdm(as_completed(futures), total=n_shards, desc=f"Converting shards -> {out_dir}")
        for fut in bar:
            arrow_file = futures[fut]
            try:
                total_rows += fut.result()
            except Exception as e:
                raise RuntimeError(f"Failed converting shard {arrow_file}: {e}") from e
            bar.set_postfix(rows=total_rows)

    # Merge the per-partition index.json files into a single index.json at out_dir.
    merge_index(out_dir)
    return total_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the saved arrow dataset")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for the MDS dataset")
    parser.add_argument(
        "--id-column", type=str, default=None,
        help=f"Name of the string id column. If omitted, auto-detects one of {ID_COLUMN_CANDIDATES}.",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="If the input is a DatasetDict, only convert this split. Otherwise all splits are converted.",
    )
    parser.add_argument(
        "--dtype", type=str, default="int32",
        help="numpy dtype for input_ids (default int32; int16 is too small for a ~50k vocab, uint16 works).",
    )
    parser.add_argument(
        "--compression", type=str, default=None,
        help="MDS compression (e.g. 'zstd'). Default: none (raw MDS, ready for NoStreamingDataset).",
    )
    parser.add_argument(
        "--size-limit", type=str, default="64mb",
        help="Approximate shard size limit (default 64mb).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Parallel worker processes; arrow shards are converted concurrently (default 1).",
    )
    args = parser.parse_args()

    # Validate the dtype early (also errors clearly on a bad name).
    np.dtype(args.dtype)

    splits = discover_splits(args.input)

    if args.split:
        if list(splits) == [None]:
            raise ValueError("--split given but the input is a single Dataset, not a DatasetDict.")
        if args.split not in splits:
            raise ValueError(f"Split '{args.split}' not in dataset. Available: {list(splits)}")
        splits = {args.split: splits[args.split]}

    counts = {}
    for split_name, arrow_files in splits.items():
        out_dir = os.path.join(args.output, split_name) if split_name else args.output
        counts[split_name] = convert_split(
            arrow_files, out_dir, args.id_column, args.dtype, args.compression, args.size_limit, args.num_workers
        )

    print("\nDone. Wrote:")
    for split_name, n in counts.items():
        out_dir = os.path.join(args.output, split_name) if split_name else args.output
        print(f"  {out_dir}: {n} samples")

    # Suggest a ready-to-paste training config snippet.
    has_named_splits = any(s is not None for s in splits)
    example_split = next((s for s in splits if s is not None), None)
    print("\nExample train-config `dataset` section:")
    print("  train_loader:")
    print("    name: text")
    print("    dataset:")
    print("      streaming: false")
    print(f"      local: {args.output}")
    if has_named_splits:
        print(f"      split: {example_split}")
    print("      max_seq_len: ${max_seq_len}")


if __name__ == "__main__":
    main()
