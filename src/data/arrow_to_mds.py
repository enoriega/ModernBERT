"""
Convert a local, pre-tokenized HuggingFace *arrow* dataset (saved via
`Dataset.save_to_disk` / `DatasetDict.save_to_disk`) into MDS format ready for
the training loop (`NoStreamingDataset` / `StreamingTextDataset` in
`src/text_data.py`).

The input dataset is expected to already contain tokenized `input_ids` plus a
single string id column named either `pmcid` or `pmid` (or one given via
`--id-column`). Output uses the pre-tokenized MDS layout `MDS_COLS_TOKENIZED`
(`input_ids` as ndarray + `id` as str).

Example:
    python arrow_to_mds.py --input /path/to/arrow_ds --output /path/to/mds_out
    python arrow_to_mds.py -i ds -o out --compression zstd --dtype int32
"""

import argparse
import os

import numpy as np
import tqdm
from datasets import Dataset, DatasetDict, load_from_disk
from streaming import MDSWriter

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


def convert_split(dataset: Dataset, out_dir: str, id_column: str, dtype: np.dtype, compression, size_limit):
    """Write a single split's rows to an MDS folder at out_dir."""
    if "input_ids" not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain an 'input_ids' column. Found: {dataset.column_names}"
        )
    id_col = resolve_id_column(dataset.column_names, id_column)
    dtype_max = np.iinfo(dtype).max

    os.makedirs(out_dir, exist_ok=True)
    with MDSWriter(
        out=out_dir,
        columns=MDS_COLS_OUT,
        compression=compression,
        size_limit=size_limit,
    ) as writer:
        for row in tqdm.tqdm(dataset, desc=f"Writing {out_dir}", total=len(dataset)):
            input_ids = np.asarray(row["input_ids"], dtype=dtype)
            # Guard against silent overflow when down-casting token ids.
            if input_ids.size and int(input_ids.max()) > dtype_max:
                raise ValueError(
                    f"A token id exceeds the max representable by {dtype} ({dtype_max}). "
                    f"Use a wider --dtype."
                )
            writer.write({"input_ids": input_ids, "id": str(row[id_col])})
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the arrow dataset (load_from_disk)")
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
        help="numpy dtype for input_ids (default int32; int16 is too small for a ~50k vocab).",
    )
    parser.add_argument(
        "--compression", type=str, default=None,
        help="MDS compression (e.g. 'zstd'). Default: none (raw MDS, ready for NoStreamingDataset).",
    )
    parser.add_argument(
        "--size-limit", type=str, default="64mb",
        help="Approximate shard size limit (default 64mb).",
    )
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)

    ds = load_from_disk(args.input)

    if isinstance(ds, DatasetDict):
        splits = {args.split: ds[args.split]} if args.split else dict(ds)
        if args.split and args.split not in ds:
            raise ValueError(f"Split '{args.split}' not in dataset. Available: {list(ds.keys())}")
    else:
        if args.split:
            raise ValueError("--split given but the input is a single Dataset, not a DatasetDict.")
        splits = {None: ds}

    counts = {}
    for split_name, split_ds in splits.items():
        out_dir = os.path.join(args.output, split_name) if split_name else args.output
        counts[split_name] = convert_split(
            split_ds, out_dir, args.id_column, dtype, args.compression, args.size_limit
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
