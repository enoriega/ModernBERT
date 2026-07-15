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
import glob
import json
import os

import numpy as np
import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
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


def _load_split_from_arrow_files(split_dir: str) -> Dataset:
    """Load a saved-to-disk split by reading its arrow files directly.

    Bypasses `dataset_info.json`, whose feature schema can fail to parse when the
    dataset was written by a newer `datasets` version than the one installed
    (e.g. the `List` feature type). Column types are recovered from the arrow
    schema instead, which is all this converter needs.
    """
    state_path = os.path.join(split_dir, "state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            files = [os.path.join(split_dir, e["filename"]) for e in json.load(f)["_data_files"]]
    else:
        files = sorted(glob.glob(os.path.join(split_dir, "*.arrow")))
    if not files:
        raise ValueError(f"No arrow files found in {split_dir}")
    return concatenate_datasets([Dataset.from_file(f) for f in files])


def load_splits(path: str):
    """Return an ordered dict {split_name_or_None: Dataset} for a saved arrow dataset.

    Tries `load_from_disk` first; on any failure (typically a datasets-version
    metadata mismatch) falls back to reading the arrow files directly.
    """
    try:
        ds = load_from_disk(path)
        return dict(ds) if isinstance(ds, DatasetDict) else {None: ds}
    except Exception as e:
        print(f"load_from_disk failed ({type(e).__name__}: {e}); falling back to raw arrow-file loading.")

    dict_manifest = os.path.join(path, "dataset_dict.json")
    if os.path.isfile(dict_manifest):
        with open(dict_manifest) as f:
            split_names = json.load(f)["splits"]
        return {name: _load_split_from_arrow_files(os.path.join(path, name)) for name in split_names}
    return {None: _load_split_from_arrow_files(path)}


def convert_split(dataset: Dataset, out_dir: str, id_column: str, dtype: np.dtype, compression, size_limit):
    """Write a single split's rows to an MDS folder at out_dir."""
    if "input_ids" not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain an 'input_ids' column. Found: {dataset.column_names}"
        )
    id_col = resolve_id_column(dataset.column_names, id_column)
    dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max

    os.makedirs(out_dir, exist_ok=True)
    with MDSWriter(
        out=out_dir,
        columns=MDS_COLS_OUT,
        compression=compression,
        size_limit=size_limit,
    ) as writer:
        for row in tqdm.tqdm(dataset, desc=f"Writing {out_dir}", total=len(dataset)):
            # Build at full width first so we can give a clear error instead of a
            # raw numpy OverflowError when a token id doesn't fit the target dtype.
            wide = np.asarray(row["input_ids"], dtype=np.int64)
            if wide.size and (int(wide.max()) > dtype_max or int(wide.min()) < dtype_min):
                raise ValueError(
                    f"Token id {int(wide.max())} does not fit dtype {dtype} "
                    f"(range [{dtype_min}, {dtype_max}]). Use a wider --dtype (int32 is the default; "
                    f"int16 is too small for a ~50k vocab)."
                )
            writer.write({"input_ids": wide.astype(dtype), "id": str(row[id_col])})
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

    splits = load_splits(args.input)

    if args.split:
        if list(splits) == [None]:
            raise ValueError("--split given but the input is a single Dataset, not a DatasetDict.")
        if args.split not in splits:
            raise ValueError(f"Split '{args.split}' not in dataset. Available: {list(splits)}")
        splits = {args.split: splits[args.split]}

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
