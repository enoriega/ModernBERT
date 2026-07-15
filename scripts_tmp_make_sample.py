"""Copy the first N samples of an MDS split into a new dir for smoke testing."""
import argparse
from streaming import StreamingDataset, MDSWriter
import numpy as np
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="MDS dir containing the split subfolder")
p.add_argument("--split", default="train")
p.add_argument("--out", default="data_sample")
p.add_argument("--out_split", default="train")
p.add_argument("--n", type=int, default=2000)
args = p.parse_args()

ds = StreamingDataset(local=args.src, split=args.split, shuffle=False)
sample0 = ds[0]
# Infer columns from the first sample (pretokenized input_ids expected).
cols = {}
for k, v in sample0.items():
    if isinstance(v, np.ndarray):
        cols[k] = f"ndarray:{v.dtype}"
    elif isinstance(v, int):
        cols[k] = "int"
    elif isinstance(v, str):
        cols[k] = "str"
print("Columns:", cols)

with MDSWriter(out=f"{args.out}/{args.out_split}", columns=cols, compression=None) as w:
    for i in tqdm(range(min(args.n, len(ds)))):
        w.write(dict(ds[i]))
print(f"Wrote {min(args.n, len(ds))} samples to {args.out}/{args.out_split}")
