from pathlib import Path
import numpy as np
import argparse

import einops
import safetensors
import pandas as pd
import torch

from cheap.pretrained import CHEAP_pfam_shorten_2_dim_32
from plaid.datasets import NUM_FUNCTION_CLASSES, NUM_ORGANISM_CLASSES
from plaid.utils import get_random_sequence_crop_batch

"""
Configs
"""

parser = argparse.ArgumentParser(
    help="Holdout features from a parquet file"
    "If function_idx or organism_idx is -1, no filtering will be applied."
    "If function_idx or organism_idx is -2, only those that have the null label will be included."
)
parser.add_argument(
    "--parquet_file", type=str, default="/data/lux70/data/pfam/val.parquet"
)
parser.add_argument("--function_idx", type=int, default=-1)
parser.add_argument("--organism_idx", type=int, default=-1)
parser.add_argument("--max_num_batches", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--outdir", type=str, default="/data/lux70/data/pfam/")
args = parser.parse_args()

device = torch.device("cuda")


"""
Set up settings, data, and model
"""
cond_code = f"f{args.function_idx}_o{args.organism_idx}"
outdir = Path(args.outdir) / cond_code

if args.function_idx == -1:
    function_idx = None
elif args.function_idx == -2:
    function_idx = NUM_FUNCTION_CLASSES
else:
    function_idx = args.function_idx

if args.organism_idx == -1:
    organism_idx = None
elif args.organism_idx == -2:
    organism_idx = NUM_ORGANISM_CLASSES
else:
    organism_idx = args.organism_idx


# subset parquet to only those that
df = pd.read_parquet(args.parquet_file)


sequences = df["sequences"].values
cheap_pipeline = CHEAP_pfam_shorten_2_dim_32()
cheap_pipeline.to(device)
all_feats = []


"""
Save the median length so we can use it for sampling later
"""


def round_to_multiple(x, multiple):
    return int(multiple * round(x / multiple))


sequence_lengths = [len(seq) for seq in sequences]
median_sequence_length = round_to_multiple(np.median(sequence_lengths), 4)

with open(outdir / "stats.log", "w") as f:
    f.write(f"median_sequence_length: {median_sequence_length}\n")


"""
Loop through batchs
"""
# get the features in batches
all_features = []

for batch_idx, start_idx in range(0, len(sequences), args.batch_size):
    if batch_idx > args.max_num_batches:
        break

    sequences_batch = sequences[start_idx : start_idx + args.batch_size]
    sequences_batch = get_random_sequence_crop_batch(sequences_batch, args.max_seq_len)
    emb, mask = cheap_pipeline(sequences_batch)
    emb, mask = emb.cpu().numpy(), mask.cpu().numpy()

    masks = einops.repeat(masks, "N L -> N L C", C=32)  # (N, L, 32)
    emb = emb * masks  # (N, L, 32)
    emb = emb.sum(dim=1) / masks.sum(dim=1)  # (N, 32)
    all_features.append(emb)


"""
Save to disk
"""
emb = np.concatenate(all_features, axis=0)

safetensors.torch.save_file({"features": emb}, outdir / "holdout_mean_pool.pt")
