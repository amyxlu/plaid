"""
```
git clone https://github.com/amyxlu/RFDiffusion.git
cd RFDiffusion
uv venv --python 3.8
source .venv/bin/activate
uv pip install -r requirements
```
"""
from tqdm import trange
from pathlib import Path
import time

from chroma import Chroma

chroma = Chroma()

import argparse

parser = argparse.ArgumentParser(description="Sample Chroma")
parser.add_argument(
    "--length", "-l",
    type=int,
    default=512,
    help="Length of the protein to sample",
)
parser.add_argument(
    "--num_samples", "-n",
    type=int,
    default=64,
    help="Number of samples to generate",
)
parser.add_argument(
    "--outdir", "-o",
    type=str,
    default="/data/lux70/plaid/chroma/generated/structures/",
    help="Output directory for generated structures"
) 
args = parser.parse_args()

L = args.length
num_samples = args.num_samples  

# for L in trange(64, 513, 8):
outdir = Path(args.outdir) / f"length{L}/"

if not outdir.exists():
    outdir.mkdir(parents=True)

for i in trange(num_samples):
    start = time.time()

    protein = chroma.sample(chain_lengths=[L])
    protein.to(str(outdir / f"sample{i}.pdb"))

    end = time.time()

    print(f"Length {L} sample {i} took {end - start:.2f} seconds")

    with open(outdir / "../sample_times.csv", "a") as f:
        f.write(f"{L},{i},{end - start:.2f}\n")
