"""
```
git clone https://github.com/generatebio/chroma.git
pip install -e chroma # use `-e` for it to be editable locally. 
```
"""
from tqdm import trange
from pathlib import Path
import time

from chroma import Chroma

chroma = Chroma()

for L in trange(64, 513, 8):
    outdir = Path(f"/data/lux70/plaid/chroma/generated/structures/length{L}/")

    if not outdir.exists():
        outdir.mkdir(parents=True)

    for i in trange(64):
        start = time.time()

        protein = chroma.sample(chain_lengths=[L])
        protein.to(str(outdir / f"sample{i}.pdb"))

        end = time.time()

        print(f"Length {L} sample {i} took {end - start:.2f} seconds")

        with open(outdir / "../sample_times.csv", "a") as f:
            f.write(f"{L},{i},{end - start:.2f}\n")
