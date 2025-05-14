'''

```
git clone https://github.com/evolutionaryscale/esm.git
cd esm

uv venv --python 3.10
source .venv/bin/activate
pip install -e .
```

Need to add token to HF_TOKEN as an env. variable. 
'''
from pathlib import Path
import time

from tqdm import trange

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig


# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
login()

# This will download the model weights and instantiate the model on your machine.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda") # or "cpu"


for L in trange(64, 513, 8):
    outdir = Path(f"/data/lux70/plaid/esm3/generated/structures/length{L}/")

    if not outdir.exists():
        outdir.mkdir(parents=True)

    for i in trange(64):
        start = time.time()

        prompt = "_" * L
        protein = ESMProtein(sequence=prompt)

        # Generate the sequence, then the structure. This will iteratively unmask the sequence track.
        protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))

        # We can show the predicted structure for the generated sequence.
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))

        # save protein PDB
        protein.to_pdb(outdir / f"sample{i}.pdb")

        end = time.time()

        print(f"Length {L} sample {i} took {end - start:.2f} seconds")

        with open(outdir / "../sequences.fasta", "a") as f:
            f.write(f">sample{i}\n{protein.sequence}\n")

        with open(outdir / "../sample_times.csv", "a") as f:
            f.write(f"{i},{end - start:.2f}\n")


