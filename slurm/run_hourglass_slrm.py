import uuid
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--flags", type=str, default="")
parser.add_argument("--mem", default="200G")
args = parser.parse_args()

flags = ""
flags += args.flags

defaults = f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH -c 64
#SBATCH --gpus {args.n_gpus}
#SBATCH --mem {args.mem}
#SBATCH --time=10-00:00:00
#SBATCH --job-name train 

eval "$(conda shell.bash hook)"
conda activate plaid 

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
cd /homefs/home/lux70/code/plaid/

nvidia-smi
srun python train_hourglass_vq.py {flags}
"""

hashid = uuid.uuid4().hex[:7]
slrm_spec_fname = f"tmp-{hashid}.slrm"

with open(slrm_spec_fname, "w") as f:
    f.write(defaults)

subprocess.run(["sbatch", slrm_spec_fname])
os.remove(slrm_spec_fname)
