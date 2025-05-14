import uuid
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--n_gpus_per_node", type=int, default=1)
parser.add_argument("-n", "--n_nodes", type=int, default=1)
parser.add_argument("-c", "--n_cpus_per_task", type=int, default=12)
parser.add_argument("-f", "--flags", type=str, default="")
args = parser.parse_args()

flags = ""

flags += args.flags

if args.n_nodes > 1:
    multi_node_flags = """
    export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
    export NCCL_SOCKET_IFNAME="^lo,docker,veth"
    """
else:
    multi_node_flags = ""

defaults = f"""#!/usr/bin/env bash
#SBATCH --partition gpu2
#SBATCH --nodes {args.n_nodes} 
#SBATCH --ntasks-per-node {args.n_gpus_per_node} 
#SBATCH --gpus-per-node {args.n_gpus_per_node}
#SBATCH --cpus-per-task {args.n_cpus_per_task}
#SBATCH --time=5-00:00:00
#SBATCH --job-name train 

source ~/.bashrc
micromamba activate plaid 

{multi_node_flags}

export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=TRACE
export PYTHONUNBUFFERED=1
cd /homefs/home/lux70/code/plaid/

nvidia-smi

srun -u --cpu-bind=cores,verbose \
    python /homefs/home/lux70/code/plaid-dev-distributed/train_compositional.py {flags} \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES
"""

hashid = uuid.uuid4().hex[:7]
slrm_spec_fname = f"tmp-{hashid}.slrm"

with open(slrm_spec_fname, "w") as f:
    f.write(defaults)

subprocess.run(["sbatch", slrm_spec_fname])
os.remove(slrm_spec_fname)
