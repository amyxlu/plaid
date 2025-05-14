# for wandb sweep: sweep.slrm specifies compute requirements for wandb agent project/agentid
njobs=12
wandb_agent_id="7j1b2heo"
for i in $(seq 1 $njobs); do
    sbatch /homefs/home/lux70/code/plaid/slurm/agent_sweep_diffusion.slrm $wandb_agent_id
done
