# arg_list=(32 40 48 56 64)
# sbatch run_pipeline.slrm "${arg_list[@]}"

# cd /homefs/home/lux70/code/plaid/scripts
# for len in 100 200 300 350 400 450 500 550; do
#     python structure_dir_to_fasta.py -p /data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${len}/generated/structures
# done

# model_id="ksme77o6"

# for ((len=32; len<=256; len+=4)); do
#     sbatch run_pipeline.slrm \
#     ++sample.length=$len \
#     ++sample.output_root_dir="/data/lux70/plaid/artifacts/samples/${model_id}/by_length/$len" \
#     ++sample.model_id=${model_id} \
#     ++log_to_wandb=False
# done

##############################
# Benchmark 
##############################


# sbatch run_pipeline.slrm \
#     ++sample.num_samples=100 \
#     ++sample.length=600 \
#     ++sample.batch_size=-1 \
#     ++sample.model_id="ksme77o6" \
#     ++sample.output_root_dir=/data/lux70/plaid/artifacts/samples/benchmark_sampling_speed/${model_id}/samples${batch_size} \
#     ++sample.use_compile=True \
#     ++decode.use_compile=True \
#     ++decode.batch_size=16 \
#     ++inverse_generate_sequence.max_length=650 \
#     ++inverse_generate_structure.max_seq_len=650 \
#     ++phantom_generate_sequence.max_length=650 

sbatch run_pipeline.slrm \
    ++sample.num_samples=1 \
    ++sample.length=300 \
    ++sample.batch_size=1 \
    ++sample.model_id="ksme77o6" \
    ++sample.output_root_dir=/data/lux70/plaid/sampling_speed/batchsize1/plaid100m \
    ++sample.use_compile=False \
    ++decode.use_compile=False \
    ++decode.batch_size=1 \
    ++inverse_generate_sequence.max_length=650 \
    ++inverse_generate_structure.max_seq_len=650 \
    ++phantom_generate_sequence.max_length=650 

sbatch run_pipeline.slrm \
    ++sample.num_samples=1 \
    ++sample.length=300 \
    ++sample.batch_size=1 \
    ++sample.model_id="5j007z42" \
    ++sample.output_root_dir=/data/lux70/plaid/sampling_speed/batchsize1/plaid100m \
    ++sample.use_compile=False \
    ++decode.use_compile=False \
    ++decode.num_recycles=1 \
    ++decode.batch_size=1 \
    ++inverse_generate_sequence.max_length=650 \
    ++inverse_generate_structure.max_seq_len=650 \
    ++phantom_generate_sequence.max_length=650 

# sbatch run_pipeline.slrm \
#     ++sample.num_samples=100 \
#     ++sample.length=300 \
#     ++sample.batch_size=-1 \
#     ++sample.model_id="5j007z42" \
#     ++sample.output_root_dir=/data/lux70/plaid/sampling_speed/plaid2b \
#     ++sample.use_compile=False \
#     ++decode.use_compile=False \
#     ++decode.batch_size=8 \
#     ++inverse_generate_sequence.max_length=650 \
#     ++inverse_generate_structure.max_seq_len=650 \
#     ++phantom_generate_sequence.max_length=650 

# sbatch run_pipeline.slrm \
#     ++sample.num_samples=100 \
#     ++sample.length=300 \
#     ++sample.batch_size=-1 \
#     ++sample.model_id="ksme77o6" \
#     ++sample.output_root_dir=/data/lux70/plaid/sampling_speed/plaid100m \
#     ++sample.use_compile=False \
#     ++decode.use_compile=False \
#     ++decode.batch_size=8 \
#     ++inverse_generate_sequence.max_length=650 \
#     ++inverse_generate_structure.max_seq_len=650 \
#     ++phantom_generate_sequence.max_length=650 

##############################
# Timesteps
##############################

# len=148
# function_idx=2219
# organism_idx=3617

# model_id="5j007z42"

# for ((t=110; t<=1100; t+=100)); do
#     sbatch run_pipeline.slrm \
#     ++sample.length=$len \
#     ++sample.output_root_dir="/data/lux70/plaid/artifacts/samples/${model_id}/timestep/len${len}_timestep${t}_f${function_idx}_o${organism_idx}" \
#     ++sample.sampling_timesteps=$t \
#     ++log_to_wandb=False
# done