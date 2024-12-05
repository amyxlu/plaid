root_dir="/data/lux70/plaid/artifacts/natural_binned_lengths"

for ((len=64; len<=512; len+=8)); do
    sbatch fold_natural.slrm \
        --fasta_file ${root_dir}/binstart${len}/generated/sequences.fasta \
        --outdir ${root_dir}/maxlen${len}/generated/structures \
        --max_seq_len 520 \
        --batch_size 8
done


# python pipeline/run_consistency.py \
#     ++samples_dir=${root_dir}/maxlen${maxlen} \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=$batch_size \
#     ++inverse_generate_structure.max_seq_len=$maxlen
