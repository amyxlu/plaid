sbatch sample_by_median_length.slrm \
    --loop_over='organism' \
    --start_idx=1 \
    --n_samples=32 \
    --end_idx=100

sbatch sample_by_median_length.slrm \
    --loop_over='organism' \
    --start_idx=101 \
    --n_samples=32 \
    --end_idx=200

# sbatch sample_by_median_length.slrm \
#     --loop_over='organism' \
#     --start_idx=201 \
#     --n_samples=32 \
#     --end_idx=300

# sbatch sample_by_median_length.slrm \
#     --loop_over='organism' \
#     --start_idx=301 \
#     --n_samples=32 \
#     --end_idx=400

# sbatch sample_by_median_length.slrm \
#     --loop_over='organism' \
#     --start_idx=401 \
#     --n_samples=32 \
#     --end_idx=500

# sbatch sample_by_median_length.slrm \
#     --loop_over='organism' \
#     --start_idx=501 \
#     --n_samples=32 \
#     --end_idx=600
