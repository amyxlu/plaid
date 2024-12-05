# for ((len=32; len<=175; len+=4)); do
#     sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/artifacts/samples/by_length/$len
# done

# Protpardelle takes chain B instead
# sampdir=/data/lux70/plaid/baselines/protpardelle/samples_by_length

# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir
#     sbatch run_consistency.slrm ++samples_dir=$subdir ++inverse_generate_sequence.designed_chain='B'
#   fi
# done

# sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/baselines/proteingenerator/by_length

# sbatch run_consistency.slrm experiment=multiflow_consistency 

# done


#### Multiflow
# sample_dir=/data/lux70/plaid/baselines/multiflow/skip8_64per/
# for subdir in "$sample_dir"/*/; do
#     if [ -d "$subdir" ]; then
#         echo $subdir
#         sbatch run_consistency.slrm ++samples_dir=$subdir
#     fi
# done


#### Natural
# sample_dir="/data/lux70/plaid/artifacts/natural_binned_lengths/"
# echo $sample_dir
# for subdir in "$sample_dir"/*/; do
#     if [ -d "$subdir" ]; then
#         echo $subdir
#         sbatch run_consistency.slrm ++samples_dir=$subdir
#     fi
# done


# sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/artifacts/natural_binned_lengths/binstart512
# sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/artifacts/natural_binned_lengths/binstart464
# sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/artifacts/natural_binned_lengths/binstart336
# sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/artifacts/natural_binned_lengths/binstart224
# sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/artifacts/natural_binned_lengths/binstart96


# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=128


# maxlen=200
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=64

# maxlen=300
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=64

# maxlen=350
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=32

# maxlen=400
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=32

# maxlen=450
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=16

# maxlen=500
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=8

# maxlen=550
# echo "Running consistency experiments for max_length=$maxlen"


maxlen=610
sbatch run_consistency.slrm \
    ++samples_dir=/data/lux70/plaid/sampling_speed/plaid100m/ \
    ++inverse_generate_sequence.max_length=$maxlen \
    ++phantom_generate_sequence.max_length=$maxlen \
    ++inverse_generate_structure.max_seq_len=$maxlen \
    ++inverse_generate_structure.batch_size=8

maxlen=610
sbatch run_consistency.slrm \
    ++samples_dir=/data/lux70/plaid/sampling_speed/plaid2b/ \
    ++inverse_generate_sequence.max_length=$maxlen \
    ++phantom_generate_sequence.max_length=$maxlen \
    ++inverse_generate_structure.max_seq_len=$maxlen \
    ++inverse_generate_structure.batch_size=8
