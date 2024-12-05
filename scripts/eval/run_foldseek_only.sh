############## By length ############### 
# sampdir=/data/lux70/plaid/artifacts/samples/5j007z42/by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir
#     sbatch run_foldseek_only.slrm --samples_dir $subdir --use_designability_filter
#     sbatch run_mmseqs_only.slrm --samples_dir $subdir
#   fi
# done

############## shorter ###############

# sampdir=/data/lux70/plaid/artifacts/samples/ksme77o6/by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir
#     sbatch run_foldseek_only.slrm $subdir
#   fi
# done

############## ProteinGenerator############### 
# sbatch run_foldseek_only.slrm /data/lux70/plaid/baselines/proteingenerator/by_length/ --use_designability_filter
# sbatch run_mmseqs_only.slrm /data/lux70/plaid/baselines/proteingenerator/by_length/


############### Protpardelle ############### 
sampdir=/data/lux70/plaid/baselines/protpardelle/samples_by_length
for subdir in "$sampdir"/*/; do
  if [ -d "$subdir" ]; then
      echo $subdir
      # sbatch run_mmseqs_only.slrm --samples_dir $subdir
      sbatch run_foldseek_only.slrm --samples_dir $subdir --use_designability_filter
  fi
done

# ############### Multiflow ############### 
# sampdir=/data/lux70/plaid/baselines/multiflow/skip8_64per
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir
#     sbatch run_mmseqs_only.slrm --samples_dir $subdir
#     # sbatch run_foldseek_only.slrm --samples_dir $subdir --use_designability_filter
#   fi
# done

############### Natural ############### 
# sampdir=/data/lux70/plaid/artifacts/natural_binned_lengths
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir
#     sbatch run_mmseqs_only.slrm --samples_dir $subdir --use_designability_filter
#     sbatch run_foldseek_only.slrm --samples_dir $subdir --use_designability_filter
#   fi
# done

