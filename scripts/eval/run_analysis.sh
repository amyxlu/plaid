#! /bin/bash

############### Scaling ############### 
sampdir=/data/lux70/plaid/artifacts/samples/scaling
for model_id in 'btmop0c8' 'reklt5kg' '5j007z42' 'ksme77o6' 'lqp25b7g' '6ryvfi2v'; do
  for subdir in "$sampdir"/${model_id}/*/; do
    echo $sampdir/$model_id/$subdir
    sbatch run_analysis.slrm $sampdir/$model_id/$subdir
  done
done


############### By length ############### 
# sampdir=/data/lux70/plaid/artifacts/samples/5j007z42/scratch_by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_analysis.slrm $subdir$len
#   fi
# done

############### ProteinGenerator############### 
# sbatch run_analysis.slrm /data/lux70/plaid/baselines/proteingenerator/by_length/

############### ProtPardelle ############### 
# sampdir=/data/lux70/plaid/baselines/protpardelle/samples_by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_analysis.slrm $subdir$len
#   fi
# done

# ############## Multiflow ############### 
# sampdir=/data/lux70/plaid/baselines/multiflow/skip8_64per
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_analysis.slrm $subdir$len
#   fi
# done

# ############## Natural############### 
# sampdir=/data/lux70/plaid/artifacts/natural_binned_lengths
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_analysis.slrm $subdir$len
#   fi
# done
