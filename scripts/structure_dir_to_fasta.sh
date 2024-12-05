sampdir=/data/lux70/plaid/baselines/multiflow/skip8_64per

for subdir in "$sampdir"/*/; do
  if [ -d "$subdir" ]; then
    echo $subdir/generated
    python structure_dir_to_fasta.py -p $subdir/generated
  fi
done
