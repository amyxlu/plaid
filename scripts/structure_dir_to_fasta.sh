sampdir=/data/lux70/plaid/chroma/skip8_64per 
for subdir in "$sampdir"/*/; do
  if [ -d "$subdir" ]; then
    # Check if sequences.fasta already exists in the structures directory
    if [ ! -f "$subdir/generated/sequences.fasta" ]; then
      ls $subdir/generated/structures
      python structure_dir_to_fasta.py -p $subdir/generated/structures
    else
      echo "sequences.fasta already exists in $subdir/generated/structures, skipping..."
    fi
  fi
done