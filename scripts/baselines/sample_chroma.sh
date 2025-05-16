numsamples=64
# for ((len=64; len<=512; len+=8)); do
# for ((len=328; len<=512; len+=8)); do
#     sbatch sample_chroma.slrm -l $len -n $numsamples
# done

for len in 328 392 400 448 472 488 504; do
    sbatch sample_chroma.slrm -l $len -n $numsamples
    sleep 3
done
