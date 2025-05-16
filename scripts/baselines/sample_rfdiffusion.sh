numsamples=64
outdir=/data/lux70/rfdiffusion/skip8_64per

for ((len=64; len<=512; len+=8)); do
    sbatch sample_rfdiffusion.slrm $len $numsamples ${outdir}/length${len}
    sleep 3
done
