# for organism in 758 1411 2436 158 1326 1294 300 799 1265 716 333 1357 1388 1452 818; do
# for organism in 758 818 2436 1326; do
#     sbatch loop_organism.slrm $organism
# done

# for ((org_idx=0; org_idx<=100; len+=2)); do
#     sbatch loop_latent_only.slrm $org_idx 
# done

# Gammaproteobacteria, bacteria
# Pezizomycotina, fungi
# 1326, human
# 1357, soybean
# 818, yeast
# 2436, ecoli
# 92 9FUNG
# 66 archaea
# 47 9chlo
# 2475 virus Caudoviricetes
# 822 cannabis
# 313 insect
# 1931

function_idx=2219
# organism_idx=3617
length=60

for organism_idx in 1326 1357 818 2436 4 49 92 66 47 2475 822 313 1931; do
    sbatch run_pipeline.slrm \
        sample=sample_conditional \
        ++sample.function_idx=$function_idx \
        ++sample.organism_idx=$organism_idx \
        ++sample.cond_scale=3 \
        ++sample.length=$length \
        ++sample.num_samples=100 \
        ++sample.output_root_dir=/data/lux70/plaid/artifacts/samples/5j007z42/conditional/organism \
        ++run_cross_consistency=False \
        ++run_self_consistency=False \
        ++run_analysis=False
done