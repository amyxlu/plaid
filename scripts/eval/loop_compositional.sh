
# 758 ['BOVIN']
# 1411 ['CHICK']
# 2436 ['ECOLI']
# 158 ['HORSE']
# 1326 ['HUMAN']
# 1294 ['MAIZE']
# 300 ['MOUSE']
# 799 []
# 1265 ['RABIT']
# 716 []
# 333 ['SHEEP']
# 1357 ['SOYBN']
# 1388 ['TOBAC']
# 1452 ['WHEAT']
# 818 ['YEAST']

# function_idx=841
# length=200
# organism_idx=2436
# subdir=""

# sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir

# organism_idx=2475
# function_idx=1197
# sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir

organism_idx=2436
for function_idx in 772 782 841 852 959 1059 1142 1428 1981; do
    echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir
    sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
done

### prokaryotic only functions:
organism_idx=2436
function_idx=1982
sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir

### plant functions
organism_idx=1398
for function_idx in 1199 1980; do
    sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
done




# length=212
# function_idx=323
# for organism_idx in 1326 300; do
#     sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
# done

###################
# loop organism 
###################

function_idx=54
cond_scale=3
length="None"
subdir=""
sampling_timesteps=500

for organism_idx in 818 2436 1326 3702 300 1357 758; do
    echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir 
    sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir 
done

###################
# cond sclaing ablation 
###################

# organism_idx=1326
# length=256
# subdir="cond_scale"
# function_idx=5

# for cond_scale in 0 1 2 3 4 5 6 7 8 9 10; do
#     echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir
#     sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
# done


###################
# Unconditional sampling
###################

# function_idx=2219
# organism_idx=3617
# cond_scale=3
# length=100
# subdir="timesteps"
subdir="thermophiles"

# for sampling_timesteps in 25 50 100 200 400 800 1000; do
#     echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir SamplingTimesteps $sampling_timesteps
#     sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir $sampling_timesteps
# done

for organism_idx in 698 2165 2234; do
    for function_idx in 2219; do
        for length in 16 32 64; do
            for cond_scale in 0 3; do
                sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
            done
        done
    done
done


for organism_idx in 698 2165 2234; do
    for function_idx in 125 213 852; do
        for length in "None"; do
            for cond_scale in 3 7; do
                sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
            done
        done
    done
done