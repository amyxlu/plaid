############
# tanh
############

# sbatch train_hourglass.slrm resume_from_model_id=jzlv54wl
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_8" ++hourglass.lr=1e-4
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=1 ++logger.name="pfam_tanh_512_4" ++hourglass.lr=1e-4

# sbatch train_hourglass.slrm experiment=hvq/bounded/rocklin.yaml
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_32"
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_64"

# sbatch train_hourglass.slrm experiment=hvq/fsq/pfam_full ++hourglass.fsq_levels=[16,16,16,16] ++logger.name="pfam_fsq_16_16_16_16" 
# sbatch train_hourglass.slrm experiment=hvq/fsq/pfam_full ++hourglass.fsq_levels=[8,8,8,6,5] ++logger.name="pfam_fsq_86"  

# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=2 ++logger.name="cath_tanh_256_4"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2 ++logger.name="cath_tanh_256_8"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=64 ++hourglass.shorten_factor=2 ++logger.name="cath_tanh_256_16"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2 ++logger.name="cath_tanh_256_32"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2 ++logger.name="cath_tanh_256_64"

# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++logger.name="pfam_tanh_512_8" ++hourglass.lr=6e-5
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_8" ++hourglass.lr=6e-5
# sbatch train_hourglass.slrm experiment=hvq/bounded/uniref_full.yaml ++hourglass.downproj_factor=128 ++logger.name="uniref_tanh_512_8" ++hourglass.lr=6e-5

# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_4"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_8"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=64 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_16"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_32"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_64"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=8 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_128"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_256"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=2 ++hourglass.shorten_factor=1 ++logger.name="cath_tanh_512_512"

# sbatch train_hourglass.slrm --resume_from_model_id mm9fe6x9	


############
# vqvae 
############
# for codebook_size in 16 64 256 1024 4096 16384; do
#     sbatch train_hourglass.slrm experiment=hvq/vq/cath ++hourglass.n_e=${codebook_size} ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=1
# done

# sbatch train_hourglass.slrm experiment=hvq/vq/cath ++hourglass.n_e=65536 ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=1


# ############
# # FSQ 
# ############

# for fsq_levels in "[4,4]" "[4,4,4]" "[8,6,5]" "[8,5,5,5]" "[7,5,5,5,5]" "[8,8,8,6,5]" "[8,8,8,5,5,5]"; do
#     sbatch train_hourglass.slrm experiment=hvq/fsq/cath_small ++hourglass.fsq_levels=${fsq_levels} ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=1
# done


############
# Resume 
############
# for id in 5lha1r65 g69njjq4 pjr3qhkp 4p7efzza a1z93f0w qmbg5t8m sbrib6ob yvberwqn 4e0ct6zj 7dga88v8 mu72a8vd onvtfdg2 w8qlmbfk 5v3np72i; do
#     sbatch train_hourglass.slrm resume_from_model_id=$id
# done

# for shorten_factor in 1 2 4 8; do
#     python run_hourglass_slrm.py --n_gpus 1 --flags "experiment=/hvq/bounded/cath_small ++use_compression_callback=False ++hourglass.shorten_factor=${shorten_factor} ++hourglass.downproj_factor=1"
# done

for id in 3j8kj6qy emhebt4i x0a30hwj p6etb6im i34xqstz npqabyji 3n9mrtx6 8ohx6d6b; do
    python run_hourglass_slrm.py --n_gpus 1 --flags "resume_from_model_id=$id"
done