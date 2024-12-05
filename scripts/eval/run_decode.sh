# # for ((len=176; len<=256; len+=4)); do
# for ((len=32; len<=175; len+=4)); do
#     sbatch run_decode.slrm \
#     ++npz_path=/data/lux70/plaid/artifacts/samples/by_length/$len/latent.npz \
#     ++batch_size=8
# done


python run_decode.py \
    ++npz_path=/data/lux70/plaid/sampling_speed/batchsize1/plaid2b/latent.npz \
    ++batch_size=-1 \
    ++use_compile=False \
    ++num_recycles=1 \
    ++delete_esm_lm=True

python run_decode.py \
    ++npz_path=/data/lux70/plaid/sampling_speed/batchsize1/plaid100m/latent.npz \
    ++batch_size=-1 \
    ++use_compile=False \
    ++num_recycles=1 \
    ++delete_esm_lm=True
