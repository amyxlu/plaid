for model_id in "8ebs7j9h" "mm9fe6x9" "kyytc8i9" "fbbrfqzk" "13lltqha" "uhg29zk4" "ich20c3q" "7str7fhl" "g8e83omk"; do
    python run_diffusion_slrm.py --n_gpus 1 --flags "experiment=hdif/cath_dit ++compression_model_id=${model_id}"
done

# for model_id in q3m9fhii 5z4iaak9 3rs1hxky 2tjrgcde v2cer77t yfel5fnl 1hr1x9r5 1b64t79h; do
#     python run_diffusion_slrm.py --n_gpus 1 --flags "experiment=hdif/cath_dit ++compression_model_id=${model_id}"
# done