for id in 1b64t79h yfel5fnl v2cer77t 3rs1hxky; do
    for task in "human" "yeast" "ss" "contact"; do
        sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
    done
done


# task="human"
# for id in 2tjrgcde 5z4iaak9 q3m9fhii; do
#     sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
# done


# task="yeast"
# for id in 2tjrgcde 5z4iaak9 q3m9fhii; do
#     sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
# done


# task="contact"
# for id in 3rs1hxky 5z4iaak9 q3m9fhii; do
#     sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
# done


# task="ss"
# for id in 1hr1x9r5 3rs1hxky 5z4iaak9 q3m9fhii; do
#     sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
# done