task=beta

for id in identity g8e83omk 7str7fhl ich20c3q uhg29zk4 13lltqha fbbrfqzk kyytc8i9 mm9fe6x9 8ebs7j9h; do
    sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
done

