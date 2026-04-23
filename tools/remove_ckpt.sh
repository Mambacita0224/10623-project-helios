DIR="./"
find "$DIR" -type d -name "pytorch_model" -exec rm -rf {} +
find "$DIR" -type d -name "distributed_checkpoint" -exec rm -rf {} +
find "$DIR" -type f -name "random_states_*" -exec rm -f {} +
find "$DIR" -type f -name "scheduler.bin*" -exec rm -f {} +