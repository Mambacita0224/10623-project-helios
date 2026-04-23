#!/bin/bash
export WANDB_MODE="offline"
export WANDB_API_KEY=""
export TOKENIZERS_PARALLELISM=true

export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR
#################################################################
## Torch
#################################################################
export TOKENIZERS_PARALLELISM=false
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export TORCH_NCCL_ENABLE_MONITORING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
#################################################################


#################################################################
## NCCL
#################################################################
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=3600000

export NCCL_DEBUG=WARN  # disable the verbose NCCL logs
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0  # was 1
export NCCL_SHM_DISABLE=0  # was 1
export NCCL_P2P_LEVEL=NVL

export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
#################################################################

# #################################################################
# ## DIST
# #################################################################
# MASTER_ADDR=$ARNOLD_WORKER_0_HOST
# ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# export MASTER_PORT=${ports[0]}
# NNODES=$ARNOLD_WORKER_NUM
# NODE_RANK=$ARNOLD_ID
# GPUS_PER_NODE=$ARNOLD_WORKER_GPU
# # GPUS_PER_NODE=1
# # NNODES=1
# # NODE_RANK=0
# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# if [ ! -z $RDZV_BACKEND ]; then
#     DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_id 9863 --rdzv_backend c10d"
#     export NCCL_SHM_DISABLE=1
# fi

# echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"

#################################################################
## ACCELERATE CONFIG
#################################################################
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
export MASTER_PORT=${ports[0]}
NUM_MACHINES=$ARNOLD_WORKER_NUM
MACHINE_RANK=$ARNOLD_ID
NUM_PROCESSES_PER_MACHINE=$ARNOLD_WORKER_GPU

# export CUDA_VISIBLE_DEVICES=0
# NUM_PROCESSES_PER_MACHINE=1
# NUM_MACHINES=1
# MACHINE_RANK=0

ACCELERATE_ARGS="--num_machines $NUM_MACHINES --machine_rank $MACHINE_RANK --num_processes $((NUM_PROCESSES_PER_MACHINE*NUM_MACHINES)) --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"

echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"

# accelerate launch \
#     $ACCELERATE_ARGS \
#     train_helios.py \
#     --config scripts/training/configs/stage_3_post.yaml \
#     2>&1 | tee ./train.log

accelerate launch \
    $ACCELERATE_ARGS \
    --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
    train_helios.py \
    --config scripts/training/configs/stage_3_post.yaml \
    2>&1 | tee ./train.log