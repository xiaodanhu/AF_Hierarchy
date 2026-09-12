#!/bin/bash
# FineGym Table-3 training: one system on 3 GPUs via DeepSpeed data parallel.
# Usage: bash scripts/run_finegym_t3.sh <base|tifad|champ> [gpus=0,1,2] [port=29710]
set -u
sys=${1:?usage: run_finegym_t3.sh <base|tifad|champ> [gpus] [port]}
gpus=${2:-0,1,2}
port=${3:-29710}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs ckpt
CUDA_VISIBLE_DEVICES=$gpus deepspeed --master_port="$port" train_shard.py \
  --config "configs/finegym_t3_${sys}.yaml" --output "fg3_${sys}" \
  2>&1 | tee "logs/fg3_${sys}.log"
