#!/bin/bash
# FineGym Table-3 training: one system on 3 GPUs via DeepSpeed data parallel.
# Usage: bash scripts/run_finegym_t3.sh <base|tifad|champ|champ2> [gpus=0,1,2] [port=29710]
#   champ  = champion stage 1 (prompt ensemble, action-level attention,
#            apparatus + element-set losses, duration prior)
#   champ2 = champion stage 2: APA phase-level attention, initialised from the
#            best stage-1 checkpoint with all other parameters frozen
set -u
sys=${1:?usage: run_finegym_t3.sh <base|tifad|champ|champ2> [gpus] [port]}
gpus=${2:-0,1,2}
port=${3:-29710}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs ckpt
cfg=$sys; extra=""
if [ "$sys" = champ2 ]; then
  cfg=champ_stage2
  extra="--init_from ckpt/finegym_t3_champ_fg3_champ/vit_best_model --freeze_except text_pathway.phase_attn"
fi
CUDA_VISIBLE_DEVICES=$gpus deepspeed --master_port="$port" train_shard.py \
  --config "configs/finegym_t3_${cfg}.yaml" --output "fg3_${sys}" $extra \
  2>&1 | tee "logs/fg3_${sys}.log"
