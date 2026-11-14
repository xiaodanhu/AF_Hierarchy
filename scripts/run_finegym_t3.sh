#!/bin/bash
# FineGym l_d=3 (element-level detection) training: one system via DeepSpeed
# data parallel. For the full l_d=2 (element-set level) program with the
# two-level video branch see scripts/run_fg2_chain.sh.
# Usage: bash scripts/run_finegym_t3.sh <base|tifad|champ|champ2|champ3|champ3_noas> [gpus=0,1,2] [port=29710]
#   champ   = champion stage 1 (prompt ensemble, action-level attention,
#             apparatus + element-set losses, duration prior)
#   champ2  = champion stage 2: phase-level neighbour attention (text ASA),
#             initialised from the best stage-1 checkpoint, all else frozen
#   champ3  = champion stage 3: VIDEO branch of the action--sub-action
#             attention (libs/modeling/vsa.py), initialised from the best
#             stage-2 checkpoint, all else frozen. Needs the cached CLIP
#             features of the stage-2 model (scripts/cache_fg3_clip_feats.py);
#             scripts/run_fg3_champ3_vsa3gpu.sh runs cache + inert check +
#             init-equality check + training in one go.
set -u
sys=${1:?usage: run_finegym_t3.sh <base|tifad|champ|champ2|champ3> [gpus] [port]}
gpus=${2:-0,1,2}
port=${3:-29710}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs ckpt
cfg=$sys; extra=""
case "$sys" in
  champ2)
    cfg=champ_stage2
    extra="--init_from ckpt/finegym_t3_champ_fg3_champ/vit_best_model --freeze_except text_pathway.phase_attn" ;;
  champ3)
    cfg=champ_stage3_vsa
    CACHE=${CACHE:-/data3/xiaodan8/FineGym/cache_fg3_champ2}
    extra="--init_from ckpt/finegym_t3_champ_stage2_fg3_champ2/vit_best_model --freeze_except video_subtree_attn --feature_cache $CACHE --select_metric video_held_mAP -c 1" ;;
esac
CUDA_VISIBLE_DEVICES=$gpus deepspeed --master_port="$port" train_shard.py \
  --config "configs/finegym_t3_${cfg}.yaml" --output "fg3_${sys}" $extra \
  2>&1 | tee "logs/fg3_${sys}.log"
