#!/bin/bash
# Eval-only of the FineDiving and FineGym champions with cached CLIP features,
# dumping the video-level predictions (A6_EVAL_SAVE_PRED=1 -> actionformer6/output/<dataset>/
# pred_<ts>_video_level.json) for the verification kits (ground truth vs prediction).
# One dispatcher job, one card, FineDiving then FineGym (~10-15 min each once the memory is free).
# usage (dispatcher): a6pred<TAB>bash scripts/a6_eval_pred_job.sh
source /data3/xiaodan8/tifad/scripts/gpu_hold.sh
cd /data3/xiaodan8/actionformer6 || exit 1
tag=${2:-a6pred}; only=${1:-both}   # usage: a6_eval_pred_job.sh [fd|fg|both] [tag]
export FINEGYM_DATA_ROOT=/data3/xiaodan8/FineGym FINEGYM_RGB_ROOT=/data3/xiaodan8/FineGym/RGB
trap 'release_card $tag' EXIT
hold_card $tag
wait_free 7000 360 || echo "[$(date '+%F %T')] $tag: no 7 GB free after 6 h, launching anyway" >> logs/$tag.wait.log
run_eval() {  # run_eval <name> <config> <ckpt> <cache>
  echo "[$(date '+%F %T')] $1: start" >> logs/$tag.log
  A6_EVAL_SAVE_PRED=1 deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config $2 \
    --output $1 --init_from $3 --freeze_except video_subtree_attn --feature_cache $4 --eval_only > logs/$1.log 2>&1
  echo "[$(date '+%F %T')] $1: rc=$? $(grep -o 'predictions dumped with prefix [^ ]*' logs/$1.log | tail -1) $(grep -o "\[eval_only\] {.*" logs/$1.log | tail -1 | cut -c1-200)" >> logs/$tag.log
}
if [ $only != fg ]; then
  run_eval fdpred configs/finediving/v3/fd_champ_stage3_nested_s1_1gpu.yaml ckpt/fd_champ_stage3_nested_s1_a4000_fda_s1_nested/vit_best_model /data3/xiaodan8/FineDiving/cache_fd3_s1 &
  _p=$!; sleep 180; release_card $tag; wait $_p
fi
if [ $only != fd ]; then
  run_eval fgpred configs/finegym_t3_champ_stage3_vsa.yaml ckpt/finegym_t3_champ_stage3_vsa_fg3_champ3g_vsa/vit_best_model /data3/xiaodan8/FineGym/cache_fg3_champ2 &
  _p=$!; sleep 180; release_card $tag; wait $_p
fi
echo "[$(date '+%F %T')] $tag: DONE" >> logs/$tag.log
