#!/bin/bash
# FineDiving EVAL-ONLY of a stage-3 checkpoint on ONE GPU (dispatcher sets CUDA_VISIBLE_DEVICES),
# optionally with the boundary-jitter probe (A6_VSA_JITTER=<delta> chunk lengths; 0 = control).
# usage: bash scripts/fd_eval_job.sh <tag> <config under configs/finediving/v3/> <ckpt tag dir> <delta>
#   e.g. bash scripts/fd_eval_job.sh fdjit05 fd_champ_stage3_nested_s1_1gpu.yaml ckpt/fd_champ_stage3_nested_s1_1gpu_fdK3ref/vit_best_model 0.5
tag=$1; cfg=$2; ck=$3; delta=${4:-0}
cd /data3/xiaodan8/actionformer6 || exit 1
CACHE=/data3/xiaodan8/FineDiving/cache_fd3_s1
for _i in $(seq 1 180); do
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0})
  [ "${_free:-0}" -ge 8000 ] && break
  sleep 60
done
A6_VSA_JITTER=$delta deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/finediving/v3/$cfg \
  --output $tag --init_from $ck --freeze_except video_subtree_attn --feature_cache $CACHE --eval_only > logs/$tag.log 2>&1
grep -h "ZSL video\|eval_only" logs/$tag.log | tail -3
