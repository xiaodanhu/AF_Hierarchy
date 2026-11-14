#!/bin/bash
# actionformer6 EVAL-ONLY of a checkpoint on ONE GPU (dispatcher sets CUDA_VISIBLE_DEVICES), with optional envs
# (order controls A6_VSA_SHUFFLE=reverse|perm, A6_TEXT_SHUFFLE=..., jitter A6_VSA_JITTER=<delta>).
# usage: bash scripts/a6_eval_job.sh <fd|fg> <tag> <config path under configs/> <ckpt tag dir> [ENV=VAL ...]
#   e.g. bash scripts/a6_eval_job.sh fd fdK3_vrev finediving/v3/fd_champ_stage3_nested_s1_1gpu.yaml ckpt/fd_champ_stage3_nested_s1_1gpu_fdK3ref/vit_best_model A6_VSA_SHUFFLE=reverse
ds=$1; tag=$2; cfg=$3; ck=$4; shift 4
cd /data3/xiaodan8/actionformer6 || exit 1
if [ "$ds" = fd ]; then CACHE=/data3/xiaodan8/FineDiving/cache_fd3_s1; else CACHE=/data3/xiaodan8/FineGym/cache_fg3_champ2; export FINEGYM_DATA_ROOT=/data3/xiaodan8/FineGym FINEGYM_RGB_ROOT=/data3/xiaodan8/FineGym/RGB; fi
for _i in $(seq 1 240); do
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0})
  [ "${_free:-0}" -ge 8000 ] && break
  sleep 60
done
env "$@" deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/$cfg --output $tag \
  --init_from $ck --freeze_except video_subtree_attn --feature_cache $CACHE --eval_only > logs/$tag.log 2>&1
grep -h "ZSL video" logs/$tag.log | tail -2
