#!/bin/bash
# FineDiving stage-3 variant on ONE GPU (dispatcher sets CUDA_VISIBLE_DEVICES).
# usage: bash scripts/fd_stage3_job.sh <tag> <config under configs/finediving/v3/>
tag=$1; cfg=$2
cd /data3/xiaodan8/actionformer6 || exit 1
# 2026-11-07: these jobs need ~7 GB; the dispatcher's 6 GB launch gate is not
# enough when the card is shared, so hold the slot until >= 8 GB is free
# (up to 3 h) before touching the GPU. fdK5LB OOMed twice without this.
for _i in $(seq 1 180); do
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0})
  [ "${_free:-0}" -ge 8000 ] && break
  sleep 60
done
CK=ckpt/fd_champ_s1_a4000_fda_s1_champ/vit_best_model
CACHE=/data3/xiaodan8/FineDiving/cache_fd3_s1
deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/finediving/v3/$cfg \
  --output $tag --init_from $CK --freeze_except video_subtree_attn --feature_cache $CACHE \
  --select_metric video_held_mAP -c 1 > logs/$tag.log 2>&1
