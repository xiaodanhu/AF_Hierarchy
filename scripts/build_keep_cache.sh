#!/bin/bash
cd /home/jupyter/xhu3/video/Dec14/af_fg3_v3 || exit 1
export PATH=/home/jupyter/xhu3/.conda/envs/qwen/bin:$PATH FINEGYM_RGB_ROOT=/home/jupyter/xhu3/video/dataset/finegym/RGB256png FINEGYM_FRAME_EXT=.png
CACHE=/home/jupyter/xhu3/video/dataset/finegym/cache_fg2_s1_keep
CUDA_VISIBLE_DEVICES=3 python scripts/cache_fg3_clip_feats.py --config configs/quick/fg2_s1_keep_stage3_nested.yaml --ckpt ckpt/fg2_s1_keep_fg2_s1_quick_keep/epoch_005 --out $CACHE --verify 8 --workers 12 --batch 8 > logs/fg2_s1_keep_cache.log 2>&1
rc=$?
echo "[$(date '+%F %T')] s1_keep_cache rc=$rc (attempt 2, stage-1-based config)" >> logs/fg2_chain.log
