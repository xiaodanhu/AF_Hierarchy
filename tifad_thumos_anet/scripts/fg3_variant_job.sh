#!/bin/bash
# FineGym l_d=3 stage-3 VARIANT (or champion) + boundary-jitter curve on ONE GPU (dispatcher sets CUDA_VISIBLE_DEVICES).
# usage: bash scripts/fg3_variant_job.sh <tag> <config under configs/> [train|evalonly]
#   train   : train the variant from the stage-2 champion on cached CLIP features, then jitter evals 0/0.25/0.5/1.0
#   evalonly: jitter evals on an existing checkpoint dir ckpt/<cfg>_<tag>/vit_best_model (e.g. the champion fg3_champ3g_vsa)
tag=$1; cfg=$2; mode=${3:-train}
cd /data3/xiaodan8/actionformer6 || exit 1
export FINEGYM_DATA_ROOT=/data3/xiaodan8/FineGym FINEGYM_RGB_ROOT=/data3/xiaodan8/FineGym/RGB
CK2=ckpt/finegym_t3_champ_stage2_fg3_champ2/vit_best_model
CACHE=/data3/xiaodan8/FineGym/cache_fg3_champ2
CK=ckpt/${cfg%.yaml}_$tag/vit_best_model
wait_free() { for _i in $(seq 1 240); do _f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0}); [ "${_f:-0}" -ge 10000 ] && return; sleep 60; done; }
if [ "$mode" = train ] && [ ! -d "$CK" ]; then
  wait_free
  deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/$cfg --output $tag --init_from $CK2 \
    --freeze_except video_subtree_attn --feature_cache $CACHE --select_metric video_held_mAP -c 1 > logs/$tag.log 2>&1
fi
[ -d "$CK" ] || { echo "no checkpoint: $CK"; exit 1; }
for d in 0 0.25 0.5 1.0; do
  wait_free
  A6_VSA_JITTER=$d deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/$cfg --output ${tag}jit${d/./} \
    --init_from $CK --freeze_except video_subtree_attn --feature_cache $CACHE --eval_only > logs/${tag}jit${d/./}.log 2>&1
  grep -h "ZSL video" logs/${tag}jit${d/./}.log | tail -2
done
