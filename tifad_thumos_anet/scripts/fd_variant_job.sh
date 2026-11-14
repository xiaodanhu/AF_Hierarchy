#!/bin/bash
# FineDiving stage-3 VARIANT + boundary-jitter curve on ONE GPU (dispatcher sets CUDA_VISIBLE_DEVICES):
# train the variant (fd_stage3_job.sh), then evaluate its best checkpoint at jitter 0 / 0.25 / 0.5 / 1.0.
# usage: bash scripts/fd_variant_job.sh <tag> <config under configs/finediving/v3/>
#   e.g. bash scripts/fd_variant_job.sh fdNOCHAIN fd_champ_stage3_nested_s1_1gpu_NOCHAIN.yaml
tag=$1; cfg=$2
cd /data3/xiaodan8/tifad || exit 1
CK=/data3/xiaodan8/actionformer6/ckpt/${cfg%.yaml}_$tag/vit_best_model
[ -d "$CK" ] || bash scripts/fd_stage3_job.sh $tag $cfg
[ -d "$CK" ] || { echo "no checkpoint after training: $CK"; exit 1; }
for d in 0 0.25 0.5 1.0; do
  bash scripts/fd_eval_job.sh ${tag}jit${d/./} $cfg ckpt/${cfg%.yaml}_$tag/vit_best_model $d
done
