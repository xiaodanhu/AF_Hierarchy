#!/bin/bash
# Representation-validation probe (phase probe acc10 + Kendall's tau, scripts/repvalid_variant.py)
# on an arbitrary frozen checkpoint family, splits 0-2. Dispatcher-friendly (no quotes needed):
#   bash scripts/probe_job.sh <tag> <ckpt pattern with {n}> ENV=VAL ENV=VAL ...
# e.g. bash scripts/probe_job.sh cfree2 ckpt/thumos14_i3d_cfree2_{n}/mAP_best.pth.tar TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble_cfree.json ...
tag=$1; ck=$2; shift 2
cd /data3/xiaodan8/tifad || exit 1
for _i in $(seq 1 180); do
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0})
  [ "${_free:-0}" -ge 6000 ] && break
  sleep 60
done
PYTHONUTF8=1 LC_ALL=C.UTF-8 python scripts/repvalid_variant.py --tag $tag --ckpt "$ck" --config configs/thumos14_i3d.yaml \
  --envs "$*" --splits 0 1 2 > logs/probe_$tag.log 2>&1
grep -h "mean over splits" -A 2 logs/probe_$tag.log | tail -3
