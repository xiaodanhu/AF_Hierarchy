#!/bin/bash
# Temporal-order probe (actionformer6 scripts/order_probe_a6.py) on ONE GPU via the dispatcher.
# usage: bash scripts/a6_probe_job.sh <fd|fg> <tag> <config> <ckpt dir> <cache dir>
ds=$1; tag=$2; cfg=$3; ck=$4; cache=$5
cd /data3/xiaodan8/actionformer6 || exit 1
export FINEGYM_DATA_ROOT=/data3/xiaodan8/FineGym FINEGYM_RGB_ROOT=/data3/xiaodan8/FineGym/RGB
for _i in $(seq 1 180); do
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0})
  [ "${_free:-0}" -ge 5000 ] && break
  sleep 60
done
mkdir -p logs/order_probe
python scripts/order_probe_a6.py --dataset $ds --tag $tag --config $cfg --ckpt $ck --cache $cache --device cuda > logs/order_probe/$tag.log 2>&1
grep -E "^TAU|^POSITION|^SUBACTION" logs/order_probe/$tag.log
