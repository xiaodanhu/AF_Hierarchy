#!/bin/bash
# LLM step-count statistics for the K story GPU 8-bit version.
# The CPU run was ~26 min/class once two 7B models shared the box; 8-bit on
# one A4000 is ~2 s/call. Needs ~10 GB: hold the dispatcher slot until the
# card has it (up to 6 h), then run the remaining datasets. gen_step_count.py
# resumes (skips classes already in the output), so re-running is safe.
# usage (dispatcher): steps_gpu<TAB>bash scripts/steps_job.sh
cd /data3/xiaodan8/tifad || exit 1
P=/data/xiaodan8/anaconda3/envs/hierarchy/bin/python
A6=/data3/xiaodan8/actionformer6
for _i in $(seq 1 360); do
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0})
  [ "${_free:-0}" -ge 10000 ] && break
  sleep 60
done
$P scripts/gen_step_count.py --device 8bit --n 5 --ens $A6/configs/finediving/fd_dive_ensemble.json   --out configs/finediving_step_counts.json     > logs/steps_finediving.log 2>&1
$P scripts/gen_step_count.py --device 8bit --n 5 --ens $A6/configs/finegym_phrase_ensemble.json       --out configs/finegym_phrase_step_counts.json > logs/steps_finegym_phrase.log 2>&1
$P scripts/gen_step_count.py --device 8bit --n 3 --ens $A6/configs/finegym_qwen_ensemble.json         --out configs/finegym_step_counts.json        > logs/steps_finegym.log 2>&1
$P scripts/gen_step_count.py --device 8bit --n 3 --ens configs/anet_qwen_ensemble.json                --out configs/anet_step_counts.json           > logs/steps_anet.log 2>&1
