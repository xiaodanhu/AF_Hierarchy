#!/bin/bash
# FineGym t3 training job, launched by the tifad GPU dispatcher
# (queue/jobs.txt: TAG<TAB>CUDA_VISIBLE_DEVICES is set by the dispatcher).
# Usage: run_t3_job.sh <base|tifad|champ> [master_port]
#
# The dispatcher places a job whenever >=4500 MB are free, but this RGB
# end-to-end CLIP training needs ~8-12 GB — so first WAIT until >=9 GB are
# free on our assigned GPU (THUMOS queue jobs draining), then launch.
set -u
sys=$1
port=${2:-29631}
cd /data3/xiaodan8/actionformer6 || exit 1
mkdir -p logs output ckpt
g=${CUDA_VISIBLE_DEVICES:-0}
# one fg3 training per GPU at a time: hold a per-GPU lock for the whole run
# (fd 9 is inherited by deepspeed, so the lock lives until training exits)
exec 9>"/tmp/fg3_gpu${g}.lock"
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
  if [ "$free" -ge 9000 ]; then break; fi
  sleep 120
done
echo "[run_t3_job] starting fg3_${sys} on GPU $g (free=${free}MB, port=$port)"
exec deepspeed --master_port="$port" train_shard.py \
  --config "configs/finegym_t3_${sys}.yaml" \
  --output "fg3_${sys}" \
  > "logs/fg3_${sys}.log" 2>&1
