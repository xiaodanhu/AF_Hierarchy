#!/bin/bash
# Sequential 4-GPU sweep over the remaining ActivityNet ZSL splits.
# split0_75 already completed (logs/anet_zsl_s0_75.log). Aggregate with:
#   grep '\[ZSL video\]' logs/anet_zsl_s*.log
cd /data3/xiaodan8/actionformer6
port=12500
for s in 0 1 2 3 4 5 6 7 8 9; do
  for tag in 75 50; do
    [ "${s}_${tag}" = "0_75" ] && continue
    name="anet_zsl_s${s}_${tag}"
    [ -f "logs/${name}.done" ] && continue
    echo "[sweep] starting ${name} on port ${port} at $(date)"
    deepspeed --num_gpus=4 --master_port=${port} train_shard.py \
      --config configs/activitynet_i3d_zsl_split${s}_${tag}.yaml \
      --output "${name}" -p 20 -c 5 > "logs/${name}.log" 2>&1
    echo "exit=$? $(date)" > "logs/${name}.done"
    port=$((port+1))
  done
done
echo "[sweep] all done at $(date)"
