#!/bin/bash
cd /data3/xiaodan8/actionformer6
port=12510
for sp in 3 4 5 6 7 8 9; do
  [ -f "logs/anet_champ_s${sp}.done" ] && continue
  deepspeed --num_gpus=4 --master_port=${port} train_shard.py \
    --config configs/activitynet_i3d_zsl_split${sp}_75_strict_v2b.yaml \
    --output champ_s${sp} -p 20 -c 5 > logs/anet_champ_s${sp}.log 2>&1
  echo "exit=$? $(date)" > logs/anet_champ_s${sp}.done
  port=$((port+1))
done
echo "champion 10-split sweep complete $(date)"
