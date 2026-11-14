#!/bin/bash
# Chain: (A) 50-epoch transductive v2 run, then (B) 50-epoch STRICT ablation
# (no held-out annotations for specialist or phrase supervision).
cd /data3/xiaodan8/actionformer6
deepspeed --num_gpus=4 --master_port=12482 train_shard.py \
  --config configs/activitynet_i3d_zsl_split0_75.yaml \
  --output v2_50ep -p 20 -c 5 > logs/anet_v2_50ep.log 2>&1
echo "A exit=$? $(date)" > logs/anet_v2_50ep.done
deepspeed --num_gpus=4 --master_port=12483 train_shard.py \
  --config configs/activitynet_i3d_zsl_split0_75_strict.yaml \
  --output strict_50ep -p 20 -c 5 > logs/anet_strict_50ep.log 2>&1
echo "B exit=$? $(date)" > logs/anet_strict_50ep.done
