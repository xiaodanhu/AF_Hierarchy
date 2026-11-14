#!/bin/bash
cd /data3/xiaodan8/actionformer6
deepspeed --num_gpus=4 --master_port=12501 train_shard.py \
  --config configs/activitynet_i3d_zsl_split1_75_strict_v2b.yaml \
  --output champ_s1 -p 20 -c 5 > logs/anet_champ_s1.log 2>&1
deepspeed --num_gpus=4 --master_port=12502 train_shard.py \
  --config configs/activitynet_i3d_zsl_split2_75_strict_v2b.yaml \
  --output champ_s2 -p 20 -c 5 > logs/anet_champ_s2.log 2>&1
echo "champion splits done $(date)"
