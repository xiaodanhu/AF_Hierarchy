#!/bin/bash
# FineGym champion WITHOUT ancestor supervision (stage 1 only), GPUs 1,2,3
# (GPU3 allowed since 2026-09-18). GPU0 stays with the tifad dispatcher, whose
# loop is restricted to GPU0 while this runs. Waits until GPUs 1-3 have >=13GB
# free, retries on nonzero exit.
cd /data3/xiaodan8/actionformer6
until [ -f ckpt/fg3_champ_noas_done.marker ]; do
  while :; do
    ok=1
    for g in 1 2 3; do
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      [ "$f" -ge 13000 ] || ok=0
    done
    [ "$ok" -eq 1 ] && break
    sleep 120
  done
  echo "=== FG3 champ_noas starting $(date) ===" >> logs/fg3_chain.log
  CUDA_VISIBLE_DEVICES=1,2,3 deepspeed --master_port=2972$((RANDOM % 10)) train_shard.py \
    --config configs/finegym_t3_champ_noas.yaml --output fg3_champ_noas \
    > logs/fg3_champ_noas.log 2>&1
  rc=$?
  echo "=== FG3 champ_noas rc=$rc $(date) ===" >> logs/fg3_chain.log
  if [ "$rc" -eq 0 ]; then touch ckpt/fg3_champ_noas_done.marker; else sleep 600; fi
done
