#!/bin/bash
# ANet h19 program: vanilla anchors + h19 arm, 10 splits each (75/25).
cd /data3/xiaodan8/tifad

H19="TIFAD_PROMPT_ENSEMBLE=configs/anet_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 TIFAD_ARITH=configs/anet_arith.json"
declare -A R=(
  [van]=""
  [h19]="$H19"
)

rm -f /tmp/anet_tasks.txt
for k in h19 van; do for n in 0 1 2 3 4 5 6 7 8 9; do
  [ -f "ckpt/anet_i3d_75_${k}_${n}/mAP_best.pth.tar" ] && continue
  echo "$k $n" >> /tmp/anet_tasks.txt
done; done
split -n r/8 -d /tmp/anet_tasks.txt /tmp/anet_lane_

lane() {
  local lane_file=$1 gpu=$2
  while read -r k n; do
    [ -z "$k" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu env ${R[$k]} \
      python train_eval.py configs/anet_i3d_75.yaml --output $k --n $n \
      > logs/anet_${k}_75_s${n}.log 2>&1
  done < "$lane_file"
}
for i in $(seq 0 7); do
  lane /tmp/anet_lane_$(printf %02d $i) $((i % 4)) &
done
wait
echo "=== ANET PROGRAM COMPLETE $(date) ==="
echo "--- h19 per split ---"
grep -H 'max(mAP_list)' logs/anet_h19_75_s*.log
echo "--- vanilla per split ---"
grep -H 'max(mAP_list)' logs/anet_van_75_s*.log
