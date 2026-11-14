#!/bin/bash
# Hierarchy-attention A/Bs: HTCA / HAF / both, THUMOS 75/25 splits 0-2.
cd /data3/xiaodan8/tifad
B="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3"
HJ="TIFAD_HTCA_ATTRS=configs/thumos14_ttb_attrs.json TIFAD_HTCA_FAMS=configs/thumos14_fam_map.json"
declare -A R=(
  [htca]="$B $HJ TIFAD_HTCA=1"
  [haf]="$B $HJ TIFAD_HAF=0.2"
  [hh]="$B $HJ TIFAD_HTCA=1 TIFAD_HAF=0.2"
)
rm -f /tmp/hab.txt
for k in htca haf hh; do for n in 0 1 2; do
  [ -f "ckpt/thumos14_i3d_${k}_${n}/mAP_best.pth.tar" ] && continue
  echo "$k $n" >> /tmp/hab.txt
done; done
split -n r/3 -d /tmp/hab.txt /tmp/habl_
lane() {
  local lf=$1 gpu=$2
  while read -r k n; do
    [ -z "$k" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu env ${R[$k]} \
      python train_eval.py configs/thumos14_i3d.yaml --output $k --n $n \
      > logs/thumos_${k}_75_s${n}.log 2>&1
  done < "$lf"
}
lane /tmp/habl_00 2 &
lane /tmp/habl_01 3 &
lane /tmp/habl_02 3 &
wait
echo "=== HIER AB DONE $(date) ==="
for k in htca haf hh; do echo "-- $k"; grep -H 'max(mAP_list)' logs/thumos_${k}_75_s*.log; done
