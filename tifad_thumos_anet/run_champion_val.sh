#!/bin/bash
cd /data3/xiaodan8/tifad
B="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 TIFAD_HTCA_FAMS=configs/thumos14_fam_map.json"
declare -A R=(
  [famce_uw]="$B TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1"
  [sibcon]="$B TIFAD_SIBCON=0.3"
  [uwsib]="$B TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_SIBCON=0.3"
)
rm -f /tmp/cv.txt
for n in 3 4 5 6 7 8 9; do echo "famce_uw $n" >> /tmp/cv.txt; echo "sibcon $n" >> /tmp/cv.txt; done
for n in 0 1 2; do echo "uwsib $n" >> /tmp/cv.txt; done
grep -c . /tmp/cv.txt
split -n r/6 -d /tmp/cv.txt /tmp/cvl_
lane() { while read -r k n; do [ -z "$k" ] && continue; [ -f "ckpt/thumos14_i3d_${k}_${n}/mAP_best.pth.tar" ] && continue; CUDA_VISIBLE_DEVICES=$2 env ${R[$k]} python train_eval.py configs/thumos14_i3d.yaml --output $k --n $n > logs/tifad_${k}_75_s${n}.log 2>&1; done < "$1"; }
lane /tmp/cvl_00 2 & lane /tmp/cvl_01 2 & lane /tmp/cvl_02 2 & lane /tmp/cvl_03 3 & lane /tmp/cvl_04 3 & lane /tmp/cvl_05 3 &
wait
echo "=== CHAMPION VAL DONE $(date) ==="
for k in famce_uw sibcon uwsib; do echo "-- $k:"; grep -H 'max(mAP_list)' logs/tifad_${k}_75_s*.log; done
