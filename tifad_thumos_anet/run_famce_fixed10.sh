#!/bin/bash
# Complete fixed-weight famce splits 3-9 for the 10-split fixed-vs-learned contrast.
cd /data3/xiaodan8/tifad
until ! pgrep -f "train_eval.py" >/dev/null && [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 600; done
sleep 900
B="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json"
rm -f /tmp/ff.txt; for n in 3 4 5 6 7 8 9; do [ -f ckpt/thumos14_i3d_famce_$n/mAP_best.pth.tar ] || echo $n >> /tmp/ff.txt; done
split -n r/6 -d /tmp/ff.txt /tmp/ffl_ 2>/dev/null
lane() { while read -r n; do [ -z "$n" ] && continue; CUDA_VISIBLE_DEVICES=$2 env $B python train_eval.py configs/thumos14_i3d.yaml --output famce --n $n > logs/tifad_famce_75_s$n.log 2>&1; done < "$1"; }
for i in 0 1 2; do lane /tmp/ffl_$(printf %02d $i) 2 & done
for i in 3 4 5; do [ -f /tmp/ffl_$(printf %02d $i) ] && lane /tmp/ffl_$(printf %02d $i) 3 & done
wait
echo "=== FAMCE FIXED 10SPLIT DONE $(date) ==="
grep -H 'max(mAP_list)' logs/tifad_famce_75_s*.log
