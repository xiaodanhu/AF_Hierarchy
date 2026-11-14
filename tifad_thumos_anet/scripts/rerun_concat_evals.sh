#!/bin/bash
# Re-run the concatenation-control evaluations (cpair / cshort) that died of CUDA OOM inside
# their dispatcher jobs: the in-job eval.py started while 3 trainings shared GPU 2 (the 6 GB gate applies only at
# job launch). Sequential on GPU 2, waits for >= 2500 MB free before each; skips evals that already have a result;
# for splits still training it waits for "All done!" and for the in-job eval to exit first.
cd /data3/xiaodan8/tifad || exit 1
P2=/data/xiaodan8/anaconda3/envs/parsing2/bin/python
COMMON="TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3"
for job in cpair:configs/thumos14_qwen_ensemble_concatpair.json cshort:configs/thumos14_qwen_ensemble_concatshort.json; do
  tag=${job%%:*}; ens=${job#*:}; [ -f "$ens" ] || { echo "missing $ens"; exit 1; }
  for n in 0 1 2 3 4 5 6 7 8 9; do
    L=logs/${tag}e_s$n.log
    grep -q "Avearge mAP" $L 2>/dev/null && continue
    until grep -q "All done!" logs/${tag}_s$n.log 2>/dev/null; do sleep 120; done
    while pgrep -f "eval.py configs/thumos14_i3d.yaml ckpt/thumos14_i3d_${tag}_$n " >/dev/null; do sleep 60; done
    grep -q "Avearge mAP" $L 2>/dev/null && continue
    [ -f ckpt/thumos14_i3d_${tag}_$n/mAP_best.pth.tar ] || { echo "[$(date '+%T')] no checkpoint for $tag s$n"; continue; }
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)" -ge 2500 ]; do sleep 60; done
    echo "[$(date '+%T')] eval $tag s$n"
    CUDA_VISIBLE_DEVICES=2 env TIFAD_PROMPT_ENSEMBLE=$ens $COMMON $P2 eval.py configs/thumos14_i3d.yaml ckpt/thumos14_i3d_${tag}_$n --n $n > $L 2>&1
    grep "Avearge mAP" $L || echo "  FAILED again: $(tail -1 $L | cut -c1-140)"
  done
done
echo "all concat evals done"
