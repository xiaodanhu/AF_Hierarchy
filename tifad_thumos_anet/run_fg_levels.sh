#!/bin/bash
# FineGym PER-LEVEL hierarchical evaluation program (assets from
# scripts/gen_fg_level_splits.py -- see its docstring for the full protocol).
# ACTIVITY = gym99 set (14, readable names in configs/finegym_set_map.json);
# ACTION = element (99); 5 splits hold out 3 whole sets each
# (seeds RandomState(13+n), pairwise-distinct apparatus, VT never drawn).
# PRIMARY METRIC: activity-level unseen mAP (avg tIoU 0.3:0.1:0.7 over the
# 3 held-out set names), the 'max(mAP_list)' line of each log.
#
#   arm A (armA)  ACTIVITY-LEVEL TRAINING: train on the 11 seen sets as flat
#                 activity classes (Validation relabeled element->set in
#                 finegym_setflat_{n}.json), eval on the 3 held-out sets.
#   arm B (armB)  ELEMENT-TRAINED + COMPOSED-TO-ACTIVITY EVAL: train on the
#                 seen-set ELEMENTS (original element labels), eval with
#                 FINEGYM_LEVEL_COMPOSE=1 -- each held-out activity expands
#                 to [set name + member element names] in the text tower and
#                 the per-snippet logits are max-pooled back to the
#                 activity (element level = auxiliary support).
#   control       after training, arm B ckpts are re-evaluated WITHOUT the
#                 compose expansion (activity name only) to isolate the
#                 compose contribution.
#
# GPUs 2/3 ONLY (GPUs 0/1 are STRICTLY RESERVED -- never schedule there),
# 1 lane per GPU. DO NOT LAUNCH while other chains run; this script
# self-waits (below) and is meant to be started with:
#   cd /data3/xiaodan8/tifad && nohup bash run_fg_levels.sh > logs/fg_levels_runner.log 2>&1 &
cd /data3/xiaodan8/tifad
mkdir -p logs ckpt

# ---- SELF-WAITING gate: proceed only when ALL hold CONTINUOUSLY 15 min ----
#  * no train_eval.py / self_train.py / eval.py processes (bracket-safe
#    patterns so the pgrep never matches itself or this script)
#  * GPUs 2 and 3 each < 1GB used
idle_since=0
while true; do
  ok=1
  pgrep -f "[t]rain_eval.py" >/dev/null && ok=0
  pgrep -f "[s]elf_train.py" >/dev/null && ok=0
  pgrep -f "[e]val.py" >/dev/null && ok=0
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -dc 0-9)
  m3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -dc 0-9)
  [ -n "$m2" ] && [ "$m2" -lt 1000 ] || ok=0
  [ -n "$m3" ] && [ "$m3" -lt 1000 ] || ok=0
  if [ "$ok" -eq 1 ]; then
    now=$(date +%s)
    [ "$idle_since" -eq 0 ] && idle_since=$now
    [ $((now - idle_since)) -ge 900 ] && break
  else
    idle_since=0
  fi
  sleep 60
done
echo "=== FG LEVELS gate passed, starting $(date) ==="

# h19 base + per-arm extras. Both arms use the MERGED 113-key ensemble
# (element + set prompts) so activity names, element names, and the compose
# expansion all have 11-sentence entries. SEM_DEGRADE parent stems:
# arm A degrades activity -> apparatus stem (finegym_set_arith.json),
# arm B degrades element -> element parent stem (finegym_arith.json).
FGBASE="TIFAD_PROMPT_ENSEMBLE=configs/finegym_setmix_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3"
declare -A R=(
  [armA]="$FGBASE TIFAD_ARITH=configs/finegym_set_arith.json"
  [armB]="$FGBASE TIFAD_ARITH=configs/finegym_arith.json FINEGYM_LEVEL_COMPOSE=1 FINEGYM_LEVEL_MAP=configs/finegym_set_members.json"
)
declare -A CFG=(
  [armA]=finegym_i3d_setflat
  [armB]=finegym_i3d_setlevel
)

rm -f /tmp/fgl_tasks.txt
for k in armA armB; do for n in 0 1 2 3 4; do
  [ -f "ckpt/${CFG[$k]}_${n}_${k}_${n}/mAP_best.pth.tar" ] && continue
  echo "$k $n" >> /tmp/fgl_tasks.txt
done; done
split -n r/2 -d /tmp/fgl_tasks.txt /tmp/fgl_lane_
lane() {
  local lane_file=$1 gpu=$2
  while read -r k n; do
    [ -z "$k" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu env ${R[$k]} \
      python train_eval.py configs/${CFG[$k]}_${n}.yaml --output $k --n $n \
      > logs/fg_${CFG[$k]#finegym_i3d_}_${k}_s${n}.log 2>&1
  done < "$lane_file"
}
lane /tmp/fgl_lane_00 2 &
lane /tmp/fgl_lane_01 3 &
wait
echo "=== FG LEVELS TRAININGS DONE $(date) ==="

# ---- arm B compose-OFF control evals (same ckpt, activity name only) ----
for n in 0 1 2 3 4; do
  CK=ckpt/finegym_i3d_setlevel_${n}_armB_${n}
  [ -f "$CK/mAP_best.pth.tar" ] || { echo "missing $CK"; continue; }
  CUDA_VISIBLE_DEVICES=$((2 + n % 2)) env $FGBASE \
    python eval.py configs/finegym_i3d_setlevel_${n}.yaml $CK --n $n \
    > logs/fg_setlevel_armB_nocomp_s${n}.log 2>&1 &
  [ $((n % 2)) -eq 1 ] && wait
done
wait
echo "=== FG LEVELS COMPLETE $(date) ==="
echo "--- arm A (activity-trained) unseen activity mAP per split ---"
grep -H 'max(mAP_list)' logs/fg_setflat_armA_s*.log
echo "--- arm B (element-trained + compose) unseen activity mAP per split ---"
grep -H 'max(mAP_list)' logs/fg_setlevel_armB_s*.log
echo "--- arm B control (compose OFF) ---"
grep -H 'Avearge mAP' logs/fg_setlevel_armB_nocomp_s*.log
