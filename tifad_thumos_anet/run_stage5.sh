#!/bin/bash
# STAGE 5 — phrase-level zero-shot protocol (see STAGE5_PROTOCOL.md).
# 2 arms x 5 splits = 10 runs, 6 lanes: 3 per GPU on GPUs 2 and 3 ONLY
# (GPUs 0/1 are reserved — never schedule there).
cd /data3/xiaodan8/tifad
mkdir -p logs ckpt

ENVCOMMON="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_phrasemix_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3"
declare -A ARMENV=(
  [A]="$ENVCOMMON"                     # flat-phrase baseline
  [B]="$ENVCOMMON TIFAD_SEM_DEGRADE=0" # ours: actions as attributes
)
declare -A ARMCFG=(
  [A]="thumos14_i3d_phraseflat"
  [B]="thumos14_i3d_phrase"
)
declare -A ARMTAG=( [A]="s5A" [B]="s5B" )

ckdir() { echo "ckpt/${ARMCFG[$1]}_$2_${ARMTAG[$1]}_$2"; }

# task list (interleaved so round 1 covers both arms); skip finished runs
rm -f /tmp/s5_tasks.txt
for n in 0 1 2 3 4; do for arm in B A; do
  [ -f "$(ckdir $arm $n)/DONE" ] && continue
  echo "$arm $n" >> /tmp/s5_tasks.txt
done; done
if [ ! -s /tmp/s5_tasks.txt ]; then echo "nothing to do"; else
split -n r/6 -d /tmp/s5_tasks.txt /tmp/s5_lane_

lane() {
  local lane_file=$1 gpu=$2
  while read -r arm n; do
    [ -z "$arm" ] && continue
    local cd_=$(ckdir $arm $n) log=logs/s5_${arm}_s${n}.log
    rm -rf "$cd_"   # partial run: restart clean
    CUDA_VISIBLE_DEVICES=$gpu env ${ARMENV[$arm]} \
      python train_eval.py configs/${ARMCFG[$arm]}_${n}.yaml \
      --output ${ARMTAG[$arm]} --n $n > "$log" 2>&1
    grep -q "All done!" "$log" && touch "$cd_/DONE"
  done < "$lane_file"
}

GPUS=(2 2 2 3 3 3)
for i in 0 1 2 3 4 5; do
  f=/tmp/s5_lane_$(printf %02d $i)
  [ -f "$f" ] && lane "$f" ${GPUS[$i]} &
done
wait
fi
echo "=== STAGE 5 runs complete $(date) ==="

# ---- comparison table ----
echo
echo "=== STAGE 5: novel-phrase mAP (avg tIoU 0.3:0.7, best epoch) ==="
printf "%-6s %-28s %10s %10s\n" split held-out "A(flat)" "B(ours)"
suma=0; sumb=0; cnt=0
for n in 0 1 2 3 4; do
  held=$(paste -sd'+' splits/train_phrase/THUMOS14/test/split_${n}.list)
  a=$(grep -oP 'max\(mAP_list\) \K[0-9.]+' logs/s5_A_s${n}.log 2>/dev/null)
  b=$(grep -oP 'max\(mAP_list\) \K[0-9.]+' logs/s5_B_s${n}.log 2>/dev/null)
  printf "%-6s %-28s %10s %10s\n" $n "$held" "${a:--}" "${b:--}"
  [ -n "$a" ] && [ -n "$b" ] && { suma=$(echo "$suma+$a"|bc); sumb=$(echo "$sumb+$b"|bc); cnt=$((cnt+1)); }
done
if [ $cnt -gt 0 ]; then
  printf "%-6s %-28s %10s %10s\n" mean "($cnt splits)" \
    $(echo "scale=2;$suma/$cnt"|bc) $(echo "scale=2;$sumb/$cnt"|bc)
fi
echo
echo "--- per-tIoU (0.3..0.7) at best epoch ---"
for n in 0 1 2 3 4; do
  for arm in A B; do
    t=$(grep 'tiou_mAP_list_best' logs/s5_${arm}_s${n}.log 2>/dev/null | tail -1)
    [ -n "$t" ] && echo "$arm s$n: $t"
  done
done
