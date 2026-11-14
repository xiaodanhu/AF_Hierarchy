#!/bin/bash
# FineDiving v3 chain on 3x RTX A4000 (16 GB).
#
# Phases:
#   champ3  = stage 1, two-level champion
#   cache3  = frozen CLIP feature cache for stage 3
#   nested3 = + nested video ASA K=3 x Kp=5
#             (its saved "best" checkpoint is the epoch-0 eval, because selection uses unseen mAP)
#
# Batch: micro-batch 6 per GPU x 3 GPUs x 2 accumulation = 36, learning rate 8.49e-5 = 8e-5 * sqrt(36/32).
set -u
cd /data3/xiaodan8/actionformer6 || exit 1
GPUS=${GPUS:-0,1,2}
FD_ROOT=${FD_ROOT:-/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving}
CACHE=${CACHE:-$FD_ROOT/../cache_fd3_s1}
C=configs/finediving/v3
LOG=logs/fd_chain_a4000.log
mkdir -p logs ckpt

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ds() { CUDA_VISIBLE_DEVICES=$GPUS deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py "$@"; }

run_phase() {   # run_phase <name> <args...>
  local name=$1; shift
  [ -f "ckpt/fda_${name}_done.marker" ] && { log "$name already done, skipping"; return 0; }
  local cfgstem outname resume=""
  cfgstem=$(basename "$(echo "$@" | sed -n 's/.*--config \([^ ]*\).*/\1/p')" .yaml)
  outname=$(echo "$@" | sed -n 's/.*--output \([^ ]*\).*/\1/p')
  [ -f "ckpt/${cfgstem}_${outname}/latest" ] && resume="--resume $(cat "ckpt/${cfgstem}_${outname}/latest")"
  [ -n "$resume" ] && log "$name: resuming from $resume"
  log "=== $name starting on GPUs $GPUS ==="
  ds "$@" $resume > "logs/fda_${name}.log" 2>&1; local rc=$?
  log "=== $name rc=$rc ==="
  grep -h "\[ZSL video\]  \(seen\|held-out\)" "logs/fda_${name}.log" | tail -2 | tee -a "$LOG"
  if [ "$rc" -eq 0 ]; then touch "ckpt/fda_${name}_done.marker"; else
    grep -hE "Error|Traceback|out of memory" "logs/fda_${name}.log" | tail -3 | tee -a "$LOG"; return 1; fi
}

log "=== FineDiving chain starting (split 1, keep protocol, 3x A4000 recipe) ==="

run_phase champ3 --config $C/fd_champ_s1_a4000.yaml --output fda_s1_champ || exit 1

CK=ckpt/fd_champ_s1_a4000_fda_s1_champ/vit_best_model
if [ ! -f ckpt/fda_cache3_done.marker ]; then
  log "=== cache3 -> $CACHE (from $CK) ==="
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} python scripts/cache_fg3_clip_feats.py \
      --config $C/fd_champ_stage3_nested_s1_a4000.yaml --ckpt "$CK" --out "$CACHE" \
      --verify 8 --workers 12 --batch 4 > logs/fda_cache3.log 2>&1
  rc=$?; log "cache3 rc=$rc"
  grep -hE "\[verify .* worst|Traceback|Error" logs/fda_cache3.log | tail -2 | tee -a "$LOG"
  [ "$rc" -eq 0 ] && [ -f "$CACHE/meta.json" ] && touch ckpt/fda_cache3_done.marker || { log "cache3 failed"; exit 1; }
fi

run_phase nested3 --config $C/fd_champ_stage3_nested_s1_a4000.yaml --output fda_s1_nested \
  --init_from "$CK" --freeze_except video_subtree_attn --feature_cache "$CACHE" \
  --select_metric video_held_mAP -c 1 || exit 1

log "=== chain complete ==="
log "stage 1 (expect ~69.2 / 54.7):"; grep -h "\[ZSL video\]" logs/fda_champ3.log | tail -2 | tee -a "$LOG"
log "stage 3 FINAL epoch (expect ~74.9 / 53.9):"; grep -h "\[ZSL video\]" logs/fda_nested3.log | tail -2 | tee -a "$LOG"
