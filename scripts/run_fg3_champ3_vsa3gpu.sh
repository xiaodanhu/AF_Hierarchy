#!/bin/bash
# FineGym champion STAGE 3: video branch of the action--sub-action attention
# (libs/modeling/vsa.py) trained on the FROZEN stage-2 champion with cached
# CLIP features, using the deepspeed LAUNCHER across GPUs 1,2,3 (user,
# 2026-09-19: "FineGym needs to run with DeepSpeed on multiple GPU"),
# matching the 3-GPU pattern of scripts/run_fg3_chain.sh (stages 1/2). GPU 0
# is left for the tifad dispatcher's THUMOS/ActivityNet program
# (run_dispatcher.sh temporarily restricted to GPU 0 while this runs; restore
# 0 1 2 3 when this chain finishes).
# Supersedes scripts/run_fg3_champ3_vsa.sh (single-GPU version, stopped).
# Phases, each a full deepspeed launch on the same 3 GPUs:
#   1. cache   : frozen CLIP-encoder output of every train/test window
#                (scripts/cache_fg3_clip_feats.py, bitwise-verified; single
#                process is enough -- runs on GPU 1 alone)
#   2. inert   : stage-2 checkpoint, flag OFF, cached features, eval only
#                -> must reproduce logs/fg3_champ2.log (held-out 0.2520 /
#                seen 0.5850 video, 0.2450 / 0.5769 window)
#   2b. init   : stage-3 config (flag ON, zero-init) eval only -> same numbers
#   3. train   : stage 3, everything frozen except video_subtree_attn,
#                15 epochs, eval every epoch, best by held-out video mAP,
#                3-GPU DeepSpeed (effective batch 48, matching stages 1/2)
cd /data3/xiaodan8/actionformer6
PY=/data/xiaodan8/anaconda3/envs/parsing2/bin/python
CACHE=/data3/xiaodan8/FineGym/cache_fg3_champ2
CK2=ckpt/finegym_t3_champ_stage2_fg3_champ2/vit_best_model
LOG=logs/fg3_champ3_vsa3gpu.log
GPUS="1,3"   # 2026-09-19: GPU2 has an unrelated user job (train_gated.py, ~13GB); GPUs 1,2,3 waited 12h+ with no progress; use the two fully-free GPUs instead
IFS="," read -ra GPUS_ARR <<< "$GPUS"
MIN_FREE=${MIN_FREE:-13000}   # per-GPU, matching run_fg3_chain.sh's gate
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_gpus() {   # blocks until GPUs 1,2,3 each have >= MIN_FREE MB free
  while :; do
    ok=1
    for g in ${GPUS_ARR[@]}; do
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      [ "$f" -ge "$MIN_FREE" ] || ok=0
    done
    [ "$ok" -eq 1 ] && return
    sleep 60
  done
}

ds3() {   # ds3 <args...> -- deepspeed launcher across GPUS
  CUDA_VISIBLE_DEVICES=$GPUS deepspeed --master_port=$((29600 + RANDOM % 300)) \
    train_shard.py "$@"
}

log "=== stage 3 (video branch, 3-GPU deepspeed) chain starting; waiting for GPUs $GPUS (2-GPU fallback, GPU2 busy with another user's job) >= ${MIN_FREE} MB free each ==="
wait_gpus
log "GPUs $GPUS ready ($(for g in ${GPUS_ARR[@]}; do nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g | tr '\n' ' '; done)MB free)"

# ---- 1. cache (single process is enough; reuses the bitwise-verified cache if present) ----
if [ ! -f "$CACHE/meta.json" ]; then
  log "phase 1: caching frozen CLIP features -> $CACHE (log logs/fg3_champ3_cache.log)"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 $PY scripts/cache_fg3_clip_feats.py \
    --config configs/finegym_t3_champ_stage3_vsa.yaml --ckpt $CK2 --out $CACHE \
    --verify 8 --workers 24 --batch 8 > logs/fg3_champ3_cache.log 2>&1
  rc=$?; log "phase 1 rc=$rc in $((($(date +%s) - t0) / 60)) min"
  grep -h "\[verify\|done:" logs/fg3_champ3_cache.log | tee -a "$LOG"
  [ $rc -eq 0 ] || { log "caching failed; abort"; exit 1; }
else
  log "phase 1: cache present ($CACHE/meta.json), skipping"
fi

# ---- 2. byte-inert check: flag OFF, stage-2 ckpt, cached features ----
log "phase 2: byte-inert check (stage-2 config, flag off, cached features, eval only) -> logs/fg3_champ3g_inert.log"
t0=$(date +%s)
ds3 --config configs/finegym_t3_champ_stage2.yaml --output fg3_champ3g_inert \
  --init_from $CK2 --freeze_except text_pathway.phase_attn --feature_cache $CACHE \
  --eval_only > logs/fg3_champ3g_inert.log 2>&1
log "phase 2 rc=$? in $((($(date +%s) - t0) / 60)) min"
grep -h "\[ZSL video\]\|\[ZSL window\]\|Video-level action mAP\|Window-level action mAP" logs/fg3_champ3g_inert.log | tee -a "$LOG"

# ---- 2b. init-equality: flag ON (zero-init), eval only ----
log "phase 2b: init-equality check (stage-3 config, zero-init branch, eval only) -> logs/fg3_champ3g_init.log"
t0=$(date +%s)
ds3 --config configs/finegym_t3_champ_stage3_vsa.yaml --output fg3_champ3g_init \
  --init_from $CK2 --freeze_except video_subtree_attn --feature_cache $CACHE \
  --eval_only > logs/fg3_champ3g_init.log 2>&1
log "phase 2b rc=$? in $((($(date +%s) - t0) / 60)) min"
grep -h "\[ZSL video\]\|\[ZSL window\]" logs/fg3_champ3g_init.log | tee -a "$LOG"

# ---- 3. train the video branch ----
log "phase 3: training stage 3 (frozen except video_subtree_attn), 3-GPU -> logs/fg3_champ3g_train.log"
t0=$(date +%s)
ds3 --config configs/finegym_t3_champ_stage3_vsa.yaml --output fg3_champ3g_vsa \
  --init_from $CK2 --freeze_except video_subtree_attn --feature_cache $CACHE \
  --select_metric video_held_mAP -c 1 > logs/fg3_champ3g_train.log 2>&1
rc=$?
log "phase 3 rc=$rc in $((($(date +%s) - t0) / 60)) min"
grep -h "\[select\]\|Best model saved\|\[Train\]: Epoch .* finished" logs/fg3_champ3g_train.log | tee -a "$LOG"
log "=== stage 3 chain done (rc=$rc) ==="
