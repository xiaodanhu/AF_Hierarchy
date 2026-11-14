#!/bin/bash
# FineGym champion STAGE 3: video branch of the action--sub-action attention
# (libs/modeling/vsa.py) trained on the FROZEN stage-2 champion with cached
# CLIP features, on ONE GPU. Phases (each a separate process, launched
# back-to-back so the GPU is not lost in between):
#   1. cache   : frozen CLIP-encoder output of every train/test window
#                (scripts/cache_fg3_clip_feats.py, bitwise-verified)
#   2. inert   : stage-2 checkpoint, flag OFF, cached features, eval only
#                -> must reproduce logs/fg3_champ2.log (held-out 0.2520 /
#                seen 0.5850 video, 0.2450 / 0.5769 window)
#   2b. init   : stage-3 config (flag ON, zero-init) eval only -> same numbers
#   3. train   : stage 3, everything frozen except video_subtree_attn,
#                15 epochs, eval every epoch, best by held-out video mAP
# GPU policy: GPUs 0-3 allowed; wait until one has >= 12 GB free (poll
# nvidia-smi), then use that single GPU. The tifad dispatcher is never
# touched (no PAUSE, no kills); each phase claims its memory up-front
# (FG3_RESERVE_GB) so the dispatcher's own 6 GB launch gate protects it.
cd /data3/xiaodan8/actionformer6
PY=/data/xiaodan8/anaconda3/envs/parsing2/bin/python
CACHE=/data3/xiaodan8/FineGym/cache_fg3_champ2
CK2=ckpt/finegym_t3_champ_stage2_fg3_champ2/vit_best_model
LOG=logs/fg3_champ3_vsa.log
MIN_FREE=${MIN_FREE:-12000}
export FG3_RESERVE_GB=${FG3_RESERVE_GB:-8}
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

pick_gpu() {   # blocks until some GPU has >= MIN_FREE MB free; echoes its index
  while :; do
    for g in 0 1 2 3; do
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      if [ "$f" -ge "$MIN_FREE" ]; then echo $g; return; fi
    done
    sleep 3
  done
}

# single-process DeepSpeed launch (no launcher => the memory claim happens
# right after `import torch`, ~5 s after exec)
ds1() {  # ds1 <gpu> <args...>
  local g=$1; shift
  CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$((29500 + RANDOM % 400)) $PY train_shard.py --local_rank 0 "$@"
}

log "=== stage 3 (video branch) chain starting; waiting for a GPU with >= ${MIN_FREE} MB free ==="
G=$(pick_gpu)
log "GPU $G selected (free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G) MB)"

# ---- 1. cache ----
if [ ! -f "$CACHE/meta.json" ]; then
  log "phase 1: caching frozen CLIP features -> $CACHE (log logs/fg3_champ3_cache.log)"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$G $PY scripts/cache_fg3_clip_feats.py \
    --config configs/finegym_t3_champ_stage3_vsa.yaml --ckpt $CK2 --out $CACHE \
    --verify 8 --workers 24 --batch 8 > logs/fg3_champ3_cache.log 2>&1
  rc=$?; log "phase 1 rc=$rc in $((($(date +%s) - t0) / 60)) min"
  grep -h "\[verify\|done:" logs/fg3_champ3_cache.log | tee -a "$LOG"
  [ $rc -eq 0 ] || { log "caching failed; abort"; exit 1; }
else
  log "phase 1: cache present ($CACHE/meta.json), skipping"
fi

# ---- 2. byte-inert check: flag OFF, stage-2 ckpt, cached features ----
log "phase 2: byte-inert check (stage-2 config, flag off, cached features, eval only) -> logs/fg3_champ3_inert.log"
t0=$(date +%s)
ds1 $G --config configs/finegym_t3_champ_stage2.yaml --output fg3_champ3_inert \
  --init_from $CK2 --freeze_except text_pathway.phase_attn --feature_cache $CACHE \
  --eval_only > logs/fg3_champ3_inert.log 2>&1   # (freeze only keeps the unused optimizer small)
log "phase 2 rc=$? in $((($(date +%s) - t0) / 60)) min"
grep -h "\[ZSL video\]\|\[ZSL window\]\|Video-level action mAP\|Window-level action mAP" logs/fg3_champ3_inert.log | tee -a "$LOG"

# ---- 2b. init-equality: flag ON (zero-init), eval only ----
log "phase 2b: init-equality check (stage-3 config, zero-init branch, eval only) -> logs/fg3_champ3_init.log"
t0=$(date +%s)
ds1 $G --config configs/finegym_t3_champ_stage3_vsa.yaml --output fg3_champ3_init \
  --init_from $CK2 --freeze_except video_subtree_attn --feature_cache $CACHE \
  --eval_only > logs/fg3_champ3_init.log 2>&1
log "phase 2b rc=$? in $((($(date +%s) - t0) / 60)) min"
grep -h "\[ZSL video\]\|\[ZSL window\]" logs/fg3_champ3_init.log | tee -a "$LOG"

# ---- 3. train the video branch ----
log "phase 3: training stage 3 (frozen except video_subtree_attn) -> logs/fg3_champ3_train.log"
t0=$(date +%s)
ds1 $G --config configs/finegym_t3_champ_stage3_vsa.yaml --output fg3_champ3_vsa \
  --init_from $CK2 --freeze_except video_subtree_attn --feature_cache $CACHE \
  --select_metric video_held_mAP -c 1 > logs/fg3_champ3_train.log 2>&1
rc=$?
log "phase 3 rc=$rc in $((($(date +%s) - t0) / 60)) min"
grep -h "\[select\]\|Best model saved\|\[Train\]: Epoch .* finished" logs/fg3_champ3_train.log | tee -a "$LOG"
log "=== stage 3 chain done (rc=$rc) ==="
