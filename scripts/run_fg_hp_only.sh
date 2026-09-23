#!/bin/bash
# HP-ONLY intermediate FineGym baselines (2026-09-23, reviewer Q5): Ti-FAD +
# hierarchical prompt with MEAN aggregation (no ASA, no AS, no DP), keep protocol.
# Queued after scripts/run_fg_tifad_as.sh (waits for ckpt/fgas_t3_done.marker),
# then l_d=2 (configs/finegym_t2_tifad_hp.yaml) and l_d=3 (finegym_t3_tifad_hp.yaml)
# sequentially on all four GPUs (mb12 x 4 = 48). Each config is smoke-tested for
# two batches first (FG_SMOKE_ITERS=2); a failing smoke aborts that level.
# Markers ckpt/fghp_<lvl>_done.marker, logs logs/fghp_<lvl>.log, chain log logs/fghp_chain.log.
cd /data3/xiaodan8/actionformer6
PAUSE=/data3/xiaodan8/tifad/queue/PAUSE
LOG=logs/fghp_chain.log
S=/tmp/claude-1002/-data3-xiaodan8-actionformer6/15f23ac7-2198-4c7c-b3a0-fcf5333478bb/scratchpad/smoke_fg2
MIN_FREE=${MIN_FREE:-13000}
mkdir -p "$S"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
wait_gpus() { while :; do ok=1; for g in 0 1 2 3; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); [ "$f" -ge "$MIN_FREE" ] || ok=0; done; [ "$ok" -eq 1 ] && return; sleep 120; done; }
log "=== HP-only chain queued; waiting for the Ti-FAD+AS chain (ckpt/fgas_t3_done.marker) ==="
until [ -f ckpt/fgas_t3_done.marker ]; do sleep 300; done
touch "$PAUSE"
for lvl in t2 t3; do
  cfg=configs/finegym_${lvl}_tifad_hp.yaml
  if [ ! -f "ckpt/fghp_${lvl}_smoke_ok.marker" ]; then
    wait_gpus
    sed -e 's/^  epochs: [0-9]*,.*/  epochs: 1,/' "$cfg" | sed 's/^opt: {/opt: {\n  warmup_epochs: 1,/' > "$S/finegym_${lvl}_tifad_hp.yaml"
    log "=== smoke fghp_${lvl} ==="
    FG_SMOKE_ITERS=2 CUDA_VISIBLE_DEVICES=3 deepspeed --master_port=$((28000 + RANDOM % 900)) train_shard.py \
      --config "$S/finegym_${lvl}_tifad_hp.yaml" --output smoke_fghp_${lvl} > logs/smoke_fghp_${lvl}.log 2>&1
    rc=$?; rm -rf ckpt/*smoke_fghp_${lvl}*
    if [ "$rc" -ne 0 ]; then log "smoke fghp_${lvl} FAILED (rc=$rc); skipping this level"; continue; fi
    touch "ckpt/fghp_${lvl}_smoke_ok.marker"
  fi
  until [ -f "ckpt/fghp_${lvl}_done.marker" ]; do
    wait_gpus
    log "=== fghp_${lvl} starting on GPUs 0-3 ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py \
      --config "$cfg" --output fghp_${lvl} > logs/fghp_${lvl}.log 2>&1
    rc=$?; log "=== fghp_${lvl} rc=$rc ==="
    grep -h "\[ZSL video\]" logs/fghp_${lvl}.log | tail -2 | tee -a "$LOG"
    grep -h "\[select\]\|Best model saved" logs/fghp_${lvl}.log | tail -2 | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then touch "ckpt/fghp_${lvl}_done.marker"; else sleep 600; fi
  done
done
rm -f "$PAUSE"
log "=== HP-only chain COMPLETE ==="
