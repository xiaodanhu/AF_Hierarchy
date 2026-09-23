#!/bin/bash
# Ti-FAD + ANCESTOR SUPERVISION baselines on FineGym (2026-09-23, reviewer item 3:
# "Ti-FAD has no ancestor loss and cannot use the coarse annotations"). Under the
# paper's coarse-supervision protocol (held_out_mode keep) the baseline gets the
# same ancestor loss as our model. Sequential on all four GPUs (mb12 x 4 = 48):
#   1. l_d=2  configs/finegym_t2_tifad_as.yaml  -> ckpt/finegym_t2_tifad_as_fgas_t2
#   2. l_d=3  configs/finegym_t3_tifad_as.yaml  -> ckpt/finegym_t3_tifad_as_fgas_t3
# Markers ckpt/fgas_<lvl>_done.marker, logs logs/fgas_<lvl>.log, chain log
# logs/fgas_chain.log; retries on nonzero exit; holds the tifad dispatcher PAUSE.
# Results: grep -h "\[select\]\|Best model saved" logs/fgas_t2.log logs/fgas_t3.log
cd /data3/xiaodan8/actionformer6
PAUSE=/data3/xiaodan8/tifad/queue/PAUSE
LOG=logs/fgas_chain.log
MIN_FREE=${MIN_FREE:-13000}
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
wait_gpus() { while :; do ok=1; for g in 0 1 2 3; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); [ "$f" -ge "$MIN_FREE" ] || ok=0; done; [ "$ok" -eq 1 ] && return; sleep 120; done; }
touch "$PAUSE"
log "=== Ti-FAD + AS chain starting (PAUSE held) ==="
for lvl in t2 t3; do
  until [ -f "ckpt/fgas_${lvl}_done.marker" ]; do
    wait_gpus
    log "=== fgas_${lvl} starting on GPUs 0-3 ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py \
      --config configs/finegym_${lvl}_tifad_as.yaml --output fgas_${lvl} > logs/fgas_${lvl}.log 2>&1
    rc=$?; log "=== fgas_${lvl} rc=$rc ==="
    grep -h "\[ZSL video\]" logs/fgas_${lvl}.log | tail -2 | tee -a "$LOG"
    grep -h "\[select\]\|Best model saved" logs/fgas_${lvl}.log | tail -2 | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then touch "ckpt/fgas_${lvl}_done.marker"; else sleep 600; fi
  done
done
rm -f "$PAUSE"
log "=== Ti-FAD + AS chain COMPLETE ==="
