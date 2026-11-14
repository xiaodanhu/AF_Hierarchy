#!/bin/bash
# 2026-11-08 11:20 — GPU 3 chain order changed fd (tree, re-grounds the 467
# OOM windows at 336 px, then trains) -> fd_oracle -> thumos (grounding already complete: prior cuts).
cd /data3/xiaodan8/tifad || exit 1
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
echo 0 > queue/MAXJOBS_gpu3
log "launcher v2 up (pid $$): fd -> fd_oracle -> thumos"
for set in fd fd_oracle thumos; do
  log "chain $set start"
  CUDA_VISIBLE_DEVICES=3 bash scripts/vlm_hier_job.sh $set 0
  log "chain $set exit code $?"
done
rm -f queue/MAXJOBS_gpu3
log "all chains done; GPU 3 returned to the dispatcher"
