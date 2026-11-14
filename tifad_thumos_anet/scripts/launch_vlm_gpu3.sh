#!/bin/bash
# Launch the VLM fine-tune on GPU 3 once the K chains finish.
# Waits until the 30 K-sweep chains (k5m/k6m/k7m, 10 splits each) have their final evals,
# then hands GPU 3 to the C3 chains: queue/MAXJOBS_gpu3=0 stops the dispatcher from placing
# anything new on card 3; each chain wrapper waits for >= 11 GB free before touching the GPU.
# Order: fd (tree) -> thumos (split 0) -> fd_oracle. Logs: logs/vlm_hier/*_chain.log, launcher.log.
cd /data3/xiaodan8/tifad || exit 1
mkdir -p logs/vlm_hier
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
log "launcher up (pid $$), waiting for the K chains"
while true; do
  n=$(ls logs/k[567]m3e_s*.log 2>/dev/null | xargs -r grep -l "Ave.*mAP" 2>/dev/null | wc -l)
  [ "$n" -ge 30 ] && break
  sleep 300
done
log "K chains complete ($n/30 final evals); closing GPU 3 to the dispatcher"
echo 0 > queue/MAXJOBS_gpu3
for set in fd thumos fd_oracle; do
  log "chain $set start"
  CUDA_VISIBLE_DEVICES=3 bash scripts/vlm_hier_job.sh $set 0
  log "chain $set exit code $?"
done
rm -f queue/MAXJOBS_gpu3
log "all chains done; GPU 3 returned to the dispatcher"
