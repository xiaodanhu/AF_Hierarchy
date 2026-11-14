#!/bin/bash
# GPU 3 (11-09 v5): the fd chain collapsed (training prompts <= 8 classes vs 52 in the test prompt; see the
# vlm_hier_job.sh v3 header). Re-run as fd_full on GPU 3 (cap 0, card drained), then return the card to the dispatcher.
cd /data3/xiaodan8/tifad || exit 1
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
echo 0 > queue/MAXJOBS_gpu3
while [ "$(pgrep -fc '[d]ispjob_gpu3')" -gt 0 ]; do sleep 60; done
log "GPU 3 drained; chain fd_full start on GPU 3 (launcher pid $$)"
CUDA_VISIBLE_DEVICES=3 bash scripts/vlm_hier_job.sh fd_full 0
log "chain fd_full exit code $?"
rm -f queue/MAXJOBS_gpu3
log "GPU 3 returned to the dispatcher"
