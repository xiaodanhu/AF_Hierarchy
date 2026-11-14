#!/bin/bash
# 2026-11-08 12:45 relaunch after OOM at step 170 (now 280 px, 6 classes, 7k-token guard)
# 2026-11-08 11:45 run the THUMOS C3 chain on GPU 2 as soon as the three
# dispatcher jobs there finish (queue/MAXJOBS_gpu2=0 stops new placements). Grounding is already complete
# (prior cuts), so the chain goes train -> infer tuned -> infer zero-shot -> eval. Returns GPU 2 afterwards.
cd /data3/xiaodan8/tifad || exit 1
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
log "GPU-2 THUMOS launcher up (pid $$), waiting for dispatcher jobs on GPU 2 to drain"
while [ "$(pgrep -fc '[d]ispjob_gpu2')" -gt 0 ]; do sleep 120; done
log "GPU 2 drained; chain thumos start on GPU 2"
CUDA_VISIBLE_DEVICES=2 bash scripts/vlm_hier_job.sh thumos 0
log "chain thumos (GPU 2) exit code $?"
rm -f queue/MAXJOBS_gpu2
log "GPU 2 returned to the dispatcher"
