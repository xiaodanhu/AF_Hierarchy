#!/bin/bash
# 2026-11-09 00:45 — after the unclean reboot of 11-08 23:31: GPU 3 runs fd (training resumes from the
# epoch-2 adapter via vlm_hier_job.sh auto-resume) -> fd_oracle. THUMOS runs on GPU 2 (launch_vlm_gpu2_thumos.sh).
cd /data3/xiaodan8/tifad || exit 1
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
echo 0 > queue/MAXJOBS_gpu3
log "launcher v3 up (pid $$) after reboot: fd (resume) -> fd_oracle on GPU 3"
for set in fd fd_oracle; do
  log "chain $set start"
  CUDA_VISIBLE_DEVICES=3 bash scripts/vlm_hier_job.sh $set 0
  log "chain $set exit code $?"
done
rm -f queue/MAXJOBS_gpu3
log "GPU 3 chains done; GPU 3 returned to the dispatcher"
