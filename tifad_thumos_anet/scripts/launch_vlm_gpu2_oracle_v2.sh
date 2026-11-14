#!/bin/bash
# GPU 2 (11-09 v2): relaunched after the fd collapse fix (vlm_hier_job.sh v3: --prompt_list all + sanity gate).
# Once the concatenation controls (cpair_s*, cshort_s*) are done, claim GPU 2 (cap 0, drain) and run fd_oracle there.
cd /data3/xiaodan8/tifad || exit 1
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
log "GPU-2 oracle launcher v2 up (pid $$): waits for the concatenation controls, then fd_oracle on GPU 2"
while grep -qE '^(cpair|cshort)_s[0-9]' queue/jobs.txt 2>/dev/null || pgrep -f -- '--output c(pair|short) |thumos14_i3d_c(pair|short)_' >/dev/null; do sleep 300; done
log "concatenation controls done; claiming GPU 2 (cap 0, draining)"
echo 0 > queue/MAXJOBS_gpu2
while [ "$(pgrep -fc '[d]ispjob_gpu2')" -gt 0 ]; do sleep 120; done
log "GPU 2 drained; chain fd_oracle start on GPU 2"
CUDA_VISIBLE_DEVICES=2 bash scripts/vlm_hier_job.sh fd_oracle 0
log "chain fd_oracle exit code $?"
rm -f queue/MAXJOBS_gpu2
log "GPU 2 returned to the dispatcher"
