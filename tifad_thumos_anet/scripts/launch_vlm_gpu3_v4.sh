#!/bin/bash
# GPU 3: finish the running fd training (pid $1), then the fd chain with the zero-shot
# pass on the 150-window subsample; the oracle chain moves to GPU 2 (scripts/launch_vlm_gpu2_oracle.sh).
cd /data3/xiaodan8/tifad || exit 1
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
TP=${1:?fd train.py pid}
echo 0 > queue/MAXJOBS_gpu3
log "launcher v4 up (pid $$): waiting for fd training (pid $TP), then chain fd (zero-shot on the subsample) on GPU 3"
while kill -0 $TP 2>/dev/null; do sleep 120; done
log "fd training process exited; chain fd start on GPU 3"
CUDA_VISIBLE_DEVICES=3 bash scripts/vlm_hier_job.sh fd 0
log "chain fd exit code $?"
rm -f queue/MAXJOBS_gpu3
log "GPU 3 chains done; GPU 3 returned to the dispatcher"
