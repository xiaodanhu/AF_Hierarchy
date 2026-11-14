#!/bin/bash
# Re-run the FineDiving tree chain (training OOMed on 2026-11-08 04:11: frames were full-res 1280x720 -> ~23k
# visual tokens; --size now defaults to 336). Waits for the main launcher (pid in launcher.log) to finish its
# thumos + fd_oracle chains, keeps GPU 3 closed to the dispatcher, then resumes fd at the training stage
# (its grounding output train_nested.jsonl is kept).
cd /data3/xiaodan8/tifad || exit 1
log() { echo "[$(date '+%F %T')] $*" >> logs/vlm_hier/launcher.log; }
main=$(grep -oE 'launcher up \(pid [0-9]+\)' logs/vlm_hier/launcher.log | tail -1 | grep -oE '[0-9]+')
log "fd retry launcher up (pid $$), waiting for launcher $main"
while kill -0 "$main" 2>/dev/null; do sleep 300; done
echo 0 > queue/MAXJOBS_gpu3
log "chain fd (retry, 336 px) start"
CUDA_VISIBLE_DEVICES=3 bash scripts/vlm_hier_job.sh fd 0
log "chain fd (retry) exit code $?"
rm -f queue/MAXJOBS_gpu3
log "fd retry done; GPU 3 returned to the dispatcher"
