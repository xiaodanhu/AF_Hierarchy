#!/bin/bash
# GPU dispatcher daemon keep GPUs 0/1 fully
# stacked. Pops jobs from queue/jobs.txt (TAB-separated: TAG<TAB>CMD, where
# CMD may contain {GPU}); launches when a GPU has >=4500MB free and <4 of
# our jobs; coexists with legacy gated runners via the memory check.
# GPU2 allowed since 2026-09-11; GPU3 allowed since 2026-09-18 . Writes queue/dispatcher.log.
cd /data3/xiaodan8/tifad
QF=queue/jobs.txt; LK=/tmp/dispatch.lock
touch "$QF"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> queue/dispatcher.log; }
log "dispatcher up (pid $$)"
while true; do
  # queue/PAUSE stops new placements (used by the FG3 FineGym chain, which
  # needs both GPUs to itself); running jobs are unaffected.
  if [ -f queue/PAUSE ]; then sleep 60; continue; fi
  for g in 0; do   # 2026-09-19: GPUs 1-3 lent to FineGym stage-3 3-GPU DeepSpeed run; restore 0 1 2 3 when it finishes
    # fill each GPU up to the job cap in one pass (short eval jobs would
    # otherwise leave GPUs idle between 60 s cycles); 20 s between
    # placements lets a job claim its memory before the next check
    for _slot in 1 2 3 4; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      njobs=$(pgrep -fc "[d]ispjob_gpu$g" || true)
      [ "$free" -ge 6000 ] && [ "$njobs" -lt 4 ] || break
      job=$(flock $LK sh -c "head -1 $QF && sed -i '1d' $QF")
      [ -z "$job" ] && break
      tag=${job%%$'\t'*}; cmd=${job#*$'\t'}
      cmd=${cmd//\{GPU\}/$g}
      log "launch [$tag] on GPU$g (free=${free}MB)"
      bash -c "DISPJOB=dispjob_gpu$g; exec -a dispjob_gpu$g bash -c 'CUDA_VISIBLE_DEVICES=$g $cmd'" > logs/disp_$tag.log 2>&1 &
      sleep 5
    done
  done
  sleep 10
done
