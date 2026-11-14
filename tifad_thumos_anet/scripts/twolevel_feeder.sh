#!/bin/bash
# Re-queues two-level jobs that died (e.g. CUDA OOM when two nested stage-3
# jobs share a GPU): a job whose chain log exists, does not end in DONE, and
# has no running twolevel_split.sh process is appended to queue/jobs.txt
# again (the chain skips completed stages). Checks every 10 min; exits when
# every (cell, split) of the given cells is DONE.
# Usage: nohup bash scripts/twolevel_feeder.sh t75 t50 a75 a50 > logs/twolevel_feeder.log 2>&1 &
cd /data3/xiaodan8/tifad
cells="${*:-t75 t50 a75 a50}"
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
while :; do
  todo=0
  for c in $cells; do for n in 0 1 2 3 4 5 6 7 8 9; do
    f=logs/tl_${c}_s$n.log
    grep -q "DONE" "$f" 2>/dev/null && continue
    todo=$((todo + 1))
    [ -f "$f" ] || continue                                   # not started yet (still queued)
    pgrep -f "twolevel_split.sh $c $n\$" > /dev/null && continue   # running
    grep -q "	bash scripts/twolevel_split.sh $c $n " queue/jobs.txt && continue   # already re-queued
    log "requeue tl_${c}_s$n (last line: $(tail -1 "$f" | cut -c1-60); error: $(grep -h -m1 'Error\|error' logs/tl[123]_${c}_s$n.log logs/tl[23]e_${c}_s$n.log 2>/dev/null | tail -1 | cut -c1-80))"
    printf 'tl_%s_s%d\tbash scripts/twolevel_split.sh %s %d >> logs/tl_%s_s%d.log 2>&1\n' $c $n $c $n $c $n | flock /tmp/dispatch.lock tee -a queue/jobs.txt > /dev/null
  done; done
  [ "$todo" -eq 0 ] && { log "all done for cells: $cells"; exit 0; }
  sleep 600
done
