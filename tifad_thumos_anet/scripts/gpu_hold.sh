# helpers for dispatcher jobs that need more GPU memory than the
# dispatcher's 6 GB launch gate guarantees. Sourced by a6_stage3_job.sh and
# twolevel_split_v2.sh. The dispatcher (run_dispatcher.sh) re-reads
# queue/MAXJOBS_gpu<g> each cycle, so a job can freeze further placements on
# its card (cap = current job count) until the memory it needs is free, then
# lift the freeze once it has claimed its memory.
#   hold_card <tag>      freeze placements on this card, register a marker
#   wait_free <MB> <min> block until the card has >= MB free (at most <min> minutes)
#   release_card <tag>   drop the marker; remove the cap file when no marker remains
# Manual cleanup after a SIGKILL: rm -f queue/MAXJOBS_gpu* ; rm -rf queue/hold_gpu*.d
Q=/data3/xiaodan8/tifad/queue
G=${CUDA_VISIBLE_DEVICES:-0}
hold_card() {
  mkdir -p "$Q/hold_gpu$G.d"; touch "$Q/hold_gpu$G.d/$1"
  local n; n=$(pgrep -fc "[d]ispjob_gpu$G" || true); [ "${n:-0}" -lt 1 ] && n=1
  echo "$n" > "$Q/MAXJOBS_gpu$G"
  echo "[$(date '+%F %T')] $1: holding GPU$G at $n jobs" >> "$Q/hold.log"
}
wait_free() {
  local _i f
  for _i in $(seq 1 "$2"); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G")
    [ "${f:-0}" -ge "$1" ] && return 0
    sleep 60
  done
  return 1
}
release_card() {
  rm -f "$Q/hold_gpu$G.d/$1"
  if [ -z "$(ls -A "$Q/hold_gpu$G.d" 2>/dev/null)" ]; then rm -f "$Q/MAXJOBS_gpu$G"; rmdir "$Q/hold_gpu$G.d" 2>/dev/null; fi
  echo "[$(date '+%F %T')] $1: released GPU$G" >> "$Q/hold.log"
}
