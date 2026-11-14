#!/bin/bash
# One-off handoff (2026-09-12): the FG3 base run (deepspeed pid $1) was
# launched by an older copy of run_fg3_chain.sh whose bash process was stopped
# after the script was edited (bash reads scripts lazily; editing a running
# script is unsafe). Wait for that base run to finish, mark it done if its
# log ends with "All done!", then start the current chain, which continues
# with tifad -> champ -> champ2.
cd /data3/xiaodan8/actionformer6 || exit 1
pid=$1
while kill -0 "$pid" 2>/dev/null; do sleep 120; done
if grep -q "All done!" logs/fg3_base.log; then
  touch ckpt/fg3_base_done.marker
  echo "=== FG3 base finished (handoff) $(date) ===" >> logs/fg3_chain.log
else
  echo "=== FG3 base did NOT finish cleanly (handoff); chain will retry it $(date) ===" >> logs/fg3_chain.log
fi
rm -f /data3/xiaodan8/tifad/queue/PAUSE
exec bash scripts/run_fg3_chain.sh
