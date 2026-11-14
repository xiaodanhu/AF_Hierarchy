#!/bin/bash
# TWO-LEVEL SUB-ACTION PROGRAM on THUMOS14 / ActivityNet (2026-09-25).
# Queues one dispatcher job per (cell, split) running scripts/twolevel_split.sh
# (stage 1 -> stage 2 -> fused eval -> nested-VSA stage 3 -> fused eval).
# Usage: bash scripts/run_twolevel.sh queue t75 t50 [a75 a50]   (default: all four)
#        bash scripts/run_twolevel.sh table
cd /data3/xiaodan8/tifad
Q() { flock /tmp/dispatch.lock tee -a queue/jobs.txt > /dev/null; }
case "$1" in
  queue) shift; cells="${*:-t75 t50 a75 a50}"
    for c in $cells; do for n in 0 1 2 3 4 5 6 7 8 9; do
      printf 'tl_%s_s%d\tbash scripts/twolevel_split.sh %s %d > logs/tl_%s_s%d.log 2>&1\n' $c $n $c $n $c $n
    done; done | Q
    echo "queued: $(wc -l < queue/jobs.txt) jobs in queue/jobs.txt" ;;
  table)
    /data/xiaodan8/anaconda3/envs/parsing2/bin/python - <<'EOF'
import re, glob, os
ref = {'t75': ('THUMOS14 75/25', 51.17, 51.00, 46.8), 't50': ('THUMOS14 50/50', 45.40, 45.66, 41.2),
       'a75': ('ActivityNet 75/25', 35.38, 35.51, 34.7), 'a50': ('ActivityNet 50/50', 33.11, 33.25, 32.0)}
def m(path):
    try:
        t = open(path).read(); v = re.findall(r'Ave\w+ mAP: ([\d.]+)', t)
        return float(v[-1]) if v else None
    except FileNotFoundError:
        return None
print('two-level sub-actions (K=10 phases x K\'=3 sub-phases), fused evals; one-level champion in [] for reference')
for c, (name, ch2, ch3, tifad) in ref.items():
    e2 = [m(f'logs/tl2e_{c}_s{n}.log') for n in range(10)]; e3 = [m(f'logs/tl3e_{c}_s{n}.log') for n in range(10)]
    f = lambda xs: (sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs)), sum(x is not None for x in xs))
    a2, n2 = f(e2); a3, n3 = f(e3)
    print(f'{name:18s} text-only two-level {a2:6.2f} [{n2:2d}/10]  (one-level {ch2})   +nested video {a3:6.2f} [{n3:2d}/10]  (one-level {ch3})   Ti-FAD {tifad}')
    print('   per split text:', ['%.2f' % x if x is not None else '  -  ' for x in e2])
    print('   per split full:', ['%.2f' % x if x is not None else '  -  ' for x in e3])
EOF
    ;;
  *) echo "usage: $0 queue [cells] | table"; exit 1 ;;
esac
