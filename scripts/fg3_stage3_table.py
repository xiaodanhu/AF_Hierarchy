#!/usr/bin/env python3
"""Per-epoch table for the FineGym champion stage-3 (video branch) run.

Parses logs/fg3_champ3_train.log (or the file given as argv[1]) for the
'[ZSL window]' / '[ZSL video]' seen / held-out lines that follow each
'Epoch: N , Test mAP' block and prints one row per epoch plus the stage-2
reference (logs/fg3_champ2.log, best checkpoint = epoch 5).
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'logs/fg3_champ3_train.log'
rows, cur = [], {}
for line in open(path, errors='replace'):
    m = re.search(r'\[ZSL (window|video)\]\s+(seen|held-out)-class mAP:\s+([0-9.]+)', line)
    if m:
        cur[f'{m.group(1)}_{m.group(2)}'] = float(m.group(3)) * 100
        continue
    m = re.search(r'Epoch:\s+(\d+) , Test mAP:\s+([0-9.]+)', line)
    if m:
        cur['epoch'] = int(m.group(1)) + 1
        cur['video_all'] = float(m.group(2)) * 100
        rows.append(cur)
        cur = {}
    m = re.search(r'\[Train\]: Epoch (\d+) finished with lr=([0-9.]+) Total Time: ([0-9.]+) mins', line)
    if m:
        cur['lr'] = float(m.group(2))
        cur['train_min'] = float(m.group(3))

print(f"{'epoch':>5} {'lr':>9} {'train_min':>9} | {'video held(20)':>14} {'video seen(79)':>14} {'video all':>9} | "
      f"{'window held':>11} {'window seen':>11}")
best = None
for r in rows:
    print(f"{r.get('epoch', -1):>5} {r.get('lr', float('nan')):>9.2e} {r.get('train_min', float('nan')):>9.1f} | "
          f"{r.get('video_held-out', float('nan')):>14.2f} {r.get('video_seen', float('nan')):>14.2f} "
          f"{r.get('video_all', float('nan')):>9.2f} | {r.get('window_held-out', float('nan')):>11.2f} "
          f"{r.get('window_seen', float('nan')):>11.2f}")
    if 'video_held-out' in r and (best is None or r['video_held-out'] > best['video_held-out']):
        best = r
if best:
    print(f"best by held-out video mAP: epoch {best['epoch']}  held {best['video_held-out']:.2f}  "
          f"seen {best['video_seen']:.2f}  (stage-2 reference: held 25.20  seen 58.50, epoch 5)")
