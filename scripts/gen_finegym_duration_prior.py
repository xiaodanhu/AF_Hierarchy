#!/usr/bin/env python3
"""Generate the FineGym Table-3 duration prior (champion component DP).

Per SEEN class: log-normal (mu, sigma) over instance durations (seconds)
from the Dec16 gym99 TRAIN labels (new_value spans).
Per HELD-OUT class (attribute table `held_out_classes`): Jaccard-weighted
mixture over SEEN classes using the 48-dim binary attribute rows, moment-
matched in log space to a single log-normal:
    mu_h    = sum_s w_s mu_s
    sigma_h = sqrt( sum_s w_s (sigma_s^2 + mu_s^2) - mu_h^2 )
Held-out train durations are NEVER used (leakage-aware protocol), even
though the transductive dataset keeps those segments.

Usage (from the repo root):
    python scripts/gen_finegym_duration_prior.py \
        --labels /data3/xiaodan8/FineGym/annotation/Dec16/gym99_train_label.txt \
        --attr-table configs/finegym_attribute_table_v2.json \
        --out configs/finegym_t3_duration_prior.json

Phrase level (l_d=2 of apparatus > phrase > action > phase; 14 classes):
instances = contiguous runs of same-phrase actions inside each activity
(same rule as FineGymSlideDataset._derive_hierarchy_gt), attributes from
the phrase attribute table (OR over member actions):
    python scripts/gen_finegym_duration_prior.py --level phrase \
        --labels /data3/xiaodan8/FineGym/annotation/Dec16/gym99_train_label.txt \
        --attr-table configs/finegym_phrase_attribute_table.json \
        --out configs/finegym_t2_duration_prior.json
"""
import argparse
import json
import math
from collections import defaultdict

SIGMA_FLOOR = 0.10   # log-space std floor (numerical safety)


def _span(a):
    s = float(a['span'][0].strip('<>').split()[0])
    e = float(a['span'][1].strip('<>').split()[0])
    return s, e


def parse_spans(labels_path):
    durs = defaultdict(list)
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for act in rec.get('new_value', []):
                for a in act.get('actions', []):
                    cid = int(a['action_id'][1:])
                    s = float(a['span'][0].strip('<>').split()[0])
                    e = float(a['span'][1].strip('<>').split()[0])
                    if e > s:
                        durs[cid].append(e - s)
    return durs


def parse_phrase_spans(labels_path, verbalizer_path):
    """Phrase-instance durations: within each activity, sort the actions by
    start and merge contiguous runs of the same phrase (verbalizer map) into
    one instance [run_start, max run_end] — the _derive_hierarchy_gt rule."""
    with open(verbalizer_path) as f:
        verb = json.load(f)
    a2p = {aid: int(info['phrase'][1:]) for aid, info in verb['actions'].items()}
    durs = defaultdict(list)
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for act in rec.get('new_value', []):
                actions = []
                for a in act.get('actions', []):
                    pid = a2p.get(a['action_id'])
                    if pid is None:
                        continue
                    s, e = _span(a)
                    actions.append((s, e, pid))
                actions.sort(key=lambda x: x[0])
                runs = []
                cur = None
                for s, e, pid in actions:
                    if cur is not None and cur[2] == pid:
                        cur[1] = max(cur[1], e)
                    else:
                        if cur is not None:
                            runs.append(cur)
                        cur = [s, e, pid]
                if cur is not None:
                    runs.append(cur)
                for s, e, pid in runs:
                    if e > s:
                        durs[pid].append(e - s)
    return durs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--attr-table', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--level', choices=['action', 'phrase'], default='action',
                    help="'action' (default, 99 gym99 classes) or 'phrase' "
                         "(14 phrase classes; needs the phrase attribute table)")
    ap.add_argument('--verbalizer', default='configs/finegym_zsl_verbalizer.json',
                    help='action->phrase map (phrase level only)')
    args = ap.parse_args()

    with open(args.attr_table) as f:
        table = json.load(f)
    held = sorted(int(a[1:]) for a in table['held_out_classes'])
    attrs = {int(a[1:]): [int(x) for x in e['attrs']]
             for a, e in table['actions'].items()}
    num_classes = len(attrs)
    seen = [c for c in range(num_classes) if c not in held]

    if args.level == 'phrase':
        durs = parse_phrase_spans(args.labels, args.verbalizer)
    else:
        durs = parse_spans(args.labels)

    # ---- seen classes: fit log-normal ----
    stats = {}
    sigmas = []
    for c in seen:
        d = durs.get(c, [])
        assert len(d) > 0, f"seen class {c} has no train instances"
        logs = [math.log(x) for x in d]
        mu = sum(logs) / len(logs)
        if len(logs) > 1:
            var = sum((x - mu) ** 2 for x in logs) / (len(logs) - 1)
        else:
            var = 0.0
        sigma = max(math.sqrt(var), SIGMA_FLOOR)
        stats[c] = {'mu': mu, 'sigma': sigma, 'n': len(d), 'kind': 'seen'}
        sigmas.append(sigma)

    # ---- held-out classes: Jaccard-weighted seen mixture ----
    for h in held:
        ah = attrs[h]
        ws = []
        for s in seen:
            a_s = attrs[s]
            inter = sum(1 for x, y in zip(ah, a_s) if x and y)
            union = sum(1 for x, y in zip(ah, a_s) if x or y)
            ws.append(inter / union if union else 0.0)
        tot = sum(ws)
        if tot <= 0:
            ws = [1.0 / len(seen)] * len(seen)
        else:
            ws = [w / tot for w in ws]
        mu_h = sum(w * stats[s]['mu'] for w, s in zip(ws, seen))
        m2 = sum(w * (stats[s]['sigma'] ** 2 + stats[s]['mu'] ** 2)
                 for w, s in zip(ws, seen))
        sigma_h = max(math.sqrt(max(m2 - mu_h ** 2, 0.0)), SIGMA_FLOOR)
        topw = sorted(zip(ws, seen), reverse=True)[:3]
        stats[h] = {'mu': mu_h, 'sigma': sigma_h, 'n': 0, 'kind': 'held_out',
                    'top_seen_weights': [[s, round(w, 4)] for w, s in topw]}

    out = {
        'meta': {
            'labels': args.labels,
            'attr_table': args.attr_table,
            'sigma_floor': SIGMA_FLOOR,
            'num_seen': len(seen),
            'num_held_out': len(held),
            'held_out': held,
            'note': ('log-normal duration prior in SECONDS; held-out = '
                     'Jaccard-weighted moment-matched seen mixture '
                     '(no held-out train durations used)'),
        },
        'classes': {str(c): stats[c] for c in range(num_classes)},
    }
    # Level tag only for the phrase prior so the action-level (t3) output
    # stays byte-identical to the previously generated file.
    if args.level == 'phrase':
        out['meta']['level'] = 'phrase'
        out['meta']['verbalizer'] = args.verbalizer
        out['meta']['instance_rule'] = ('contiguous runs of same-phrase actions inside '
                                        'an activity (FineGymSlideDataset._derive_hierarchy_gt)')
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    mus = [stats[c]['mu'] for c in range(num_classes)]
    print(f"wrote {args.out}: {num_classes} classes "
          f"(seen {len(seen)}, held-out {len(held)}); "
          f"exp(mu) range {math.exp(min(mus)):.2f}-{math.exp(max(mus)):.2f} s")


if __name__ == '__main__':
    main()
