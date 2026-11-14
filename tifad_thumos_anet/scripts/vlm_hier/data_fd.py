#!/usr/bin/env python
"""FineDiving (split s1, 13/52 dive codes held out) windows for the hierarchical VLM (C3).

Mirrors the detector's loader (finegym_slide, use_raw_video, window_length 64 x stride 4
= 10.24 s @ 25 fps): ONE window per clip that STARTS at the clip's instance span
(= the dive start); train and test alike. Targets are nested: the dive (group, name,
span) with its annotated sub-actions. Protocol `keep`: held-out dives in training
windows keep their span and group; their class-name tokens are excluded from the
loss (mask_name, applied in train.py) and they carry no sub-action list — the
generative twin of the detector's masked classification loss with boundary +
group supervision kept.

Frames: pre-built store /data3/xiaodan8/FineDiving/OpenDataLab___FineDiving/RGB/<video>/<frame>.jpg
(native 25 fps, 1280x720); we take 4 fps = 40 frames per window, nearest stored frame.

  python scripts/vlm_hier/data_fd.py --out data/vlm_hier/fd_s1
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import r1

FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
A6 = '/data3/xiaodan8/actionformer6'
GROUP = {'FWD': 'forward', 'BCK': 'backward', 'REV': 'reverse', 'INW': 'inward', 'TWI': 'twisting', 'ARM': 'armstand'}
WIN = 256 / 25.0          # 10.24 s
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/vlm_hier/fd_s1')
    ap.add_argument('--fps', type=float, default=4.0)
    ap.add_argument('--attr', default=f'{A6}/configs/finediving/fd_dive_attribute_table_s1.json')
    ap.add_argument('--sub_source', default='tree', choices=['tree', 'annot'],
                    help="tree: children = the dive's ordered members from the definition tree, spans to be self-labelled by ground.py "
                         "(one annotated level, like our method); annot: annotated sub-action spans (oracle ceiling)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    verb = json.load(open(f'{A6}/configs/finediving/fd_dive_verbalizer.json'))['actions']
    _sub = json.load(open(f'{A6}/configs/finediving/fd_sub_verbalizer.json'))['actions']
    subverb = {k: v['name'] for k, v in _sub.items()}                       # verbalizer id -> name (dive members use this space)
    raw2name = {v['raw_id']: v['name'] for v in _sub.values()}            # label files use RAW ids (raw c28 = '0.5 somersault pike'); raw c15 (2 dives) has no entry -> skipped
    held = set(json.load(open(a.attr))['held_out_classes'])
    by_code = {v['code']: v for v in verb.values()}
    meta = {}
    for sp in ('train', 'test'):
        meta.update(json.load(open(f'{FD}/video_metadata_cache_{sp}.json')))
    store = {}
    def nearest(video, t):
        if video not in store:
            d = f'{FD}/RGB/{video}'
            store[video] = np.array(sorted(int(f[:-4]) for f in os.listdir(d))) if os.path.isdir(d) else np.array([])
        idx = store[video]
        if len(idx) == 0:
            return None, None
        want = int(round(t * meta[video]['fps']))
        j = int(idx[np.abs(idx - want).argmin()])
        return j, abs(j - want) / meta[video]['fps']
    nd = lambda video: max(len(str(meta[video]['total_frames'])), 6)
    stats = {}
    for sp in ('train', 'test'):
        rows, n_unknown, n_missing, off = [], 0, 0, []
        for line in open(f'{FD}/finediving_{sp}_label.txt'):
            if not line.strip():
                continue
            r = json.loads(line); clip = r['video']; video = clip.rsplit('_', 1)[0]
            if video not in meta:
                n_missing += 1; continue
            s0 = sec(r['span'][0]); fps_n = meta[video]['fps']
            n = int(WIN * a.fps)                                   # 40 frames
            frames, idxs = [], []
            for k in range(n):
                j, d = nearest(video, s0 + k / a.fps)
                if j is None:
                    break
                frames.append(f'{FD}/RGB/{video}/{j:0{nd(video)}d}.jpg'); off.append(d)
            if len(frames) < n:
                n_missing += 1; continue
            tgt = []
            for x in r['raw_value']:
                code = x['activity_id']; v = by_code[code]
                ds, de = sec(x['span'][0]) - s0, sec(x['span'][1]) - s0
                if de <= 0 or ds >= WIN:
                    continue
                if a.sub_source == 'annot':
                    subs = [{'action': raw2name[q['action_id']], 'start': r1(max(0.0, sec(q['span'][0]) - s0)), 'end': r1(min(WIN, sec(q['span'][1]) - s0))}
                            for q in x['actions'] if q['action_id'] in raw2name]
                    subs = [q for q in subs if q['end'] > q['start']]
                else:                                                  # definition tree: ordered members, no spans yet
                    subs = [{'action': subverb[m]} for m in v['members']]
                inst = {'group': GROUP[v['activity']], 'action': v['name'], 'start': r1(max(0.0, ds)), 'end': r1(min(WIN, de)), 'sub': subs, 'cls': v['id'], 'code': code}
                if sp == 'train' and v['id'] in held:               # keep protocol: span + group supervised, class NOT (train.py masks the
                    inst = {'group': inst['group'], 'action': v['name'], 'start': inst['start'], 'end': inst['end'],   # name tokens out of the loss,
                            'mask_name': True, 'cls': v['id'], 'code': code}                                          # like the detector's masked cls loss); no sub list
                    n_unknown += 1
                tgt.append(inst)
            tgt.sort(key=lambda t: t['start'])
            rows.append({'id': clip, 'video': clip, 'raw_video': video, 'win_start': s0, 'win_len': WIN, 'fps': a.fps, 'frames': frames,
                         'target': tgt, 'pos_classes': sorted({t['cls'] for t in tgt if t['action'] != 'unknown'})})
        with open(os.path.join(a.out, f'{sp}.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        stats[sp] = {'windows': len(rows), 'instances': sum(len(r['target']) for r in rows), 'held_out_masked_in_train': n_unknown,
                     'clips_without_frames': n_missing, 'frame_time_offset_s': {'mean': float(np.mean(off)), 'p99': float(np.percentile(off, 99))}}
        print(sp, stats[sp], flush=True)
    classes = [{'id': v['id'], 'code': v['code'], 'name': v['name'], 'group': GROUP[v['activity']], 'held_out': v['id'] in held,
                'members': [subverb[m] for m in v['members']]} for v in sorted(verb.values(), key=lambda v: v['id'])]
    sub_roots = {}
    ens = json.load(open(f'{A6}/configs/finediving/fd_sub_ensemble.json'))
    for k, sents in ens.items():
        sub_roots[re.sub(r'^\([A-Z]+\) ', '', k)] = sents[0]
    json.dump({'classes': classes, 'sub_actions': sorted(set(subverb.values())), 'sub_roots': sub_roots, 'groups': sorted(set(GROUP.values())),
               'held_out': sorted(held), 'win_len': WIN, 'fps': a.fps, 'sub_source': a.sub_source, 'stats': stats}, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print('->', a.out)


if __name__ == '__main__':
    main()
