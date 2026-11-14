#!/usr/bin/env python
"""THUMOS14 75/25 windows for the hierarchical VLM (C3).

Mirrors the tifad protocol: training = the 200 'Validation' videos; annotations of
the split's unseen classes are dropped (they become background, exactly what the
detector sees); test = the 212 'Test' videos, prompt = the unseen classes only,
scored by the same evaluator as C1.

Frames: /data3/xiaodan8/thumos/frames_1fps_336/<vid>/%05d.jpg, frame k (1-based) = second k-1.

  python scripts/vlm_hier/data_thumos.py --n 0 --out data/vlm_hier/thumos_s0
writes train.jsonl (positive windows + 1:1 empty windows) and test.jsonl.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import readable, r1

ANN = '/data3/xiaodan8/thumos/annotations/thumos14.json'
FRAMES = '/data3/xiaodan8/thumos/frames_1fps_336'
SPLIT = 'splits/train_75_test_25/THUMOS14/{mode}/split_{n}.list'


def frames_of(vid):
    d = os.path.join(FRAMES, vid)
    if not os.path.isfile(os.path.join(d, '.done')):           # ffmpeg store still being written -> treat as missing
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.jpg'))


def windows(vid, nframes, win, stride, anns, classes, mode):
    """anns: [(label, s, e)] absolute seconds, already restricted to `classes`."""
    out = []
    starts = list(range(0, max(1, nframes - win // 2), stride))
    if nframes > win and (nframes - win) not in starts:
        starts.append(nframes - win)                     # cover the tail
    for w0 in starts:
        n = min(win, nframes - w0)
        if n < 8:
            continue
        tgt = []
        for lab, s, e in anns:
            ov = min(e, w0 + n) - max(s, w0)
            if ov <= 0 or ov < min(1.0, 0.5 * (e - s)):
                continue
            tgt.append({'action': readable(lab), 'start': r1(max(s, w0) - w0), 'end': r1(min(e, w0 + n) - w0), 'cls': lab})
        tgt.sort(key=lambda t: t['start'])
        out.append({'id': f'{vid}@{w0}', 'video': vid, 'win_start': float(w0), 'win_len': float(n), 'fps': 1.0,
                    'frame_idx': list(range(w0, w0 + n)), 'target': tgt, 'pos_classes': sorted({t['cls'] for t in tgt})})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--win', type=int, default=64)
    ap.add_argument('--train_stride', type=int, default=32)
    ap.add_argument('--test_stride', type=int, default=48)
    ap.add_argument('--neg_ratio', type=float, default=1.0, help='empty windows kept per positive window (train)')
    a = ap.parse_args()
    out = a.out or f'data/vlm_hier/thumos_s{a.n}'
    os.makedirs(out, exist_ok=True)
    db = json.load(open(ANN)); db = db.get('database', db)
    seen = [l.strip() for l in open(SPLIT.format(mode='train', n=a.n)) if l.strip()]
    unseen = [l.strip() for l in open(SPLIT.format(mode='test', n=a.n)) if l.strip()]
    rng = random.Random(a.n)
    stats = {}
    for mode, subset, stride, classes in (('train', 'Validation', a.train_stride, seen), ('test', 'Test', a.test_stride, unseen)):
        rows, missing = [], []
        for vid in sorted(v for v in db if db[v]['subset'] == subset):
            fr = frames_of(vid)
            if not fr:
                missing.append(vid); continue
            anns = [(x['label'], float(x['segment'][0]), float(x['segment'][1])) for x in db[vid]['annotations'] if x['label'] in classes]
            ws = windows(vid, len(fr), a.win, stride, anns, classes, mode)
            for w in ws:
                w['frames'] = [fr[i] for i in w['frame_idx']]; del w['frame_idx']
            if mode == 'test':
                for w in ws:
                    w['target'] = []; w['pos_classes'] = []      # never used; keeps the file honest
                rows += ws
            else:
                pos = [w for w in ws if w['target']]; neg = [w for w in ws if not w['target']]
                rng.shuffle(neg)
                rows += pos + neg[:int(round(a.neg_ratio * len(pos)))]
        rng.shuffle(rows) if mode == 'train' else None
        with open(os.path.join(out, f'{mode}.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        npos = sum(1 for r in rows if r['target']); ninst = sum(len(r['target']) for r in rows)
        stats[mode] = {'windows': len(rows), 'positive': npos, 'instances': ninst, 'videos_missing_frames': len(missing)}
        print(f'{mode}: {len(rows)} windows ({npos} with targets, {ninst} instances), {len(missing)} videos without frames', flush=True)
    json.dump({'split': a.n, 'seen': seen, 'unseen': unseen, 'stats': stats, 'args': vars(a)}, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('->', out)


if __name__ == '__main__':
    main()
