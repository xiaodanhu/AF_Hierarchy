#!/usr/bin/env python
"""FineGym (element level, gym99) human-verification kit, companion to
make_verify_clips.py (THUMOS14) and make_verify_clips_fd.py (FineDiving).

For each chosen element instance of the test split: a slowed clip with the LLM phase sentence
overlaid during its equal chunk (K = 10, what the model aligns sentence k to), a 10-frame contact
sheet, and a row block in VERIFY_SHEET.md. Instances: one per held-out class (the 20 zero-shot
classes of the t3 champion), median duration, plus --n_seen seen classes. With --pred, the
champion's detections overlapping the instance are shown too (ground truth vs prediction)."""
import argparse, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_verify_clips import wrap, make_video, make_sheet
from pred_overlay import load_pred, top_preds, fmt_pred

A6 = '/data3/xiaodan8/actionformer6'
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))


def find_video(root, yid):
    for c in (f'{root}/video/gym_{yid}.mp4', f'{root}/video_new/{yid}.mp4', f'{root}/video_new/{yid}.mkv', f'{root}/video/gym_{yid}.mkv'):
        if os.path.exists(c): return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='/data3/xiaodan8/FineGym/annotation/Dec16/gym99_val_label.txt')
    ap.add_argument('--root', default='/data3/xiaodan8/FineGym')
    ap.add_argument('--trees', default=f'{A6}/configs/finegym_qwen_ensemble.json')
    ap.add_argument('--verbalizer', default=f'{A6}/configs/finegym_zsl_verbalizer.json')
    ap.add_argument('--attr', default=f'{A6}/configs/finegym_attribute_table_v2.json')
    ap.add_argument('--out', default=f'{A6}/tools/verify_clips_finegym')
    ap.add_argument('--n_seen', type=int, default=5, help='seen classes added after the held-out ones')
    ap.add_argument('--slow', type=float, default=4.0, help='elements last 1-3 s: quarter speed')
    ap.add_argument('--pred', default='', help='video-level predictions json of an eval-only run')
    ap.add_argument('--selection', default='', help='fg_selection.json (random draws, unusable annotations replaced, grade A first) instead of the median rule')
    a = ap.parse_args()
    verb = json.load(open(a.verbalizer)); acts = verb['actions']
    held = json.load(open(a.attr))['held_out_classes']
    trees = json.load(open(a.trees)); pred = load_pred(a.pred)
    key_of = lambda cid: f"({acts[cid]['activity']}) {acts[cid]['name']}"
    name_of = lambda lab: f"c{lab:02d} {acts[f'c{lab:02d}']['name']}{' (held-out)' if f'c{lab:02d}' in held else ''}"
    inst = {}
    for l in open(a.label):
        if not l.strip(): continue
        r = json.loads(l)
        for x in r['raw_value']:
            for q in x['actions']:
                s, e = map(sec, q['span'])
                if e - s >= 0.5: inst.setdefault(q['action_id'], []).append((e - s, r['video'], s, e))
    seen_pick = [c for c in sorted(inst) if c not in held and key_of(c) in trees]
    seen_pick = seen_pick[::max(1, len(seen_pick) // max(1, a.n_seen))][:a.n_seen]
    chosen = []
    for cid in list(held) + seen_pick:
        lst = inst.get(cid, [])
        if not lst or key_of(cid) not in trees: print('skip', cid); continue
        med = statistics.median(d for d, *_ in lst); lst.sort(key=lambda t: abs(t[0] - med))
        chosen.append((cid in held, cid, *lst[0]))
    meta = {}
    if a.selection:
        selj = json.load(open(a.selection)); chosen = [(d['held'], d['cid'], d['dur'], d['clip'], d['s'], d['e']) for d in selj]; meta = {(d['cid'], d['clip']): d for d in selj}
    os.makedirs(a.out, exist_ok=True); tmp = os.path.join(a.out, '_tmp'); os.makedirs(tmp, exist_ok=True)
    sheet = ['# Human verification of LLM element trees (FineGym gym99, test split)', '',
             *(['Instances: one random test instance per class (seed 0, duration within 0.5-2x the class median), the draw replaced when its annotation',
                'was unusable (span not covering the element, TV close-up, far camera); grade A = span covers the element, B = long tail after the landing. A first.', ''] if a.selection else []),
             'Chunks are the K = 10 equal parts of the annotated element, what the model aligns sentence k to.',
             'Clips run at 1/%g speed; the current sentence is overlaid during its chunk.' % a.slow,
             *(['Cyan line = the champion\'s best-scoring detection overlapping the element (class, span, score, IoU).'] if pred is not None else []), '',
             'Tick per sentence: **C** correct for this element, **P** happens but at a different time, **W** wrong / hallucinated.', '']
    for i, (is_held, cid, dur, clip, s, e) in enumerate(chosen, 1):
        key = key_of(cid); sents = trees[key]; root, phases = sents[0], sents[1:]; K = len(phases)
        video = find_video(a.root, clip.split('_E_')[0])
        if video is None: print('no video for', clip); continue
        tag = f'{i:02d}_{cid}_{clip}'
        print(f'[{i}/{len(chosen)}] {cid} {"(held-out)" if is_held else "(seen)"} {clip} {s:.1f}-{e:.1f}s', flush=True)
        preds = top_preds(pred, clip, s, e)
        xv = [(fmt_pred(p, name_of, j + 1), '#00FFFF' if j == 0 else '#9ADCFF', p['s'], p['e']) for j, p in enumerate(preds[:2])]
        xs = [(f"GT: {cid} {key}  [{s:.1f}-{e:.1f}s]", '#2E8B57')] + [(fmt_pred(p, name_of, j + 1), '#007C91') for j, p in enumerate(preds)]
        if pred is not None and not preds: xs.append(('PRED: no detection overlaps this element', '#B00020'))
        make_video(video, s, e, K, phases, key, root, os.path.join(a.out, tag + '.mp4'), tmp, slow=a.slow, extra=xv)
        make_sheet(video, s, e, K, phases, key, root, os.path.join(a.out, tag + '.png'), tmp, extra=xs if pred is not None else ())
        L = e - s
        m = meta.get((cid, clip), {})
        sheet += [f'## {i}. {key}  —  {cid} {"HELD-OUT" if is_held else "seen"}  —  `{clip}` {s:.1f}–{e:.1f} s ({L:.1f} s)' + (f"  —  grade {m['grade']}" + (f" ({m['note']})" if m.get('note') else '') if m else ''),
                  f'files: `{tag}.png`, `{tag}.mp4`', '', f'root: *{root}*', '',
                  *([('predictions overlapping the element: ' + ('; '.join(fmt_pred(p, name_of, j + 1) for j, p in enumerate(preds)) if preds else 'none')), ''] if pred is not None else []),
                  '| k | video time (s) | phase sentence | C / P / W | note |', '|---|---|---|---|---|']
        for k, t in enumerate(phases):
            sheet.append(f'| {k+1} | {s + k*L/K:.2f}–{s + (k+1)*L/K:.2f} | {t} |  |  |')
        sheet.append('')
    open(os.path.join(a.out, 'VERIFY_SHEET.md'), 'w').write('\n'.join(sheet) + '\n')
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print('done:', len(chosen), 'elements in', a.out)


if __name__ == '__main__':
    main()
