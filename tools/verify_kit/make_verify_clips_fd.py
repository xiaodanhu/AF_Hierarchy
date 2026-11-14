#!/usr/bin/env python
"""FineDiving human-verification kit, companion to make_verify_clips.py.

Same outputs as the THUMOS kit (half-speed clip with the current LLM phase
sentence overlaid during its equal chunk, 10-frame contact sheet, VERIFY_SHEET.md),
plus one thing THUMOS cannot offer: FineDiving annotates sub-action boundaries,
so the clip also shows the ANNOTATED sub-action at every moment. A rater can
therefore see directly whether phase sentence k, placed by the equal split,
describes what the annotation says is happening then.

Dives are chosen one per dive code, test split, median duration, held-out codes first.
"""
import argparse, json, os, re, subprocess, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_verify_clips import wrap, make_sheet, FONT, FONT_B
from pred_overlay import load_pred, top_preds, fmt_pred

FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))


def make_video_fd(video, s, e, K, sents, cls, root, subs, out_mp4, tmpdir, slow=2.0, width=960, extra=()):
    """subs: list of (name, start, end) annotated sub-actions, absolute seconds; extra: [(text, color, a0, b0)] prediction lines."""
    L = e - s
    filters = [f'scale={width}:-2', f'setpts={slow}*PTS']
    for i, (txt, col, a0, b0) in enumerate(extra):
        a, b = slow * max(0.0, a0 - s), slow * min(L, b0 - s)
        tp = os.path.join(tmpdir, f'x{i}.txt'); open(tp, 'w').write(txt)
        filters.append(f"drawtext=fontfile={FONT_B}:textfile={tp}:fontsize=22:fontcolor={col}:box=1:boxcolor=black@0.65:"
                       f"boxborderw=8:x=14:y={100 + 34 * i}:enable='gte(t,{a:.3f})*lt(t,{b:.3f})'")
    hp = os.path.join(tmpdir, 'hdr.txt'); open(hp, 'w').write(f'{cls}   |   {wrap(root, 90)}')
    filters.append(f"drawtext=fontfile={FONT_B}:textfile={hp}:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8:x=14:y=12")
    for k in range(K):                                   # LLM phase sentence, equal chunk k
        a, b = slow * k * L / K, slow * (k + 1) * L / K
        tp = os.path.join(tmpdir, f'c{k}.txt')
        open(tp, 'w').write(f'LLM phase {k+1}/{K}  [{s + k*L/K:.1f}-{s + (k+1)*L/K:.1f}s]\n' + wrap(sents[k], 62))
        filters.append(f"drawtext=fontfile={FONT}:textfile={tp}:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.65:boxborderw=10:x=14:y=h-th-14:enable='gte(t,{a:.3f})*lt(t,{b:.3f})'")
    for j, (name, a0, b0) in enumerate(subs):             # annotated sub-action, its true span
        a, b = slow * (a0 - s), slow * (b0 - s)
        tp = os.path.join(tmpdir, f'g{j}.txt')
        open(tp, 'w').write(f'ANNOTATED sub-action {j+1}/{len(subs)}: {name}  [{a0:.1f}-{b0:.1f}s]')
        filters.append(f"drawtext=fontfile={FONT_B}:textfile={tp}:fontsize=22:fontcolor=#7CFC00:box=1:boxcolor=black@0.65:boxborderw=8:x=14:y=58:enable='gte(t,{a:.3f})*lt(t,{b:.3f})'")
    cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f'{s:.3f}', '-to', f'{e:.3f}', '-i', video, '-vf', ','.join(filters),
           '-an', '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', out_mp4]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data3/xiaodan8/actionformer6/tools/verify_clips_finediving')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--trees', default='configs/finediving/fd_dive_ensemble.json', help='dive-level sentences (root + K phases)')
    ap.add_argument('--pred', default='', help='video-level predictions json of an eval-only run (A6_EVAL_SAVE_PRED=1)')
    ap.add_argument('--selection', default='', help='fd_selection.json (random dives from the well-annotated pool, well-aligned first) instead of the median rule')
    a = ap.parse_args()
    verb = json.load(open('configs/finediving/fd_dive_verbalizer.json'))
    subverb = {v['raw_id']: v['name'] for v in json.load(open('configs/finediving/fd_sub_verbalizer.json'))['actions'].values()}   # annotation ids are raw ids
    trees = json.load(open(a.trees))
    pred = load_pred(a.pred)
    codes = sorted(v['code'] for v in verb['actions'].values())          # detection label = index in sorted code order
    held = set(json.load(open('configs/finediving/fd_dive_attribute_table_s1.json'))['held_out_classes'])
    by_code = {v['code']: (k, f"({v['activity']}) {v['name']}") for k, v in verb['actions'].items()}
    held_codes = {verb['actions'][c]['code'] for c in held}
    name_of = lambda lab: f"{codes[lab]} {by_code[codes[lab]][1]}{' (held-out)' if codes[lab] in held_codes else ''}"
    dives = {}
    for l in open(f'{FD}/finediving_test_label.txt'):
        if not l.strip(): continue
        r = json.loads(l)
        for x in r['raw_value']:
            s, e = map(sec, x['span'])
            subs = [(subverb.get(q['action_id'], q['action_id']), *map(sec, q['span'])) for q in x['actions']]
            dives.setdefault(x['activity_id'], []).append((e - s, r['video'], s, e, subs))
    import statistics
    chosen = []
    for code, lst in dives.items():
        cid, key = by_code[code]
        med = statistics.median(d for d, *_ in lst)
        lst.sort(key=lambda t: abs(t[0] - med))
        chosen.append((cid in held, code, key, *lst[0]))
    chosen.sort(key=lambda t: (not t[0], t[1]))             # held-out codes first
    chosen = chosen[:a.n]
    meta = {}
    if a.selection:
        selj = json.load(open(a.selection))['selected'][:a.n]
        chosen = [(d['held'], d['code'], d['key'], d['dur'], d['video'], d['s'], d['e'], [tuple(x) for x in d['subs']]) for d in selj]
        meta = {d['video']: d for d in selj}
    os.makedirs(a.out, exist_ok=True); tmp = os.path.join(a.out, '_tmp'); os.makedirs(tmp, exist_ok=True)
    sheet = ['# Human verification of LLM dive trees (FineDiving, test split)', '',
             'Each clip shows TWO overlays: yellow = the LLM phase sentence assigned by the equal split (bottom);',
             'green = the ANNOTATED sub-action at that moment (top). Tick per phase sentence:',
             '**C** correct for the dive, **P** happens but at a different time, **W** wrong / hallucinated.', '',
             *(['Dives: one random test dive per code (seed 0) from the well-annotated pool (annotated sub-actions equal the dive definition and end with an entry,',
                'duration within 30% of the code median, equal chunks vs annotated sub-actions tIoU >= 0.5); codes with an empty pool are left out. Held-out codes first, then by alignment (tIoU).',
                'Cyan line = the champion\'s best-scoring detections overlapping the dive (class, span, score, IoU).', ''] if a.selection else [])]
    for i, (is_held, code, key, dur, vid, s, e, subs) in enumerate(chosen, 1):
        sents = trees[key]; root, phases = sents[0], sents[1:]; K = len(phases)
        video = f"{FD}/FineDiving/Released_FineDiving_Dataset/Untrimmed_Videos/{vid.rsplit('_', 1)[0]}.mp4"   # label id = video_name + '_' + clip_id (loader: extract_video_name)
        tag = f'{i:02d}_{code}_{vid}'
        print(f'[{i}/{len(chosen)}] {code} {"(held-out)" if is_held else "(seen)"} {vid} {s:.1f}-{e:.1f}s', flush=True)
        preds = top_preds(pred, vid, s, e)
        xv = [(fmt_pred(p, name_of, j + 1), '#00FFFF' if j == 0 else '#9ADCFF', p['s'], p['e']) for j, p in enumerate(preds[:2])]
        xs = [(f"GT: {code} {key}  [{s:.1f}-{e:.1f}s]", '#2E8B57')] + [(fmt_pred(p, name_of, j + 1), '#007C91') for j, p in enumerate(preds)]
        if pred is not None and not preds: xs.append(('PRED: no detection overlaps this dive', '#B00020'))
        make_video_fd(video, s, e, K, phases, key, root, subs, os.path.join(a.out, tag + '.mp4'), tmp, extra=xv)
        make_sheet(video, s, e, K, phases, key, root, os.path.join(a.out, tag + '.png'), tmp, extra=xs if pred is not None else ())
        L = e - s
        sheet += [f'## {i}. {key}  —  code {code} {"HELD-OUT" if is_held else "seen"}  —  `{vid}` {s:.1f}–{e:.1f} s ({L:.1f} s)' + (f"  —  chunk/annotation tIoU {meta[vid]['tiou']:.2f}" if vid in meta else ''),
                  f'files: `{tag}.png`, `{tag}.mp4`', '',
                  'annotated sub-actions: ' + '; '.join(f'{n} [{a0:.1f}–{b0:.1f}]' for n, a0, b0 in subs), '',
                  *([('predictions overlapping the dive: ' + ('; '.join(fmt_pred(p, name_of, j + 1) for j, p in enumerate(preds)) if preds else 'none')), ''] if pred is not None else []),
                  '| k | time (s) | LLM phase sentence | annotated sub-action then | C / P / W | note |', '|---|---|---|---|---|---|']
        for k, t in enumerate(phases):
            mid = s + (k + 0.5) * L / K
            gt = next((n for n, a0, b0 in subs if a0 <= mid < b0), '')
            sheet.append(f'| {k+1} | {s + k*L/K:.1f}–{s + (k+1)*L/K:.1f} | {t} | {gt} |  |  |')
        sheet.append('')
    open(os.path.join(a.out, 'VERIFY_SHEET.md'), 'w').write('\n'.join(sheet) + '\n')
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print('done:', len(chosen), 'dives in', a.out)


if __name__ == '__main__':
    main()
