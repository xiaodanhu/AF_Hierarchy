#!/usr/bin/env python
"""Build a human-verification kit for the LLM sub-action trees.

The question is whether the LLM's ordered sub-action sentences match what
actually happens in the video. The model assigns sentence k to the k-th of K
equal-length chunks of an action segment, so that is what a human should check:
for each ground-truth action instance, does chunk k show what sentence k says,
and is the order right?

For every chosen instance this writes
  * <id>.mp4   the ground-truth segment, slowed to half speed, with the current
               sub-action sentence overlaid during its chunk and a header giving
               class, chunk index and time range;
  * <id>.png   a contact sheet: the middle frame of each of the K chunks with its
               sentence underneath, for a one-glance check;
and one VERIFY_SHEET.md listing every sentence with its time range and a box to
tick, so the result can be tallied.
"""
import argparse, json, os, subprocess, textwrap
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def pick_instances(db, classes, min_dur, max_dur, per_class):
    """Per class: the instance whose duration is closest to the class median within
    [min_dur, max_dur], test videos preferred. Avoids outliers such as annotations
    that span several repetitions."""
    import statistics
    out = {}
    for vid, v in db.items():
        is_test = vid.startswith('video_test')
        for a in v['annotations']:
            lab = a['label']
            if lab not in classes:
                continue
            s, e = a['segment']
            d = e - s
            if d < min_dur or d > max_dur:
                continue
            out.setdefault(lab, []).append((is_test, d, vid, s, e))
    chosen = []
    for lab in classes:
        cands = out.get(lab, [])
        if not cands:
            print('no instance in range for', lab); continue
        med = statistics.median(d for _, d, _, _, _ in cands)
        cands.sort(key=lambda r: (not r[0], abs(r[1] - med)))
        for is_test, d, vid, s, e in cands[:per_class]:
            chosen.append((lab, vid, s, e))
    return chosen


def wrap(s, width=58):
    return '\n'.join(textwrap.wrap(s, width))


def make_video(video, s, e, K, sents, cls, root, out_mp4, tmpdir, slow=2.0, width=960, extra=()):
    """extra: [(text, color, a0, b0)] overlay lines shown during [a0, b0] (absolute seconds), e.g. the prediction."""
    L = e - s
    filters = [f'scale={width}:-2', f'setpts={slow}*PTS']
    for i, (txt, col, a0, b0) in enumerate(extra):
        a, b = slow * max(0.0, a0 - s), slow * min(L, b0 - s)
        tp = os.path.join(tmpdir, f'x{i}.txt'); open(tp, 'w').write(txt)
        filters.append(f"drawtext=fontfile={FONT_B}:textfile={tp}:fontsize=22:fontcolor={col}:box=1:boxcolor=black@0.65:"
                       f"boxborderw=8:x=14:y={58 + 34 * i}:enable='gte(t,{a:.3f})*lt(t,{b:.3f})'")
    # header: class + root sentence (always on)
    hp = os.path.join(tmpdir, 'hdr.txt')
    open(hp, 'w').write(f'{cls}   |   {wrap(root, 90)}')
    filters.append(f"drawtext=fontfile={FONT_B}:textfile={hp}:fontsize=20:fontcolor=white:"
                   f"box=1:boxcolor=black@0.55:boxborderw=8:x=14:y=12")
    for k in range(K):
        a, b = slow * k * L / K, slow * (k + 1) * L / K
        tp = os.path.join(tmpdir, f'c{k}.txt')
        open(tp, 'w').write(f'sub-action {k+1}/{K}   [{s + k*L/K:.1f}s - {s + (k+1)*L/K:.1f}s of the video]\n'
                            + wrap(sents[k], 62))
        filters.append(f"drawtext=fontfile={FONT}:textfile={tp}:fontsize=24:fontcolor=yellow:"
                       f"box=1:boxcolor=black@0.65:boxborderw=10:x=14:y=h-th-14:"
                       f"enable='gte(t,{a:.3f})*lt(t,{b:.3f})'")
    cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f'{s:.3f}', '-to', f'{e:.3f}', '-i', video,
           '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', out_mp4]
    subprocess.run(cmd, check=True)


def make_sheet(video, s, e, K, sents, cls, root, out_png, tmpdir, cols=5, fw=384, extra=()):
    """extra: [(text, color)] lines drawn under the header (e.g. the model's prediction)."""
    L = e - s
    frames = []
    for k in range(K):
        t = s + (k + 0.5) * L / K
        fp = os.path.join(tmpdir, f'f{k}.png')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{t:.3f}', '-i', video, '-frames:v', '1',
                        '-vf', f'scale={fw}:-2', fp], check=True)
        frames.append(Image.open(fp).convert('RGB'))
    fh = frames[0].height
    font = ImageFont.truetype(FONT, 15); fontb = ImageFont.truetype(FONT_B, 16); fonth = ImageFont.truetype(FONT_B, 20)
    cap_h = 15 * 6 + 24
    rows = (K + cols - 1) // cols
    hdr = 70 + 22 * len(extra)
    W, H = cols * fw + (cols + 1) * 12, hdr + rows * (fh + cap_h + 16)
    sheet = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(sheet)
    d.text((12, 10), f'{cls}  —  {root}', fill='black', font=fonth)
    d.text((12, 40), f'ground-truth segment {s:.1f}–{e:.1f} s ({L:.1f} s), split into {K} equal chunks; '
                     f'frame = middle of each chunk', fill='gray', font=font)
    for i, (txt, col) in enumerate(extra):
        d.text((12, 62 + 22 * i), txt, fill=col, font=fontb)
    for k in range(K):
        r, c = divmod(k, cols)
        x, y = 12 + c * (fw + 12), hdr + r * (fh + cap_h + 16)
        sheet.paste(frames[k], (x, y))
        d.rectangle([x, y, x + 110, y + 24], fill='black')
        d.text((x + 6, y + 4), f'{k+1}/{K}  {s + k*L/K:.1f}–{s + (k+1)*L/K:.1f}s', fill='yellow', font=fontb)
        d.text((x, y + fh + 6), wrap(sents[k], 46), fill='black', font=font)
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', default='/data3/xiaodan8/thumos/annotations/thumos14.json')
    ap.add_argument('--videos', default='/data3/xiaodan8/thumos/video')
    ap.add_argument('--trees', default='/data3/xiaodan8/tifad/configs/thumos14_qwen_ensemble.json')
    ap.add_argument('--out', default='/data3/xiaodan8/actionformer6/tools/verify_clips')
    ap.add_argument('--classes', nargs='+', default=['LongJump', 'HighJump', 'Billiards', 'GolfSwing',
                                                     'CleanAndJerk', 'Diving', 'HammerThrow', 'PoleVault'])
    ap.add_argument('--per_class', type=int, default=1)
    ap.add_argument('--min_dur', type=float, default=5.0)
    ap.add_argument('--max_dur', type=float, default=25.0)
    ap.add_argument('--selection', default='', help='ordered instance list (thumos_selection.json) instead of the median rule')
    ap.add_argument('--slow', type=float, default=2.0)
    a = ap.parse_args()

    j = json.load(open(a.ann)); db = j.get('database', j)
    trees = json.load(open(a.trees))
    os.makedirs(a.out, exist_ok=True); tmp = os.path.join(a.out, '_tmp'); os.makedirs(tmp, exist_ok=True)
    if a.selection:
        selj = json.load(open(a.selection)); chosen = [(d['cls'], d['clip'], d['s'], d['e']) for d in selj]; meta = {(d['cls'], d['clip']): d for d in selj}
    else:
        chosen = pick_instances(db, a.classes, a.min_dur, a.max_dur, a.per_class); meta = {}

    sheet = ['# Human verification of LLM sub-action trees (THUMOS14)', '',
             'For each example: open the `.png` for a one-glance check, or the `.mp4` (half speed; the',
             'current sentence is overlaid during its chunk). Chunks are the K = 10 equal parts of the',
             'ground-truth segment, exactly what the model aligns sentence k to.', '',
             'Tick per sentence:  **C** = correct for this action,  **P** = happens, but at a different position,',
             '**W** = wrong or hallucinated.  Add a note if useful.', '',
             *(['Instances: one random test instance per class (seed 0, 3-25 s), the draw replaced when its annotation was unusable',
                '(span not covering the action, camera cut, far dark view); grade A = span covers the action, B = span with a cut or a long lead. A first.', ''] if a.selection else [])]
    for i, (cls, vid, s, e) in enumerate(chosen, 1):
        sents = trees[cls]; root, subs = sents[0], sents[1:]; K = len(subs)
        video = os.path.join(a.videos, vid + '.mp4')
        tag = f'{i:02d}_{cls}_{vid}'
        print(f'[{i}/{len(chosen)}] {cls}  {vid}  {s:.1f}-{e:.1f}s', flush=True)
        make_video(video, s, e, K, subs, cls, root, os.path.join(a.out, tag + '.mp4'), tmp, slow=a.slow)
        make_sheet(video, s, e, K, subs, cls, root, os.path.join(a.out, tag + '.png'), tmp)
        L = e - s
        m = meta.get((cls, vid), {})
        sheet += [f'## {i}. {cls}  —  `{vid}`  {s:.1f}–{e:.1f} s  ({L:.1f} s)' + (f"  —  grade {m['grade']}" + (f" ({m['note']})" if m.get('note') else '') if m else ''),
                  f'files: `{tag}.png`, `{tag}.mp4`', '', f'root: *{root}*', '',
                  '| k | video time (s) | sub-action sentence | C / P / W | note |', '|---|---|---|---|---|']
        for k, t in enumerate(subs):
            sheet.append(f'| {k+1} | {s + k*L/K:.1f}–{s + (k+1)*L/K:.1f} | {t} |  |  |')
        sheet.append('')
    open(os.path.join(a.out, 'VERIFY_SHEET.md'), 'w').write('\n'.join(sheet) + '\n')
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print('done:', len(chosen), 'examples in', a.out)


if __name__ == '__main__':
    main()
