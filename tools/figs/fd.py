"""Figure: two FineDiving test dives with twelve frames,
the annotated sub-action bar against the K=3 equal chunks, and the FIVE phase sentences of each sub-action
(fd_sub_ensemble_5.json, timing-aware). Dives = the first two held-out dives of the verification kit's
selection (tools/verify_kit/fd_selection.json: random draws from the well-annotated pool, best-aligned
first), so the figure follows the same rule as the kit. Frames via ffmpeg from the untrimmed video.
Previous versions: fd_rgbstore.py (frame store, hand-picked dives), fd_pick5.py, fd_ten_phases.py."""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import Rectangle
A6 = '/data3/xiaodan8/actionformer6'; FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
S = json.load(open(f'{A6}/configs/finediving/fd_sub_ensemble_5.json'))
subv = json.load(open(f'{A6}/configs/finediving/fd_sub_verbalizer.json'))
KEY = {v['name']: f"({v['activity']}) {v['name']}" for v in subv['actions'].values()}
sel = json.load(open(f'{A6}/tools/verify_kit/fd_selection.json'))['selected']
NDIVE = int(os.environ.get('NDIVE', '3'))   # three dives 
DIVES = [d for d in sel if d['held']][:NDIVE] if len(sys.argv) < 2 else [sel[int(i)] for i in sys.argv[1].split(',')]
SHORT = {'takeoff': 'Takeoff', 'somersault': 'Som.', 'somersaults': 'Soms.', 'pike': 'Pike', 'tuck': 'Tuck', 'twist': 'Twist', 'twists': 'Twists', 'armstand': 'Arm.', 'entry into the water': 'Entry'}
def short(n):
    return ' '.join(SHORT.get(w, w.capitalize()) for w in n.replace('entry into the water', 'Entry').split())
tmp = tempfile.mkdtemp(prefix='fdfig_')
H = float(os.environ.get('HH', '9.0')); fig = new_fig(H); y = 0.05; out = []
for pi, d in enumerate(DIVES):
    vid, s, e = d['video'], d['s'], d['e']; L = e - s
    video = f"{FD}/FineDiving/Released_FineDiving_Dataset/Untrimmed_Videos/{vid.rsplit('_', 1)[0]}.mp4"
    names = [n for n, _, _ in d['subs']]; b = [a0 - s for _, a0, _ in d['subs']] + [L]; K = len(names)
    keys = [KEY[n] for n in names]; sent = [S[k][1:6] for k in keys]
    fig.text(0.05 / W, 1 - y / H, f"({'abc'[pi]}) {d['key'].split(') ', 1)[1]}, code {d['code']}", fontsize=7.5, fontweight='bold', va='top')
    y += 0.14
    nf = 5 * K   # one frame at the center of each phase chunk (K sub-action chunks x five phase chunks)
    fig.text(0.05 / W, 1 - y / H, f"video {vid}, {s:.1f}-{e:.1f} s, chunk/annotation tIoU {d['tiou']:.2f}", fontsize=6.3, va='top')
    y += 0.14; ox = 0.05; wi = W - 0.1; tw = wi / nf
    frames = []
    for j in range(nf):
        t = s + (j + 0.5) * L / nf; fp = f'{tmp}/{pi}_{j}.png'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{t:.3f}', '-i', video, '-frames:v', '1', '-vf', 'scale=320:-2', fp], check=True)
        frames.append(Image.open(fp).convert('RGB'))
    th = tw * frames[0].size[1] / frames[0].size[0]
    ax = ax_in(fig, ox, y, wi, th, H); ax.set_xlim(0, L); ax.set_ylim(1, 0)
    for j, fr in enumerate(frames): ax.imshow(np.asarray(fr), extent=(j * L / nf, (j + 1) * L / nf, 1, 0), aspect='auto')
    ax.set_xlim(0, L); ax.set_ylim(1, 0)
    y += th + 0.02
    bh = 0.19; a1 = ax_in(fig, ox, y, wi, bh, H); a1.set_xlim(0, L); a1.set_ylim(1, 0)
    for i in range(K):
        a1.add_patch(Rectangle((b[i], 0), b[i + 1] - b[i], 1, fc='0.92' if i % 2 == 0 else 'white', ec='none'))
        a1.text((b[i] + b[i + 1]) / 2, 0.52, short(names[i]), ha='center', va='center', fontsize=6.5)
    for x in b: a1.plot([x, x], [0, 1], color='black', lw=0.8)
    a1.plot([0, L], [0, 0], color='black', lw=0.5); a1.plot([0, L], [1, 1], color='black', lw=0.5)
    y += bh + 0.04
    a2 = ax_in(fig, ox, y, wi, bh, H); a2.set_xlim(0, L); a2.set_ylim(1, 0)
    for i in range(K): a2.text((i + 0.5) * L / K, 0.52, str(i + 1), ha='center', va='center', fontsize=6.5)
    for i in range(K + 1): a2.plot([i * L / K] * 2, [0, 1], color='black', lw=0.6, ls=(0, (3, 2)))
    for i in range(K):
        for j in range(1, 5): a2.plot([i * L / K + j * L / (5 * K)] * 2, [0.72, 1], color='0.45', lw=0.5)   # the five phase chunks of chunk i
    a2.plot([0, L], [0, 0], color='black', lw=0.4); a2.plot([0, L], [1, 1], color='black', lw=0.4)
    y += bh + 0.02
    if pi == 0:   # legend once, under the first dive (three dives must fit one page)
        legend = textwrap.fill(f"upper bar: annotated sub-actions (solid). lower bar: {K} equal chunks (dashed), each split into five phase chunks (ticks), one frame per phase chunk. Column k: the five phase sentences of sub-action k, aligned with chunk k.", 128)
        fig.text(0.05 / W, 1 - y / H, legend, fontsize=5.8, va='top', color='0.25', linespacing=1.2)
        y += 0.11 * legend.count('\n') + 0.14
    else: y += 0.05
    cw = wi / K; fs = 6.0; lh = fs * 1.15 / 72; yb = y
    for c in range(K):
        x = ox + c * cw; yy = y
        fig.text((x + 0.03) / W, 1 - yy / H, short(names[c]), fontsize=6.5, fontweight='bold', va='top'); yy += 0.14
        for j, sn in enumerate(sent[c]):
            ls = textwrap.wrap(f"{j+1}. " + sn, 48, subsequent_indent='    ')
            fig.text((x + 0.03) / W, 1 - yy / H, "\n".join(ls), fontsize=fs, va='top', linespacing=1.15)
            yy += len(ls) * lh + 0.03
        yb = max(yb, yy)
    y = yb + 0.06
    out.append((d['code'], vid, s, e, names))
print(out, y)
assert y <= 10.2   # scaled to \linewidth (x 0.846) in the appendix
if os.environ.get('HH') is None: print('REDO', y); sys.exit(0)
D = f'{A6}/tools/figures/'
fig.savefig(D + 'tree_examples_finediving.pdf'); fig.savefig(D + 'tree_examples_finediving.png', dpi=150)
