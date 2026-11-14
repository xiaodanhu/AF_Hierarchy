"""Figure: three FineGym held-out elements (gym99 test split, element
level) with one frame at the center of each of the K=10 equal chunks, the chunk bar, and the ten sub-action sentences of the
element-level tree (configs/finegym_qwen_ensemble.json), as in the THUMOS14 figure (thumos.py). Elements = the first held-out
element of each apparatus in the verification kit order (tools/verify_kit/fg_selection.json: random draws from the test
split, unusable annotations replaced, grade A first), so the figure follows the same rule as the kit. Frames via ffmpeg from
the full video. Two passes: `python finegym.py` prints REDO <height>, then `HH=<height> python finegym.py` writes PDF/PNG.
Optional argv: comma-separated indices into fg_selection.json."""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import Rectangle
A6 = '/data3/xiaodan8/actionformer6'; FG = '/data3/xiaodan8/FineGym'
acts = json.load(open(f'{A6}/configs/finegym_zsl_verbalizer.json'))['actions']
S = json.load(open(f'{A6}/configs/finegym_qwen_ensemble.json'))
sel = json.load(open(f'{A6}/tools/verify_kit/fg_selection.json'))
key_of = lambda cid: f"({acts[cid]['activity']}) {acts[cid]['name']}"
NEL = int(os.environ.get('NEL', '4'))   # four actions, one per apparatus 
if len(sys.argv) > 1: INST = [sel[int(i)] for i in sys.argv[1].split(',')]
else:
    INST, seen_app = [], set()
    for d in sel:
        if d['held'] and acts[d['cid']]['activity'] not in seen_app:
            INST.append(d); seen_app.add(acts[d['cid']]['activity'])
        if len(INST) == NEL: break
K = 10
def find_video(yid):
    for c in (f'{FG}/video/gym_{yid}.mp4', f'{FG}/video_new/{yid}.mp4', f'{FG}/video_new/{yid}.mkv', f'{FG}/video/gym_{yid}.mkv'):
        if os.path.exists(c): return c
    raise FileNotFoundError(yid)
tmp = tempfile.mkdtemp(prefix='fgfig_')
H = float(os.environ.get('HH', '9.0')); fig = new_fig(H); y = 0.05; out = []
for pi, d in enumerate(INST):
    cid, clip, s, e = d['cid'], d['clip'], d['s'], d['e']; L = e - s
    video = find_video(clip.split('_E_')[0])
    sent = S[key_of(cid)][1:1 + K]; assert len(sent) == K
    fig.text(0.05 / W, 1 - y / H, f"({'abcd'[pi]}) {cid} {key_of(cid)}", fontsize=7.5, fontweight='bold', va='top'); y += 0.14
    fig.text(0.05 / W, 1 - y / H, f"clip {clip}, {s:.1f}-{e:.1f} s ({L:.1f} s), one frame at the center of each of the {K} equal chunks", fontsize=6.3, va='top'); y += 0.14
    ox = 0.05; wi = W - 0.1; tw = wi / K; frames = []
    for j in range(K):
        t = s + (j + 0.5) * L / K; fp = f'{tmp}/{pi}_{j}.png'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{t:.3f}', '-i', video, '-frames:v', '1', '-vf', 'scale=320:-2', fp], check=True)
        frames.append(Image.open(fp).convert('RGB'))
    fh = tw * frames[0].size[1] / frames[0].size[0]
    ax = ax_in(fig, ox, y, wi, fh, H); ax.set_xlim(s, e); ax.set_ylim(1, 0)
    for j, fr in enumerate(frames): ax.imshow(np.asarray(fr), extent=(s + j * L / K, s + (j + 1) * L / K, 1, 0), aspect='auto', interpolation='bilinear')
    ax.set_xlim(s, e); ax.set_ylim(1, 0)
    for k in range(1, K): ax.plot([s + k * L / K] * 2, [0, 1], color='white', lw=0.4, ls=(0, (2, 2)), alpha=0.9)
    y += fh + 0.02
    bh = 0.16; ab = ax_in(fig, ox, y, wi, bh, H); ab.set_xlim(s, e); ab.set_ylim(1, 0)
    for k in range(K):
        a, b = s + k * L / K, s + (k + 1) * L / K
        ab.add_patch(Rectangle((a, 0), b - a, 1, fc='0.92' if k % 2 == 0 else 'white', ec='black', lw=0.5))
        ab.text((a + b) / 2, 0.55, str(k + 1), ha='center', va='center', fontsize=6.5)
    y += bh + 0.03
    ax2 = ax_in(fig, ox, y, wi, 0.1, H); ax2.set_xlim(s, e)
    ax2.annotate('', xy=(e, 0.5), xytext=(s, 0.5), arrowprops=dict(arrowstyle='<->', lw=0.4, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax2.text((s + e) / 2, 0.5, f" {L:.1f} s, {K} equal chunks of {L / K:.2f} s ", ha='center', va='center', fontsize=5.8, bbox=dict(fc='white', ec='none', pad=0.5))
    y += 0.14
    y = sentences(fig, H, y, sent) + 0.05   # tighter, four actions must fit one page
    out.append((cid, clip, s, e))
print(out, 'height used', y)
assert y <= 10.5
if os.environ.get('HH') is None: print('REDO', y); sys.exit(0)
fig.set_size_inches(W, H)
D = f'{A6}/tools/figures/'
fig.savefig(D + 'tree_examples_finegym.pdf'); fig.savefig(D + 'tree_examples_finegym.png', dpi=150)
