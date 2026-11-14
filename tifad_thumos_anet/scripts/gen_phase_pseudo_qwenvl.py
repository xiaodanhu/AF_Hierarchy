"""Label set D: phase start times asked from a video-language model
(Qwen2.5-VL-7B-Instruct, local snapshot, 4-bit NF4 so it fits next to the
dispatcher jobs).  Pilot: 200 seen-class GT segments of split 0 (>= 1.5 s,
round-robin over the 15 seen classes of the eval population of
scripts/gen_phase_pseudo_v2.py).

Per segment: the trimmed clip is sampled at N = clip(2*L, 16, 32) frames
(uniform inside [s, e]); every frame carries its timestamp relative to the
segment start burnt in at the top-left, and the same timestamps are listed
in the prompt together with the action name and its 10 ordered phase
sentences.  The model answers with JSON {"starts": [10 numbers]} = the start
time (seconds from the clip start) of every phase in order; phase 1 is
forced to 0, the list is made non-decreasing (cumulative max) and clipped
to [0, L]; cuts = s + starts[1:].  One greedy try, then up to 2 sampled
retries when the JSON cannot be parsed; a segment without a parsable answer
gets no cut (the writer falls back to equal-width cuts, flagged).

Output: data/phase_pseudo/v2/cuts/qwenvl_split0.npz (same layout as the
other cut files: vid, j, cuts (n,9)), the raw answers next to it
(qwenvl_split0_raw.json) and, via gen_phase_pseudo_v2.write_json, the label
file data/phase_pseudo/v2/qwenvl_split0.json.

Run (hierarchy env: transformers 4.56 / torch 2.8 / bitsandbytes 0.47):
  cd /data3/xiaodan8/tifad
  CUDA_VISIBLE_DEVICES=<gpu with >= 8 GB free> \
    /data/xiaodan8/anaconda3/envs/hierarchy/bin/python scripts/gen_phase_pseudo_qwenvl.py
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phase_pseudo_v2 as g                                    # noqa: E402

MODEL_DIR = ('/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/'
             'cc594898137f460bfe9f0759e9844b3ce807cfb5')
SYSTEM = ('You are an expert sports video analyst. You watch a short clip of one '
          'action and localise its phases in time. Answer with JSON only.')


def readable(name):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)


def pilot_segments(sp, n_pilot):
    """Round-robin over the seen classes of split sp on segments >= 1.5 s
    that every cached label set has cut (the eval population)."""
    seen, _ = g.split_classes(sp)
    sets = {n: g.load_cuts(n) for n in ('C_vmae', 'C_i3d_both', 'uniform')}
    sets = {k: v for k, v in sets.items() if v is not None}
    keys = g.eval_population(sp, sets) if sets else \
        [s['key'] for s in g.SEGS if s['label'] in set(seen) and s['L'] >= g.MIN_EVAL_SEC]
    by_cls = {}
    for k in sorted(keys):
        by_cls.setdefault(g.SEG_OF[k]['label'], []).append(k)
    order = sorted(by_cls)
    out, i = [], 0
    while len(out) < n_pilot and any(by_cls.values()):
        c = order[i % len(order)]
        i += 1
        if by_cls[c]:
            # spread over the class's videos: take from evenly spaced positions
            lst = by_cls[c]
            out.append(lst.pop(len(lst) // 2))
    return out


def sample_frames(seg, n_max=32, n_min=16):
    import decord
    from PIL import Image, ImageDraw, ImageFont
    vr = decord.VideoReader(g.VIDEO % seg['vid'], num_threads=2)
    n = int(np.clip(round(2 * seg['L']), n_min, n_max))
    rel = (np.arange(n) + 0.5) / n * seg['L']
    idx = np.clip(np.round((seg['s'] + rel) * seg['fps']).astype(int), 0, len(vr) - 1)
    frames = vr.get_batch(idx.tolist()).asnumpy()
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except Exception:
        font = ImageFont.load_default()
    imgs = []
    for f, t in zip(frames, rel):
        im = Image.fromarray(f)
        if im.width < 336:
            im = im.resize((336, int(round(im.height * 336 / im.width))))
        d = ImageDraw.Draw(im)
        txt = f't={t:.2f}s'
        d.rectangle([0, 0, 8 * len(txt) + 8, 22], fill=(0, 0, 0))
        d.text((4, 3), txt, fill=(255, 255, 0), font=font)
        imgs.append(np.asarray(im))
    return np.stack(imgs), rel


def build_prompt(seg, rel):
    phases = g.phase_sentences(seg['label'])
    L = seg['L']
    lines = [f'This clip shows one {readable(seg["label"])} action. The clip starts at t=0.00s '
             f'and ends at t={L:.2f}s. You see {len(rel)} frames sampled at t = '
             + ', '.join(f'{t:.2f}' for t in rel) + ' seconds (the time is also printed on each frame).',
             'The action goes through these 10 phases in this order:']
    for i, p in enumerate(phases):
        lines.append(f'{i + 1}. {p}')
    lines.append(f'Using the visual evidence, give the start time in seconds (from the clip start) of '
                 f'each of the 10 phases, in order. Phase 1 starts at 0.00; the start times must be '
                 f'non-decreasing and at most {L:.2f}. Do not assume equal durations: phases can be '
                 f'short or long. Answer with JSON only: {{"starts": [t1, t2, ..., t10]}}')
    return '\n'.join(lines)


NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def parse(ans, L):
    m = re.search(r'\[([^\]]*)\]', ans)
    if not m:
        return None
    nums = [float(x) for x in NUM_RE.findall(m.group(1))]
    if len(nums) < 10:
        return None
    st = np.asarray(nums[:10], dtype=np.float64)
    st[0] = 0.0
    st = np.maximum.accumulate(np.clip(st, 0.0, L))
    return st


def load_model(device_map):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, quantization_config=q, device_map=device_map, torch_dtype=torch.bfloat16)
    model.eval()
    return proc, model


def ask(proc, model, frames, prompt, sample=False, seed=0, max_new_tokens=160):
    import torch
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': [{'type': 'video', 'video': 'placeholder'},
                                         {'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], videos=[frames], return_tensors='pt').to(model.device)
    gen = dict(max_new_tokens=max_new_tokens, do_sample=sample)
    if sample:
        torch.manual_seed(seed)
        gen.update(temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inp, **gen)
    return proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0], \
        int(inp['input_ids'].shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, default=0)
    ap.add_argument('--n-pilot', type=int, default=200)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device-map', default='{"": 0}')
    args = ap.parse_args()
    os.chdir(g.ROOT)
    name = f'qwenvl_split{args.split}'
    raw_path = f'{g.CUTDIR}/{name}_raw.json'
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    keys = pilot_segments(args.split, args.n_pilot)
    print(f'[qwenvl] pilot {len(keys)} segments of split {args.split}; {len(raw)} answers cached', flush=True)
    proc, model = load_model(json.loads(args.device_map))
    import torch
    print(f'[qwenvl] model loaded, GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GB', flush=True)
    cuts, t0 = {}, time.time()
    for i, key in enumerate(keys):
        seg = g.SEG_OF[key]
        kk = f'{key[0]}|{key[1]}'
        if kk not in raw:
            frames, rel = sample_frames(seg)
            prompt = build_prompt(seg, rel)
            tries = []
            st = None
            for t in range(3):
                ans, n_in = ask(proc, model, frames, prompt, sample=(t > 0), seed=t)
                tries.append(ans)
                st = parse(ans, seg['L'])
                if st is not None:
                    break
            raw[kk] = dict(label=seg['label'], s=seg['s'], e=seg['e'], n_frames=int(len(rel)),
                           n_input_tokens=n_in, answers=tries,
                           starts=None if st is None else st.round(3).tolist())
            json.dump(raw, open(raw_path, 'w'), indent=1)
        r = raw[kk]
        if r['starts'] is not None:
            cuts[key] = np.clip(seg['s'] + np.asarray(r['starts'][1:]), seg['s'], seg['e'])
        if i % 10 == 0:
            print(f'  [{i + 1}/{len(keys)}] {seg["label"]:16s} L={seg["L"]:5.1f}s starts={r["starts"]} '
                  f'({(time.time() - t0) / (i + 1):.1f}s/seg)', flush=True)
    n_fail = sum(1 for k in keys if raw[f'{k[0]}|{k[1]}']['starts'] is None)
    print(f'[qwenvl] {len(cuts)} segments cut, {n_fail} unparsable', flush=True)
    g.save_cuts(name, cuts, dict(model='Qwen2.5-VL-7B-Instruct nf4', n_pilot=len(keys), n_unparsable=n_fail))
    seen, _ = g.split_classes(args.split)
    g.write_json(f'{g.OUTDIR}/{name}.json', cuts, set(seen),
                 dict(split=args.split, method='qwenvl', classes='seen (pilot subset; the rest fallback)',
                      generator='scripts/gen_phase_pseudo_qwenvl.py'))
    print(f'[qwenvl] wrote {g.CUTDIR}/{name}.npz and {g.OUTDIR}/{name}.json', flush=True)


if __name__ == '__main__':
    main()
