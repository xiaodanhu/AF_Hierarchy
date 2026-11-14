#!/usr/bin/env python
"""Training-free baseline where a video-LLM localizes the unseen actions directly.

A text-only LLM call on the text tree cannot place an action in time, so the
meaningful version of this baseline is a video-LLM: Qwen2.5-VL-7B
receives the untrimmed test video as 1-fps frames in sliding windows, plus the
candidate (unseen) classes with their LLM-generated definition trees, and is
asked for every instance as "<class> | start | end | confidence". Predictions
from overlapping windows are merged with per-class NMS and scored with the same
evaluator, split file and tIoU grid as the trained models (THUMOS14 75/25).

    # pipeline check without the model (oracle = GT -> mAP ~1; random -> ~0):
    python scripts/vlm_localize_baseline.py --n 0 --mock oracle --limit 3
    # real run (one 16 GB GPU, 8-bit):
    CUDA_VISIBLE_DEVICES=2 python scripts/vlm_localize_baseline.py --n 0
Run with the `hierarchy` env (transformers 4.56, bitsandbytes).
"""
import argparse, json, os, re, sys, time, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANN = '/data3/xiaodan8/thumos/annotations/thumos14.json'
VIDEOS = '/data3/xiaodan8/thumos/video'
TREES = 'configs/thumos14_qwen_ensemble.json'
SPLIT = 'splits/train_75_test_25/THUMOS14/test/split_{n}.list'
MODEL = '/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'


def readable(name):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).lower()


# ----------------------------------------------------------------------------- video
def decode_1fps(path, size):
    """Sequential decode at ~1 fps, resized to size x size (multiple of 28). Returns
    (list of HxWx3 uint8 RGB arrays, list of their timestamps in seconds).
    Uses PyAV when present, otherwise OpenCV (the LLM env has cv2 only)."""
    from PIL import Image
    frames, ts = [], []
    try:
        import av
        with av.open(path) as c:
            st = c.streams.video[0]; st.thread_type = 'AUTO'
            tb = float(st.time_base); nxt = 0.0
            for fr in c.decode(st):
                t = fr.pts * tb if fr.pts is not None else nxt
                if t + 1e-6 < nxt:
                    continue
                frames.append(np.asarray(fr.to_image().resize((size, size), Image.BILINEAR))); ts.append(t); nxt = t + 1.0
        return frames, ts
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps)))
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    frames.append(rgb); ts.append(i / fps)
            i += 1
        cap.release()
        return frames, ts


# ----------------------------------------------------------------------------- prompt
def build_prompt(classes, trees, win_len):
    lines = [f"This clip is {win_len} seconds long, shown at 1 frame per second.",
             "It may contain zero, one or several instances of these sports actions:"]
    for c in classes:
        root, subs = trees[c][0], trees[c][1:]
        lines.append(f'\n* {readable(c)}: {root}. It unfolds as: ' + ' '.join(f'({k+1}) {s}' for k, s in enumerate(subs)))
    lines.append("\nFind every instance of any listed action in this clip. For each one output exactly one line:")
    lines.append("<action name> | <start second> | <end second> | <confidence between 0 and 1>")
    lines.append("Times are seconds from the start of this clip. Be precise about when the action starts and ends. "
                 "If none of the listed actions occurs, output exactly: none")
    return '\n'.join(lines)


LINE = re.compile(r'^\s*\*?\s*([A-Za-z][A-Za-z ]+?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*([0-9]*\.?[0-9]+)')


def parse(text, name2cls, win_len):
    out = []
    for ln in text.splitlines():
        m = LINE.match(ln)
        if not m:
            continue
        name = re.sub(r'\s+', ' ', m.group(1).strip().lower())
        cls = name2cls.get(name) or name2cls.get(name.replace(' ', ''))
        if cls is None:
            continue
        s, e, conf = float(m.group(2)), float(m.group(3)), float(m.group(4))
        s, e = max(0.0, min(s, win_len)), max(0.0, min(e, win_len))
        if e <= s:
            continue
        out.append((cls, s, e, max(0.0, min(conf, 1.0))))
    return out


# ----------------------------------------------------------------------------- model
class VLM:
    def __init__(self, model_dir, device='cuda'):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_dir)
        if device == 'cpu':                                   # smoke tests only: bf16, no quantization
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, device_map='cpu').eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=device).eval()

    def ask(self, frames, prompt, max_new_tokens=400):
        msgs = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        video = np.stack(frames)                                  # (T, H, W, 3) uint8
        try:
            inputs = self.proc(text=[text], videos=[video], return_tensors='pt', fps=1.0)
        except TypeError:
            inputs = self.proc(text=[text], videos=[video], return_tensors='pt')
        inputs = inputs.to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]


# ----------------------------------------------------------------------------- merge + eval
def nms(dets, thr=0.5):
    """dets: list of (cls, s, e, score) absolute seconds -> greedy per-class NMS."""
    keep = []
    for cls in {d[0] for d in dets}:
        ds = sorted([d for d in dets if d[0] == cls], key=lambda d: -d[3])
        kept = []
        for d in ds:
            if all((min(d[2], k[2]) - max(d[1], k[1])) / (max(d[2], k[2]) - min(d[1], k[1])) < thr for k in kept):
                kept.append(d)
        keep += kept
    return keep


def evaluate(preds, n):
    from libs.utils.metrics import ANETdetection
    import torch
    det = ANETdetection(ANN, 'test', tiou_thresholds=np.linspace(0.3, 0.7, 5), subset_file=SPLIT.format(n=n))
    res = {'video-id': [p[0] for p in preds], 't-start': np.array([p[1] for p in preds]),
           't-end': np.array([p[2] for p in preds]), 'label': np.array([p[3] for p in preds]),
           'score': np.array([p[4] for p in preds])}
    det.evaluate(res, verbose=True)                # fills det.ap = (tIoU x class) AP matrix
    per = det.ap.mean(axis=1)                      # mAP per tIoU threshold
    return float(per.mean()), [float(x) for x in per]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--stride', type=int, default=48)
    ap.add_argument('--size', type=int, default=336, help='frame side, multiple of 28')
    ap.add_argument('--limit', type=int, default=0, help='first N test videos only')
    ap.add_argument('--mock', choices=['', 'oracle', 'random'], default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vids', default='', help='comma-separated video ids (overrides --limit)')
    ap.add_argument('--no_eval', action='store_true', help='inference only; save predictions (LLM env lacks the evaluator deps)')
    ap.add_argument('--eval_json', default='', help='score a saved predictions json and exit (run in the parsing2 env)')
    ap.add_argument('--max_windows', type=int, default=0, help='smoke: stop after N windows per video')
    a = ap.parse_args()

    if a.eval_json:
        d = json.load(open(a.eval_json)); preds = [tuple(x) for x in d['preds']]
        mAP, per = evaluate(preds, a.n)
        d.update({'mAP': mAP, 'per_tiou': per}); json.dump(d, open(a.eval_json, 'w'))
        print(f'RESULT split {a.n} (Qwen2.5-VL-7B, from {a.eval_json}): unseen mAP@[0.3:0.7] = {100*mAP:.2f}  per tIoU {[round(100*x,1) for x in per]}')
        return
    db = json.load(open(ANN)); db = db.get('database', db)
    trees = json.load(open(TREES))
    unseen = [l.strip() for l in open(SPLIT.format(n=a.n)) if l.strip()]
    # the evaluator built with subset_file renumbers classes by their line order in
    # that file (ANETdetection: label_dict = {name: line index}); predictions must match
    label_id = {c: i for i, c in enumerate(unseen)}
    name2cls = {readable(c): c for c in unseen}
    name2cls.update({c.lower(): c for c in unseen})
    prompt = build_prompt(unseen, trees, a.window)
    out = a.out or f'logs/vlm_s{a.n}{"_" + a.mock if a.mock else ""}.json'
    print(f'split {a.n}: unseen {unseen}; prompt {len(prompt.split())} words; window {a.window}s stride {a.stride}s', flush=True)

    vids = sorted(k for k, v in db.items() if k.startswith('video_test'))
    if a.vids:
        vids = [v for v in a.vids.split(',') if v]
    elif a.limit:
        vids = vids[:a.limit]
    vlm = None if a.mock else VLM(a.model, a.device)
    preds, raw = [], {}
    t0 = time.time()
    for i, vid in enumerate(vids):
        path = os.path.join(VIDEOS, vid + '.mp4')
        dur = db[vid]['duration']
        if a.mock:
            if a.mock == 'oracle':
                dets = [(e['label'], e['segment'][0], e['segment'][1], 1.0) for e in db[vid]['annotations'] if e['label'] in unseen]
            else:
                rng = random.Random(i)
                dets = [(rng.choice(unseen), s := rng.uniform(0, dur), min(dur, s + rng.uniform(1, 10)), rng.random()) for _ in range(5)]
        else:
            frames, ts = decode_1fps(path, a.size)
            dets = []
            for wi, w0 in enumerate(range(0, max(1, len(frames) - a.window // 2), a.stride)):
                if a.max_windows and wi >= a.max_windows:
                    break
                chunk = frames[w0:w0 + a.window]
                if len(chunk) < 4:
                    break
                try:
                    txt = vlm.ask(chunk, prompt)
                except Exception as ex:
                    print(f'  [{vid} @ {w0}s] generation failed: {ex}', flush=True); continue
                raw[f'{vid}@{w0}'] = txt
                for cls, s, e, conf in parse(txt, name2cls, len(chunk)):
                    dets.append((cls, ts[w0] + s, ts[w0] + e, conf))
            dets = nms(dets)
        preds += [(vid, s, e, label_id[cls], conf) for cls, s, e, conf in dets]
        if (i + 1) % 10 == 0 or i == len(vids) - 1:
            print(f'[{i+1}/{len(vids)}] {vid}: {len(dets)} dets, {len(preds)} total, {(time.time()-t0)/60:.1f} min', flush=True)
            json.dump({'preds': preds, 'raw': raw}, open(out, 'w'))

    if not preds:
        print('no predictions'); return
    if a.no_eval:
        print(f'saved {len(preds)} predictions to {out}; score with --eval_json {out}'); return
    mAP, per = evaluate(preds, a.n)
    json.dump({'preds': preds, 'raw': raw, 'mAP': mAP, 'per_tiou': per,
               'setting': vars(a), 'unseen': unseen}, open(out, 'w'))
    print(f'RESULT split {a.n} ({"mock " + a.mock if a.mock else "Qwen2.5-VL-7B"}): unseen mAP@[0.3:0.7] = {100*mAP:.2f}  per tIoU {[round(100*x,1) for x in per]}')


if __name__ == '__main__':
    main()
