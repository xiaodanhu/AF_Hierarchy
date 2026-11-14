#!/usr/bin/env python
"""Diagnose the collapsed FineDiving C3 run (11-09): the tuned adapter emitted the same held-out class (c04,
"forward dive, 3.5 somersaults, tuck position", 5th in the fixed class list) with no children for all 749 test
windows, while the untuned base model produced varied trees. Hypothesis: prompt shift -- training prompts list
<= 8 classes (Prompter.train_prompt: positives + random seen fill), the test prompt lists all 52 (test_prompt).
Runs the tuned adapter (greedy) on a few test windows under several prompt variants, and on a few training
windows with their own training prompt (does the adapter reproduce what it was trained on?).
  CUDA_VISIBLE_DEVICES=3 python scripts/vlm_hier/diag_fd_prompt.py --adapter ckpt/vlm_hier/fd_s1/final
"""
import argparse, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from train import Prompter
from infer import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/vlm_hier/fd_s1'); ap.add_argument('--adapter', default='ckpt/vlm_hier/fd_s1/final')
    ap.add_argument('--n_test', type=int, default=12); ap.add_argument('--n_train', type=int, default=6)
    ap.add_argument('--size', type=int, default=336); ap.add_argument('--max_new', type=int, default=192)
    ap.add_argument('--variants', default='all52,all52_shuf,train8,seen39,first8', help='comma list; the chain sanity gate uses all52 only')
    a = ap.parse_args()
    rng = random.Random(0)
    P = Prompter('fd', a.data)
    test = [json.loads(l) for l in open(os.path.join(a.data, 'test.jsonl'))]
    tf = os.path.join(a.data, 'train_nested.jsonl'); tf = tf if os.path.exists(tf) else os.path.join(a.data, 'train.jsonl')
    train = [json.loads(l) for l in open(tf)] if a.n_train else []
    sel = set(a.variants.split(','))
    held = set(P.meta['held_out'])
    t_seen = [s for s in test if s['target'][0]['cls'] not in held]; t_held = [s for s in test if s['target'][0]['cls'] in held]
    rng.shuffle(t_seen); rng.shuffle(t_held)
    tests = t_seen[:a.n_test // 2] + t_held[:a.n_test - a.n_test // 2]
    tr_named = [s for s in train if not s['target'][0].get('mask_name')]; tr_masked = [s for s in train if s['target'][0].get('mask_name')]
    rng.shuffle(tr_named); rng.shuffle(tr_masked)
    trains = tr_named[:a.n_train // 2] + tr_masked[:a.n_train - a.n_train // 2]
    proc, model = C.load_model(mode='4bit', lora=True, adapter=a.adapter, device='cuda'); model.eval(); dev = model.device

    def run(s, prompt):
        frames = C.load_frames(s['frames'], size=a.size)
        inp = C.video_inputs(proc, C.chat_text(proc, prompt), frames, s['fps'], dev)
        (text, offs, logps), = generate(model, proc, inp, a.max_new, False, 1)
        return text

    def variants(s):
        gt = s['target'][0]['cls']
        fill = [c for c in P.seen if c != gt]; rng.shuffle(fill)
        return {'all52': list(P.all_ids),                                   # the chain's test prompt (fixed order)
                'all52_shuf': rng.sample(P.all_ids, len(P.all_ids)),        # same classes, random order (position vs name)
                'train8': rng.sample([gt] + fill[:7], 8),                   # training-style prompt: GT + 7 random seen
                'seen39': list(P.seen),                                     # seen classes only (diagnostic, not a protocol)
                'first8': list(P.all_ids[:8])}                              # first 8 of the fixed list (GT usually absent)

    print(f'adapter {a.adapter}; {len(tests)} test windows, {len(trains)} train windows', flush=True)
    summary = {}; outs = {}
    for s in tests:
        t = s['target'][0]
        print(f"\n### TEST {s['id']} GT {t['cls']} {'HELD-OUT' if t['cls'] in held else 'seen'} \"{t['action']}\" [{t['start']},{t['end']}] sub={[q['action'] for q in t.get('sub', [])]}", flush=True)
        for name, cls in variants(s).items():
            if name not in sel: continue
            out = run(s, P._prompt(cls, s)); ok = C.target_json(t['action'])[1:-1] in out
            summary.setdefault(name, []).append(ok); outs.setdefault(name, []).append(out)
            print(f"  [{name:11s}] gt_in_list={t['cls'] in cls} name_ok={ok} :: {out[:300]}", flush=True)
    for s in trains:
        t = s['target'][0]
        print(f"\n### TRAIN {s['id']} {'MASKED' if t.get('mask_name') else 'named'} \"{t['action']}\" [{t['start']},{t['end']}] nsub={len(t.get('sub') or [])}", flush=True)
        out = run(s, P.train_prompt(s)); ok = C.target_json(t['action'])[1:-1] in out
        summary.setdefault('train_prompt@train', []).append(ok); print(f"  [train_prompt] name_ok={ok} :: {out[:300]}", flush=True)
        out = run(s, P.test_prompt(s)); ok = C.target_json(t['action'])[1:-1] in out
        summary.setdefault('all52@train', []).append(ok); print(f"  [all52      ] name_ok={ok} :: {out[:300]}", flush=True)
    print('\nSUMMARY name-correct:', {k: f'{sum(v)}/{len(v)}' for k, v in summary.items()}, flush=True)
    for name, o in outs.items():                       # chain sanity gate: 'distinct outputs 1/N' = collapsed
        print(f'SANITY {name}: distinct outputs {len(set(o))}/{len(o)}, names correct {sum(summary[name])}/{len(summary[name])}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
