#!/usr/bin/env python
"""Self-labelling of child spans for C3 (tree setting: ONE annotated level + our definition tree).

For every training window with targets, the BASE Qwen2.5-VL (zero-shot) sees the ground-truth
parents and their ordered children from the definition tree and places the children in time,
using the video and its own priors — variable lengths, never the equal split.
  THUMOS    : children = the K=10 steps -> K+1 "cuts"
  FineDiving: children = the dive's ordered members -> one span per member ("sub")
Invalid placements are retried with sampling; if still invalid, the LLM's text-only duration
prior (gen_step_prior.py) fills the spans. Writes train_nested.jsonl (resumable), with the
source of every child set recorded (vlm / vlm_sample / prior).

  CUDA_VISIBLE_DEVICES=2 python scripts/vlm_hier/ground.py --dataset thumos --data data/vlm_hier/thumos_s0
  CUDA_VISIBLE_DEVICES=3 python scripts/vlm_hier/ground.py --dataset fd     --data data/vlm_hier/fd_s1
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['thumos', 'fd'], required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--prior', default='', help='step prior json (default configs/<ds>_step_prior.json)')
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--mode', default='4bit'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--size', type=int, default=0); ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--max_new', type=int, default=0)
    a = ap.parse_args()
    if not a.size:
        a.size = 280 if a.dataset == 'thumos' else 336   # THUMOS: 280 px (100 tokens / frame pair) keeps dense 64 s windows under ~6.5k tokens; FineDiving 336 (small diver)
    import torch
    thumos = a.dataset == 'thumos'
    prior_path = a.prior or ('configs/thumos14_step_prior.json' if thumos else 'configs/finediving_step_prior.json')
    prior = json.load(open(prior_path)) if os.path.exists(prior_path) else {}
    if not prior:
        print(f'WARNING: no prior at {prior_path}; fallback will be uniform', flush=True)
    trees = json.load(open('configs/thumos14_qwen_ensemble.json')) if thumos else None
    meta = json.load(open(os.path.join(a.data, 'meta.json')))
    K = (len(next(iter(trees.values()))) - 1) if thumos else None
    sub_roots = meta.get('sub_roots', {})
    max_new = a.max_new or (512 if thumos else 256)
    samples = [json.loads(l) for l in open(os.path.join(a.data, 'train.jsonl'))]
    if a.limit:
        samples = samples[:a.limit]
    out_path = os.path.join(a.data, 'train_nested.jsonl')
    done = {json.loads(l)['id'] for l in open(out_path)} if os.path.exists(out_path) else set()
    proc, model = C.load_model(mode=a.mode, lora=False, device=a.device); model.eval()
    dev = model.device if a.device == 'cuda' else 'cpu'; tok = proc.tokenizer
    f = open(out_path, 'a'); t0 = time.time(); src_cnt = collections.Counter(); n_new = 0

    def gen(inp, sample, seed):
        if sample:
            torch.manual_seed(seed)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=sample, temperature=0.7 if sample else None,
                               top_p=0.9 if sample else None, pad_token_id=tok.pad_token_id)
        return tok.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)

    def match(pred, t):
        best, bd = None, 1e9
        for j, p in enumerate(pred):
            if p['action'] != t['action'].lower():
                continue
            d = abs(p['start'] - t['start']) + abs(p['end'] - t['end'])
            if d < bd:
                best, bd = j, d
        return pred[best] if best is not None and bd <= 2.0 else None

    for i, s in enumerate(samples):
        if s['id'] in done:
            continue
        todo = [t for t in s['target'] if not t.get('mask_name')]            # held-out (keep): no children
        if not todo:
            f.write(json.dumps(s) + '\n'); n_new += 1; continue
        frames = C.load_frames(s['frames'][:a.max_frames] if a.max_frames else s['frames'], size=a.size or None)
        if thumos:
            prompt = C.prompt_thumos_ground(todo, trees, s['win_len'], s['fps'])
        else:
            prompt = C.prompt_fd_ground([{**t, 'members': [q['action'] for q in t['sub']]} for t in todo], sub_roots, s['win_len'], s['fps'])
        inp = C.video_inputs(proc, C.chat_text(proc, prompt), frames, s['fps'], dev)
        placed = {}                                                      # target index -> (children, src)
        raws = []
        for attempt in range(1 + a.retries):
            try:
                text = gen(inp, attempt > 0, attempt)
            except Exception as ex:
                print(f'  [{s["id"]}] generation failed: {ex}', flush=True); break
            raws.append(text)
            pred = C.parse_instances(text, s['win_len'], nested=True)
            for ti, t in enumerate(todo):
                if ti in placed:
                    continue
                p = match(pred, t)
                if p is None:
                    continue
                if thumos:
                    cand = [C.r1(x) for x in p.get('cuts', [])]
                    if C.valid_cuts(cand, t['start'], t['end'], K):
                        placed[ti] = (cand, 'vlm' if attempt == 0 else 'vlm_sample')
                else:
                    members = [q['action'] for q in t['sub']]
                    cand = [{'action': q.get('action', ''), 'start': C.r1(q['start']), 'end': C.r1(q['end'])} for q in p.get('sub', [])]
                    if C.valid_member_spans(cand, members, t['start'], t['end']):
                        for q, m in zip(cand, members):
                            q['action'] = m
                        placed[ti] = (cand, 'vlm' if attempt == 0 else 'vlm_sample')
            if len(placed) == len(todo):
                break
        new_t = []
        for t in s['target']:
            nt = dict(t)
            if t.get('mask_name'):
                new_t.append(nt); continue
            ti = todo.index(t)
            if ti in placed:
                ch, src = placed[ti]
            else:
                src = 'prior'
                if thumos and os.environ.get('VLM_HIER_LENIENT', '') == '1':   # off by default (degenerate recoveries, 11-08)
                    for text in raws:
                        c = C.lenient_cuts(text, t['action'], t['start'], t['end'], K)
                        if c:
                            ch, src = c, 'vlm_lenient'; break
                if src == 'prior' and thumos:
                    fr = prior.get(t['cls'], {}).get('fractions') or [1.0 / K] * K
                    ch = C.prior_cuts(t['start'], t['end'], fr)
                elif src == 'prior':
                    members = [q['action'] for q in t['sub']]
                    fr = prior.get(t['cls'], {}).get('fractions')
                    if not fr or len(fr) != len(members):
                        fr = [1.0 / len(members)] * len(members)
                    cuts = C.prior_cuts(t['start'], t['end'], fr)
                    ch = [{'action': m, 'start': cuts[k], 'end': cuts[k + 1]} for k, m in enumerate(members)]
            if thumos:
                nt['cuts'] = ch
            else:
                nt['sub'] = ch
            nt['child_src'] = src; src_cnt[src] += 1
            new_t.append(nt)
        s2 = dict(s); s2['target'] = new_t; s2['ground_raw'] = raws
        f.write(json.dumps(s2) + '\n'); f.flush(); n_new += 1
        if n_new % 20 == 0 or i == len(samples) - 1:
            tot = sum(src_cnt.values())
            print(f'[{i+1}/{len(samples)}] children: ' + ', '.join(f'{k} {v} ({100*v/max(1,tot):.0f} %)' for k, v in src_cnt.items()) + f', {(time.time()-t0)/60:.1f} min', flush=True)
    print(f'done -> {out_path}: {dict(src_cnt)}', flush=True)


if __name__ == '__main__':
    main()
