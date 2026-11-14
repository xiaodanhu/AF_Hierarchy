#!/usr/bin/env python
"""QLoRA fine-tuning of Qwen2.5-VL-7B to emit the hierarchical tree of a window (C3).

One window per step (batch 1, gradient accumulation), loss on the assistant's JSON
answer only (lm_head applied to the answer positions via logits_to_keep), vision
tower frozen, gradient checkpointing on the LLM. ~10-11 GB on one A4000 at 4.6 k tokens.

  # smoke (CPU, bf16, tiny frames):  --mode bf16 --device cpu --limit 2 --size 112 --max_frames 4
  CUDA_VISIBLE_DEVICES=2 python scripts/vlm_hier/train.py --dataset thumos --data data/vlm_hier/thumos_s0 --out ckpt/vlm_hier/thumos_s0
  CUDA_VISIBLE_DEVICES=3 python scripts/vlm_hier/train.py --dataset fd     --data data/vlm_hier/fd_s1     --out ckpt/vlm_hier/fd_s1
"""
import argparse, json, math, os, random, re, sys, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # less fragmentation at 5-7k tokens
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


class Prompter:
    """Dataset-specific prompt + answer for a sample; training prompts list a random subset of SEEN classes."""
    def __init__(self, dataset, data_dir, max_classes=8, seed=0, prompt_list='subset'):
        self.dataset = dataset; self.meta = json.load(open(os.path.join(data_dir, 'meta.json'))); self.max_classes = max_classes
        self.prompt_list = prompt_list      # 'subset': positives + random seen fill (<= max_classes); 'all': the full fixed list = the test prompt (11-09, FineDiving fix)
        self.rng = random.Random(seed)
        if dataset == 'thumos':
            self.trees = json.load(open('configs/thumos14_qwen_ensemble.json'))
            self.seen, self.unseen = self.meta['seen'], self.meta['unseen']
        else:
            cls = self.meta['classes']
            self.names = {c['id']: c['name'] for c in cls}; self.groups = {c['id']: c['group'] for c in cls}
            self.seen = [c['id'] for c in cls if not c['held_out']]; self.all_ids = [c['id'] for c in cls]
            self.sub_names = self.meta['sub_actions']; self.sub_roots = self.meta['sub_roots']

    def train_prompt(self, s):
        if self.prompt_list == 'all':           # same class list and order as test_prompt -> no train/test prompt shift
            return self.test_prompt(s)
        pos = list(s['pos_classes'])
        fill = [c for c in self.seen if c not in pos]; self.rng.shuffle(fill)
        classes = pos + fill[:max(0, self.max_classes - len(pos))]; self.rng.shuffle(classes)
        return self._prompt(classes, s)

    def test_prompt(self, s):
        return self._prompt(self.unseen if self.dataset == 'thumos' else self.all_ids, s)

    def _prompt(self, classes, s):
        if self.dataset == 'thumos':
            return C.prompt_thumos(classes, self.trees, s['win_len'], s['fps'])
        return C.prompt_finediving(classes, self.names, self.groups, self.sub_names, self.sub_roots, s['win_len'], s['fps'])

    def answer(self, s):
        """Returns (answer_text, masked_char_spans): spans of class-name values whose tokens must carry no loss
        (FineDiving keep protocol: held-out dives keep span + group, the class is not supervised)."""
        keep = ('group', 'action', 'start', 'end', 'sub') if self.dataset == 'fd' else ('action', 'start', 'end', 'cuts')
        text = C.target_json([{k: t[k] for k in keep if k in t} for t in s['target']])
        spans = []
        if any(t.get('mask_name') for t in s['target']):
            import re
            # top-level objects appear in order; the i-th top-level "action" value is the i-th instance's name
            # (sub-action "action" keys live inside "sub":[...] after their parent's name, so count by instance)
            pos = 0
            for t in s['target']:
                m = re.compile(r'"action":"((?:[^"\\]|\\.)*)"').search(text, pos)
                assert m and m.group(1) == C.target_json(t['action'])[1:-1], (m.group(1) if m else None, t['action'])
                if t.get('mask_name'):
                    spans.append((m.start(1), m.end(1)))
                pos = m.end()
                for q in t.get('sub', []) or []:                         # skip the children's "action" values
                    m2 = re.compile(r'"action":"((?:[^"\\]|\\.)*)"').search(text, pos); pos = m2.end()
        return text, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['thumos', 'fd'], required=True)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--accum', type=int, default=8); ap.add_argument('--warmup', type=float, default=0.03)
    ap.add_argument('--r', type=int, default=32); ap.add_argument('--alpha', type=int, default=64)
    ap.add_argument('--max_classes', type=int, default=0, help='classes listed in a training prompt (0 = 6 for THUMOS, 8 for FineDiving)')
    ap.add_argument('--mode', default='4bit', choices=['4bit', 'bf16', 'fp32']); ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--size', type=int, default=0, help='resize frames (smoke)')
    ap.add_argument('--max_frames', type=int, default=0, help='truncate frames (smoke)')
    ap.add_argument('--log_every', type=int, default=10); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume_from', default='', help='adapter dir saved by a previous run (…/epochN): load its LoRA weights and continue from epoch N+1 (optimizer moments are not restored)')
    ap.add_argument('--max_tokens', type=int, default=7000, help='skip training windows longer than this (memory guard)')
    ap.add_argument('--train_file', default='', help='override: train.jsonl (oracle / flat) or train_nested.jsonl (self-labelled)')
    ap.add_argument('--prompt_list', default='subset', choices=['subset', 'all'], help="classes listed in training prompts: 'subset' (positives + random seen fill) or 'all' (the full fixed list, identical to the test prompt; FineDiving fix 11-09)")
    a = ap.parse_args()
    if not a.max_classes:
        a.max_classes = 6 if a.dataset == 'thumos' else 8
    if not a.size:
        a.size = 280 if a.dataset == 'thumos' else 336   # THUMOS: 280 px (100 tokens / frame pair) keeps dense 64 s windows under ~6.5k tokens; FineDiving 336 (small diver)
    import torch
    torch.manual_seed(a.seed); random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    train_file = a.train_file or os.path.join(a.data, 'train_nested.jsonl')   # ground.py output (self-labelled children); oracle: train.jsonl
    if not os.path.exists(train_file):
        train_file = os.path.join(a.data, 'train.jsonl')
    samples = [json.loads(l) for l in open(train_file)]
    print('training file', train_file, flush=True)
    if a.dataset == 'thumos' and os.environ.get('VLM_HIER_LENIENT', '') == '1':   # off by default: the recovered cuts were degenerate counting sequences (11-08)
        import collections
        K = len(next(iter(P.trees.values()))) - 1; cnt = collections.Counter()
        for s_ in samples:
            C.recover_children_thumos(s_, K)
            for t in s_['target']:
                if 'cuts' in t: cnt[t.get('child_src', '?')] += 1
        print('child sources after lenient recovery:', dict(cnt), flush=True)
    assert all(('start' in q and 'end' in q) for s_ in samples for t in s_['target'] for q in (t.get('sub') or [])), 'children without spans: run ground.py first (tree setting) or use the oracle file'
    if a.limit:
        samples = samples[:a.limit]
    P = Prompter(a.dataset, a.data, a.max_classes, a.seed, a.prompt_list)
    print('training prompt class list:', a.prompt_list, flush=True)
    proc, model = C.load_model(mode=a.mode, lora=True, r=a.r, alpha=a.alpha, device=a.device)
    tok = proc.tokenizer
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'trainable params {sum(p.numel() for p in params)/1e6:.1f}M | samples {len(samples)} | epochs {a.epochs} | accum {a.accum}', flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    total_updates = max(1, math.ceil(len(samples) * a.epochs / a.accum)); warm = max(1, int(a.warmup * total_updates))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, u - warm) / max(1, total_updates - warm)))))
    dev = model.device if a.device == 'cuda' else 'cpu'
    log = open(os.path.join(a.out, 'train_log.jsonl'), 'a')
    step = upd = 0; t0 = time.time(); run_loss = []; checked = False; n_skip = 0
    start_ep = 0
    if a.resume_from:                                          # reboot recovery (2026-11-09): continue from a saved epoch adapter
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd_ = load_file(os.path.join(a.resume_from, 'adapter_model.safetensors'))
        res_ = set_peft_model_state_dict(model, sd_)
        done_ep = int(re.search(r'epoch(\d+)$', a.resume_from.rstrip('/')).group(1))
        start_ep = done_ep + 1; step = start_ep * len(samples); upd = math.ceil(step / a.accum)
        for _ in range(upd):
            sched.step()                                        # cosine schedule positioned as if those updates had happened
        print(f'[resume] loaded {a.resume_from} ({len(sd_)} tensors, unexpected {len(res_.unexpected_keys)}); continuing at epoch {start_ep}, update {upd}, lr {sched.get_last_lr()[0]:.2e}', flush=True)
    for ep in range(start_ep, a.epochs):
        order = list(range(len(samples))); random.Random(a.seed + ep).shuffle(order)
        for i in order:
            s = samples[i]
            frames = C.load_frames(s['frames'][:a.max_frames] if a.max_frames else s['frames'], size=a.size or None)
            prompt = P.train_prompt(s); answer, mask_spans = P.answer(s)
            full = C.chat_text(proc, prompt, answer)
            inp = C.video_inputs(proc, full, frames, s['fps'], dev)
            ids = inp['input_ids']
            if ids.shape[1] > a.max_tokens:                       # rare very dense windows: skip rather than OOM
                n_skip += 1; step += 0
                if n_skip <= 5 or n_skip % 20 == 0: print(f'[skip] {s["id"]}: {ids.shape[1]} tokens > {a.max_tokens} ({n_skip} skipped so far)', flush=True)
                continue
            tail = answer + '<|im_end|>\n'
            tail_ids = tok.encode(tail, add_special_tokens=False); k = len(tail_ids)
            labels = ids[0, -k:].clone()
            if mask_spans:                                    # drop the loss on masked class-name tokens
                offs, _ = C.token_char_offsets(tok, tail_ids)
                for t in range(k):
                    c0, c1 = (offs[t - 1] if t else 0), offs[t]
                    if any(c0 < b and c1 > a_ for a_, b in mask_spans):
                        labels[t] = -100
            if not checked:                                   # the answer must be exactly the last k tokens
                assert tok.decode(ids[0, -k:]) == tail, (tok.decode(ids[0, -k:]), tail)
                print(f'[check] seq {ids.shape[1]} tokens, answer {k} tokens ({int((labels == -100).sum())} masked), video grid {inp["video_grid_thw"].tolist()}', flush=True)
                checked = True
            with torch.autocast(device_type='cuda' if a.device == 'cuda' else 'cpu', dtype=torch.bfloat16, enabled=(a.mode != 'fp32')):
                out = model(**inp, logits_to_keep=k + 1)
                logits = out.logits[:, :-1].float()                # predicts ids[:, -k:]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            (loss / a.accum).backward()
            run_loss.append(loss.item()); step += 1
            if step % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); upd += 1
                if upd % a.log_every == 0 or upd == 1:
                    rec = {'epoch': ep, 'update': upd, 'step': step, 'loss': float(np.mean(run_loss[-a.accum * a.log_every:])), 'lr': sched.get_last_lr()[0],
                           'min': (time.time() - t0) / 60, 'mem_gb': torch.cuda.max_memory_allocated() / 2**30 if a.device == 'cuda' else 0}
                    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if step % a.accum:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); upd += 1
        model.save_pretrained(os.path.join(a.out, f'epoch{ep}'))
        print(f'epoch {ep} done: mean loss {np.mean(run_loss[-len(samples):]):.4f}, {(time.time()-t0)/60:.1f} min, saved {a.out}/epoch{ep}', flush=True)
    model.save_pretrained(os.path.join(a.out, 'final'))
    print('done', flush=True)


if __name__ == '__main__':
    main()
