"""C3 LoRA-tuned Qwen2.5-VL that emits the hierarchical tree of a clip.

Shared pieces: model loading (4-bit NF4 base + LoRA on the LLM only), prompt
construction, compact JSON targets, tolerant JSON parsing, and token-log-prob
confidence for every emitted instance.

Environment: `hierarchy` (torch 2.8, transformers 4.56, peft 0.17, bitsandbytes 0.47).
"""
import json, math, os, re, random
import numpy as np

MODEL = '/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
SYSTEM = ("You are a precise video action localizer. You watch a clip and report every action instance "
          "as JSON. Times are seconds from the start of the clip. Reply with JSON only.")


def readable(name):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).replace('_', ' ').strip().lower()


# ----------------------------------------------------------------------------- prompts
def prompt_thumos(classes, trees, win_len, fps):
    """classes: THUMOS class keys (CamelCase); trees: {key: [root, sub_1..sub_K]}."""
    lines = [f"The clip is {win_len:.0f} seconds long, shown at {fps:g} frame(s) per second.",
             "It may contain zero, one or several instances of these actions:"]
    for c in classes:
        root, subs = trees[c][0], trees[c][1:]
        lines.append(f'- "{readable(c)}": {root} Steps: ' + ' '.join(f'({k+1}) {s}' for k, s in enumerate(subs)))
    lines.append('Output a JSON list, one object per instance, sorted by start time, each with its steps placed in time as K+1 cut times: '
                 '[{"action": <name exactly as listed>, "start": <seconds>, "end": <seconds>, "cuts": [t0, t1, ..., tK]}] '
                 'where step k runs from t(k-1) to tk, t0 = start, tK = end, and the cuts are non-decreasing. '
                 'Output [] if none of the listed actions occurs.')
    return '\n'.join(lines)


def prompt_thumos_ground(instances, trees, win_len, fps):
    """Self-labelling pass (no human sub-action labels): the GT actions of a training window are given; the model
    only places each action's K steps. instances: [{'action': readable name, 'cls': key, 'start', 'end'}]."""
    keys = []
    for t in instances:
        if t['cls'] not in keys:
            keys.append(t['cls'])
    K = len(trees[keys[0]]) - 1
    lines = [f"The clip is {win_len:.0f} seconds long, shown at {fps:g} frame(s) per second.",
             "It contains these action instances (seconds from the clip start):"]
    for t in instances:
        lines.append(f'- "{t["action"]}" from {t["start"]} to {t["end"]}')
    lines.append(f"Each action unfolds in these {K} ordered steps:")
    for c in keys:
        lines.append(f'- "{readable(c)}": ' + ' '.join(f'({k+1}) {s}' for k, s in enumerate(trees[c][1:])))
    lines.append('For every instance, watch the clip and place its steps in time as cut times. Output a JSON list with one object per '
                 'instance, in the given order, copying the given action, start and end: [{"action": ..., "start": ..., "end": ..., '
                 f'"cuts": [t0, t1, ..., t{K}]}}] where step k runs from t(k-1) to tk, t0 = start, t{K} = end, cuts non-decreasing. '
                 'Give each step the span where it is actually seen; steps can be short or long. Write the JSON compactly on one line, no spaces or line breaks.')
    return '\n'.join(lines)


def prompt_fd_ground(instances, sub_roots, win_len, fps):
    """Self-labelling pass for FineDiving (tree setting): the GT dive (name, span) and its ORDERED sub-actions from the
    definition tree are given; the model places each sub-action in time. instances: [{'action', 'start', 'end', 'members': [names]}]."""
    lines = [f"The clip is {win_len:.1f} seconds long, shown at {fps:g} frames per second, from a diving competition.",
             "It contains these dives (seconds from the clip start), each a sequence of sub-actions in this order:"]
    for t in instances:
        lines.append(f'- "{t["action"]}" from {t["start"]} to {t["end"]}: ' + ' -> '.join(f'"{m}"' for m in t['members']))
    names = []
    for t in instances:
        for m in t['members']:
            if m not in names:
                names.append(m)
    lines.append("Sub-action meanings:")
    for m in names:
        lines.append(f'- "{m}": {sub_roots.get(m, "")}')
    lines.append('Watch the clip and place every sub-action in time. Output a JSON list with one object per dive, in the given order, '
                 'copying the given action, start and end: [{"action": ..., "start": ..., "end": ..., '
                 '"sub": [{"action": <sub-action name>, "start": <seconds>, "end": <seconds>}, ...]}] with the sub-actions in the given '
                 'order, contiguous, covering the dive from its start to its end; give each one the span where it is actually seen. Write the JSON compactly on one line, no spaces or line breaks.')
    return '\n'.join(lines)


def prompt_finediving(dive_keys, dive_names, groups, sub_names, sub_roots, win_len, fps):
    """dive_keys: class ids in prompt order; dive_names[id] readable name; groups[id] group name;
    sub_names/sub_roots: ordered sub-action vocabulary (name -> one-sentence description)."""
    lines = [f"The clip is {win_len:.1f} seconds long, shown at {fps:g} frames per second, from a diving competition.",
             "It may contain one dive (or none). Dives are these classes, each written as group | dive name:"]
    for k in dive_keys:
        lines.append(f'- {groups[k]} | "{dive_names[k]}"')
    lines.append("A dive is a sequence of sub-actions from this vocabulary:")
    for n in sub_names:
        lines.append(f'- "{n}": {sub_roots[n]}')
    lines.append('Output a JSON list, one object per dive: [{"group": <group>, "action": <dive name exactly as listed>, '
                 '"start": <seconds>, "end": <seconds>, "sub": [{"action": <sub-action name>, "start": <seconds>, "end": <seconds>}, ...]}] '
                 'with the sub-actions in temporal order covering the dive. Output [] if there is no dive.')
    return '\n'.join(lines)


def target_json(instances):
    """Compact JSON text for a list of instance dicts (already window-relative, rounded)."""
    return json.dumps(instances, separators=(',', ':'), ensure_ascii=False)


def r1(x):
    return float(f'{x:.1f}')


# ----------------------------------------------------------------------------- parsing
def _repair(text):
    """Best-effort: take the first '[' ... matching ']' block; close dangling structures."""
    i = text.find('[')
    if i < 0:
        return '[]'
    t = text[i:]
    j = t.rfind(']')
    if j > 0:
        try:
            json.loads(t[:j + 1]); return t[:j + 1]
        except Exception:
            pass
    # truncated generation: cut at the last complete object
    k = t.rfind('}')
    if k > 0:
        cand = t[:k + 1]
        opens = cand.count('[') - cand.count(']')
        braces = cand.count('{') - cand.count('}')
        cand += '}' * max(0, braces) + ']' * max(0, opens)
        try:
            json.loads(cand); return cand
        except Exception:
            pass
    return '[]'


def parse_instances(text, win_len, nested=False):
    """Returns list of dicts {action, start, end[, group][, sub:[...]]}, clipped to [0, win_len], invalid dropped."""
    try:
        arr = json.loads(_repair(text))
    except Exception:
        return []
    if isinstance(arr, dict):
        arr = [arr]
    out = []
    for o in arr if isinstance(arr, list) else []:
        if not isinstance(o, dict) or 'action' not in o:
            continue
        try:
            s, e = float(o.get('start', 0)), float(o.get('end', 0))
        except (TypeError, ValueError):
            continue
        s, e = max(0.0, min(s, win_len)), max(0.0, min(e, win_len))
        if e <= s:
            continue
        inst = {'action': str(o['action']).strip().lower(), 'start': s, 'end': e}
        if 'group' in o:
            inst['group'] = str(o['group']).strip().lower()
        if isinstance(o.get('cuts'), list):
            try:
                inst['cuts'] = [max(0.0, min(float(x), win_len)) for x in o['cuts']]
            except (TypeError, ValueError):
                pass
        if nested:
            subs = []
            for q in o.get('sub', []) or []:
                if not isinstance(q, dict) or ('action' not in q and 'step' not in q):
                    continue
                try:
                    a, b = float(q.get('start', 0)), float(q.get('end', 0))
                except (TypeError, ValueError):
                    continue
                a, b = max(0.0, min(a, win_len)), max(0.0, min(b, win_len))
                if b > a:
                    sub = {'start': a, 'end': b}
                    if 'action' in q:
                        sub['action'] = str(q['action']).strip().lower()
                    if 'step' in q:
                        try:
                            sub['step'] = int(q['step'])
                        except (TypeError, ValueError):
                            pass
                    subs.append(sub)
            inst['sub'] = subs
        out.append(inst)
    return out


# ----------------------------------------------------------------------------- confidence from token log-probs
def token_char_offsets(tok, ids):
    """Cumulative decoded-character offset of every generated token (byte-level BPE safe)."""
    offs, text = [], ''
    for i in range(len(ids)):
        text = tok.decode(ids[:i + 1], skip_special_tokens=True)
        offs.append(len(text))
    return offs, text


def instance_confidences(text, offsets, logps, instances):
    """conf of instance i = exp(mean log-prob of the tokens covering the i-th '"action":"..."' value).
    offsets[t] = decoded length after token t; logps[t] = log-prob of token t."""
    spans = [(m.start(1), m.end(1)) for m in re.finditer(r'"action"\s*:\s*"([^"]*)"', text)]
    confs = []
    for i, inst in enumerate(instances):
        if i >= len(spans):
            confs.append(0.5); continue
        a, b = spans[i]
        toks = [t for t in range(len(offsets)) if (offsets[t - 1] if t else 0) < b and offsets[t] > a]
        confs.append(float(math.exp(np.mean([logps[t] for t in toks]))) if toks else 0.5)
    return confs


# ----------------------------------------------------------------------------- model
def load_model(model_dir=MODEL, mode='4bit', lora=True, adapter=None, r=32, alpha=64, dropout=0.05, device='cuda'):
    """mode: '4bit' (QLoRA, GPU), 'bf16' (CPU smoke or inference on a big card). lora=False -> frozen base."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    if device == 'cpu':
        torch.set_num_threads(max(8, os.cpu_count() // 2))
    proc = AutoProcessor.from_pretrained(model_dir)
    if mode == '4bit':
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True, llm_int8_skip_modules=['visual', 'lm_head'])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, quantization_config=q, torch_dtype=torch.bfloat16,
                                                                   device_map={'': 0} if device == 'cuda' else 'cpu')
    else:                                                   # 'bf16' or 'fp32' (CPU smoke)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32 if mode == 'fp32' else torch.bfloat16, device_map=device)
    for p in model.parameters():
        p.requires_grad_(False)
    if lora:
        from peft import LoraConfig, get_peft_model, PeftModel
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        else:
            targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) or m.__class__.__name__ in ('Linear4bit',)
                       if 'language_model' in n and n.split('.')[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')]
            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias='none', task_type='CAUSAL_LM')
            model = get_peft_model(model, cfg)
    return proc, model


def video_inputs(proc, text, frames, fps, device):
    """frames: list of HxWx3 uint8 (same size). Returns processor inputs on device."""
    video = np.stack(frames)
    try:
        inp = proc(text=[text], videos=[video], return_tensors='pt', fps=float(fps))
    except TypeError:
        inp = proc(text=[text], videos=[video], return_tensors='pt')
    return inp.to(device)


def chat_text(proc, prompt, answer=None):
    """Chat-template text. With answer: full supervised sequence; without: generation prompt."""
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt}]}]
    if answer is None:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    msgs.append({'role': 'assistant', 'content': answer})
    return proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)


def load_frames(paths, size=None):
    from PIL import Image
    out = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        if size and im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        out.append(np.asarray(im))
    return out


def equal_cuts(start, end, K):
    return [r1(start + k * (end - start) / K) for k in range(K + 1)]


def valid_cuts(cuts, start, end, K, tol=0.5):
    """K+1 non-decreasing cut times, first ~ start, last ~ end (tol seconds)."""
    if not isinstance(cuts, list) or len(cuts) != K + 1:
        return False
    try:
        c = [float(x) for x in cuts]
    except (TypeError, ValueError):
        return False
    if not (abs(c[0] - start) <= tol and abs(c[-1] - end) <= tol and c[-1] > c[0]):
        return False
    mn = 0.02 * (end - start)
    return all(c[i + 1] - c[i] >= mn - 1e-6 for i in range(K))      # non-decreasing AND no collapsed step


def cuts_to_spans(cuts):
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


def prior_cuts(start, end, fractions):
    """Cut times from a step-duration prior (fractions sum to 1)."""
    f = [max(1e-3, float(x)) for x in fractions]; z = sum(f); L = end - start
    out = [start]; acc = 0.0
    for x in f:
        acc += x / z; out.append(start + acc * L)
    out[-1] = end
    return [r1(x) for x in out]


def valid_member_spans(subs, members, start, end, tol=0.3):
    """One span per member, same names in order, non-decreasing, contiguous within tol, inside the parent."""
    if len(subs) != len(members):
        return False
    if any(q.get('action', '').strip().lower() != m.lower() for q, m in zip(subs, members)):
        return False
    if subs[0]['start'] < start - tol or subs[-1]['end'] > end + tol:
        return False
    for i in range(1, len(subs)):
        if subs[i]['start'] < subs[i - 1]['start'] or abs(subs[i]['start'] - subs[i - 1]['end']) > tol:
            return False
    mn = 0.02 * (end - start)
    return all(q['end'] - q['start'] >= mn for q in subs)          # no collapsed sub-action


# ----------------------------------------------------------------------------- lenient recovery of THUMOS cuts
_CUTS_RE = re.compile(r'"cuts"\s*:\s*\[([^\]]*)\]')


def lenient_cuts(text, action, start, end, K, tol=0.5):
    """Recover K+1 cuts for the instance (action, start, end) from a raw model output that the strict
    path rejected. Accepts: K+1 numbers; K numbers (model dropped the start or the end -> completed);
    K-1 interior numbers; malformed lists like ["t0": 0.9, "t1": 2.1, ...] (numbers extracted by regex).
    Returns a valid cut list (valid_cuts) or None."""
    for m in re.finditer(r'\{[^{}]*\}', text):                       # one object per instance
        blob = m.group(0)
        if f'"{action}"' not in blob.lower():
            continue
        try:
            st = float(re.search(r'"start"\s*:\s*([0-9.]+)', blob).group(1)); en = float(re.search(r'"end"\s*:\s*([0-9.]+)', blob).group(1))
        except Exception:
            continue
        if abs(st - start) > 2.0 or abs(en - end) > 2.0:
            continue
        mc = _CUTS_RE.search(blob)
        if not mc:
            continue
        nums = [float(x) for x in re.findall(r'(?<![\w."])-?[0-9]+(?:\.[0-9]+)?(?![\w"])', mc.group(1))]
        nums = [x for x in nums if not (x.is_integer() and 0 <= x <= K and f'"t{int(x)}"' in mc.group(1))]   # drop "t3"-style labels' digits
        for cand in (nums, [start] + nums, nums + [end], [start] + nums + [end]):
            if len(cand) == K + 1:
                cand = [r1(x) for x in cand]
                if valid_cuts(cand, start, end, K, tol):
                    return cand
    return None


def recover_children_thumos(sample, K, prior=None):
    """Re-derive cuts for every prior-fallback instance of a grounded THUMOS window from its saved raw
    outputs (greedy first, then samples). Mutates and returns the sample; counts in sample['recovered']."""
    raws = sample.get('ground_raw') or []
    n_rec = 0
    for t in sample['target']:
        if t.get('child_src') != 'prior' or not raws:
            continue
        for text in raws:
            c = lenient_cuts(text, t['action'], t['start'], t['end'], K)
            if c:
                t['cuts'] = c; t['child_src'] = 'vlm_lenient'; n_rec += 1; break
    sample['recovered'] = n_rec
    return sample
