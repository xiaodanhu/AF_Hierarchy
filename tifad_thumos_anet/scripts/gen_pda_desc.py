#!/usr/bin/env python
"""Generate PDA (CVPR'26, arXiv 2603.24030) CSD phase descriptions with a
LOCAL LLM (Qwen2.5-VL-7B-Instruct used as a text-only causal LM; the paper
uses GPT-4o, and reports Qwen3 within 0.7 mAP of GPT-4o in its Table 4/11).

Prompts are the paper's Table 8 (c) templates, verbatim:
  phase : "Decompose the action of <Action> into coherent three phases based
           on the natural temporal progression of the action. Please provide
           the output step by step."
          answer format "In the start phase, the person would {text}. In the
           middle phase, the person would {text}. In the end phase, the person
           would {text}."
  global: "Describe how a person does <Action>."  answer "The person would {text}."

Output json: {class_name: {"start": ..., "middle": ..., "end": ..., "glob": ...}}
with the phrases stored WITHOUT the "The person would" lead-in (the model side
wraps each as "a video of people's motion that the person would {text}").

Run (needs transformers >= 4.49 for Qwen2.5-VL; the `hierarchy` env has 4.56):
  CUDA_VISIBLE_DEVICES=2 /data/xiaodan8/anaconda3/envs/hierarchy/bin/python \
      scripts/gen_pda_desc.py --out configs/thumos14_pda_desc.json
"""
import argparse
import json
import os
import re
import sys

PHASE_Q = ("Decompose the action of {action} into coherent three phases based on "
           "the natural temporal progression of the action. Please provide the "
           "output step by step.")
PHASE_A = ("In the start phase, the person would {text}. In the middle phase, "
           "the person would {text}. In the end phase, the person would {text}.")
GLOB_Q = "Describe how a person does {action}."
GLOB_A = "The person would {text}."

# minimal harness around the paper's template: the template's "Answer:" line
# is handed to the model as the required final-answer format.
SYSTEM = ("You are a helpful assistant. Think step by step when asked to, then "
          "finish with a final answer that follows the requested format "
          "exactly, filling in each {text} with a concise phrase.")

PHASE_RE = re.compile(
    r"In the start phase,\s*the person would\s*(.+?)\.\s*"
    r"In the middle phase,\s*the person would\s*(.+?)\.\s*"
    r"In the end phase,\s*the person would\s*(.+?)\.", re.S | re.I)
GLOB_RE = re.compile(r"The person would\s*(.+?)\.(?:\s|$)", re.S | re.I)

THUMOS_CLASSES = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                  'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
                  'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                  'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                  'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
                  'VolleyballSpiking']


def readable(name):
    # CamelCase class label -> words ("BaseballPitch" -> "Baseball Pitch")
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)


def clean(s):
    s = ' '.join(s.split())
    s = s.strip().strip('*').strip()
    s = re.sub(r'^\{?text\}?\s*[:=]\s*', '', s)
    return s.rstrip('.').strip()


def load_model(model_dir, device):
    import torch
    from transformers import AutoTokenizer
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    except ImportError:
        from transformers import AutoModelForCausalLM as Cls
    tok = AutoTokenizer.from_pretrained(model_dir)
    kw = {}
    if device == 'cpu':
        kw = dict(torch_dtype=torch.bfloat16, device_map='cpu')
    elif device == '8bit':
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                  device_map={'': 0})
    else:
        kw = dict(torch_dtype=torch.bfloat16, device_map={'': 0})
    model = Cls.from_pretrained(model_dir, **kw)
    model.eval()
    return tok, model


def chat(tok, model, user, max_new_tokens=400, sample=False, seed=0):
    import torch
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    inp = tok([text], return_tensors='pt').to(model.device)
    gen = dict(max_new_tokens=max_new_tokens, do_sample=sample)
    if sample:
        torch.manual_seed(seed)
        gen.update(temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inp, **gen)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def gen_phases(tok, model, action, tries=6):
    user = (f"Question: {PHASE_Q.format(action=action)}\n"
            f"Answer: {PHASE_A}")
    raw_all = []
    for t in range(tries):
        raw = chat(tok, model, user, sample=(t > 0), seed=t)
        raw_all.append(raw)
        m = list(PHASE_RE.finditer(raw))
        if m:
            g = m[-1].groups()          # last match = the final answer
            vals = [clean(x) for x in g]
            if all(vals) and all('{' not in v for v in vals):
                return dict(start=vals[0], middle=vals[1], end=vals[2]), raw
    raise RuntimeError(f"could not parse phases for {action}:\n" +
                       "\n----\n".join(raw_all))


def gen_glob(tok, model, action, tries=6):
    user = (f"Question: {GLOB_Q.format(action=action)}\n"
            f"Answer: {GLOB_A}")
    raw_all = []
    for t in range(tries):
        raw = chat(tok, model, user, max_new_tokens=200, sample=(t > 0), seed=t)
        raw_all.append(raw)
        m = list(GLOB_RE.finditer(raw))
        if m:
            v = clean(m[-1].group(1))
            if v and '{' not in v:
                return v, raw
    raise RuntimeError(f"could not parse global for {action}:\n" +
                       "\n----\n".join(raw_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/data3/xiaodan8/.cache/models--Qwen--'
                    'Qwen2.5-VL-7B-Instruct/snapshots')
    ap.add_argument('--out', default='configs/thumos14_pda_desc.json')
    ap.add_argument('--raw', default='configs/thumos14_pda_desc_raw.json',
                    help='full LLM outputs (CoT) for the record')
    ap.add_argument('--device', default='8bit', choices=['8bit', 'bf16', 'cpu'])
    ap.add_argument('--classes', default='', help='json list file; default THUMOS14')
    args = ap.parse_args()

    model_dir = args.model
    if os.path.basename(model_dir) == 'snapshots':
        model_dir = os.path.join(model_dir, sorted(os.listdir(model_dir))[0])
    classes = THUMOS_CLASSES
    if args.classes:
        classes = json.load(open(args.classes))

    tok, model = load_model(model_dir, args.device)
    out, raw = {}, {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    if os.path.exists(args.raw):
        raw = json.load(open(args.raw))
    for c in classes:
        if c in out and all(k in out[c] for k in ('start', 'middle', 'end', 'glob')):
            continue
        action = readable(c)
        ph, r1 = gen_phases(tok, model, action)
        gl, r2 = gen_glob(tok, model, action)
        out[c] = dict(start=ph['start'], middle=ph['middle'], end=ph['end'],
                      glob=gl)
        raw[c] = dict(phases=r1, glob=r2)
        print(f"[{c}] start: {ph['start']}\n     middle: {ph['middle']}\n"
              f"     end: {ph['end']}\n     glob: {gl}", flush=True)
        json.dump(out, open(args.out, 'w'), indent=2)
        json.dump(raw, open(args.raw, 'w'), indent=2)
    print(f"wrote {len(out)} classes -> {args.out}")


if __name__ == '__main__':
    sys.exit(main())
