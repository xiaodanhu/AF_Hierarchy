#!/usr/bin/env python
"""Five timing-aware phase sentences for FineDiving (the entry occupies
more than a third of a dive, which neither the ten generated phases nor a pick of five reflects).
Same generator as the THUMOS/ActivityNet descriptions (Qwen2.5-VL-7B-Instruct, text only),
run on CPU. Writes NEW files next to the training assets
(fd_dive_ensemble_5.json, fd_sub_ensemble_5.json); the training assets are not modified.
Annotated shares (FineDiving fine-grained csv, 3000 dives): takeoff 0.29, somersaults/twists 0.35,
entry 0.36 for three-sub-action dives; four/five-sub-action dives: entry 0.37 / 0.28."""
import json, re, torch, os, sys, time
torch.set_num_threads(int(os.environ.get('NT', '32')))
from transformers import AutoModelForImageTextToText, AutoProcessor
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"; N = 5; D = '/data3/xiaodan8/actionformer6/configs/finediving'
only = sys.argv[1] if len(sys.argv) > 1 else ''          # 'dive' | 'sub' | '' (both); SMOKE=1 -> first item only
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32, device_map='cpu').eval()
def ask(prompt):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    with torch.no_grad(): gen = model.generate(**inputs, max_new_tokens=320, do_sample=False)
    txt = processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    sents = [re.sub(r'^\s*\d+[.)]\s*', '', l).strip() for l in txt.split('\n')]
    sents = [s for s in sents if len(s) > 15 and not s.lower().startswith(('here', 'sure', 'phase list'))][:N]
    while len(sents) < N: sents.append(sents[-1] if sents else 'the diver moves through the air above the pool')
    return sents
RULES = ("Each phase must be one sentence describing only what is VISIBLE (body position, rotation, twist, "
         "arm and leg configuration, motion relative to the board and the water). Do not mention the dive code. "
         f"Number them 1-{N}.")
dv = json.load(open(f'{D}/fd_dive_verbalizer.json')); sv = json.load(open(f'{D}/fd_sub_verbalizer.json'))
act_name = lambda short: next(v['name'] for v in dv['activities'].values() if v['short'] == short)
t0 = time.time(); smoke = os.environ.get('SMOKE') == '1'
if only in ('', 'dive'):
    out = {}
    for cid, a in dv['actions'].items():
        key = f"({a['activity']}) {a['name']}"
        ctx = (f"Springboard/platform diving. The dive '{a['name']}' consists of these sub-actions in order: "
               + ' -> '.join(a['member_names']) + '.')
        timing = ("Timing seen in annotated dives: the takeoff (stance, arm swing or press, leaving the board or platform) "
                  "takes about the first 30% of the dive, the somersaults and twists in the air about the middle 35%, and the entry "
                  "about the last 35%: it starts when the diver opens out of the pike or tuck, stretches into a straight vertical "
                  "line with the arms extended toward the water, and ends when the body has disappeared under the splash. "
                  f"Divide the dive into {N} equal time intervals and describe what is visible in each interval, so that the last "
                  "two phases belong to the entry.")
        p = f"Context: {ctx}\n{timing}\nList exactly {N} short visual phases of this dive, in temporal order. {RULES}"
        out[key] = [f"a video of a {a['name']}, a {act_name(a['activity'])} in springboard or platform diving"] + ask(p)
        print(f"[dive {cid}] {key} ({time.time()-t0:.0f}s)", flush=True)
        for s in out[key][1:]: print('   -', s, flush=True)
        if smoke: break
    json.dump(out, open(f'{D}/fd_dive_ensemble_5{"_smoke" if smoke else ""}.json', 'w'), indent=1)
if only in ('', 'sub'):
    out = {}
    SPAN = {'takeoff': "the takeoff spans from the diver's stance on the board or platform, through the arm swing or press, "
                       "to the moment the feet (or hands, for an armstand) leave it, about the first 30% of the dive",
            'somersault_pike': "the somersault spans only the rotation in the air in pike position (legs straight, body folded at the hips), from closing into the pike to opening out of it",
            'somersault_tuck': "the somersault spans only the rotation in the air in tuck position (knees pulled to the chest), from closing into the tuck to opening out of it",
            'twist': "the twist spans only the twisting rotation about the body's long axis in the air, from the arm wrap that starts it to the arm opening that stops it",
            'entry': "the entry spans from the moment the diver opens out of the last somersault or twist, through stretching into a straight vertical line with arms extended toward the water and hitting the water hands first, to the body disappearing under the splash, about the last 35% of the dive"}
    for cid, a in sv['actions'].items():
        key = f"({a['activity']}) {a['name']}"; ph = sv['phrases'][a['phrase']]['name']; fam = ph.replace('_', ' ')
        p = (f"Context: springboard/platform diving; the sub-action '{a['name']}' is a {fam} phase of a dive. In the annotated videos {SPAN[ph]}.\n"
             f"Divide this sub-action alone into {N} equal time intervals and describe what is visible in each, in temporal order. {RULES}")
        out[key] = [f"a video of the diving sub-action {a['name']}, a {fam} phase"] + ask(p)
        print(f"[sub {cid}] {key} ({time.time()-t0:.0f}s)", flush=True)
        for s in out[key][1:]: print('   -', s, flush=True)
        if smoke: break
    json.dump(out, open(f'{D}/fd_sub_ensemble_5{"_smoke" if smoke else ""}.json', 'w'), indent=1)
print('DONE', f'{time.time()-t0:.0f}s')
