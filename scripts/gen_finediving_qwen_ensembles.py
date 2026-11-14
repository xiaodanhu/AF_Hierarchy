#!/usr/bin/env python
"""Qwen3-VL-4B-Instruct (local, offline) phase descriptions for the FineDiving two-level hierarchy.
Same pattern as tifad/gen_qwen_descriptions_25.py: N=10 visible phases in temporal order; sentence 0 = hierarchy prompt.
Outputs configs/finediving/fd_dive_ensemble.json (52 keys "(GROUP) name", 11 sentences) and fd_sub_ensemble.json (28 keys)."""
import json, re, torch, os, sys, time
from transformers import AutoModelForImageTextToText, AutoProcessor
MODEL="Qwen/Qwen3-VL-4B-Instruct"; N=10; OUT='configs/finediving'
processor=AutoProcessor.from_pretrained(MODEL)
model=AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
def ask(prompt):
    msgs=[{"role":"user","content":[{"type":"text","text":prompt}]}]
    inputs=processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    with torch.no_grad(): gen=model.generate(**inputs, max_new_tokens=600, do_sample=False)
    txt=processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    sents=[re.sub(r'^\s*\d+[.)]\s*','',l).strip() for l in txt.split('\n')]
    sents=[s for s in sents if len(s)>15 and not s.lower().startswith(('here','sure','phase list'))]
    sents=sents[:N]
    while len(sents)<N: sents.append(sents[-1] if sents else 'the diver moves through the air above the pool')
    return sents
dv=json.load(open(f'{OUT}/fd_dive_verbalizer.json')); sv=json.load(open(f'{OUT}/fd_sub_verbalizer.json'))
out_d={}; t0=time.time()
for cid,a in dv['actions'].items():
    key=f"({a['activity']}) {a['name']}"
    ctx=(f"Springboard/platform diving. The dive '{a['name']}' (FINA code {a['code']}) consists of these sub-actions in order: "+' -> '.join(a['member_names']))
    p=(f"Context: {ctx}.\nList exactly {N} short visual phases of this dive, in temporal order from the diver standing on the board/platform to the entry into the water. "
       f"Each phase must be one sentence describing only what is VISIBLE (setting, body position, rotation, twist, arm/leg configuration, motion). Do not mention the dive code. Number them 1-{N}.")
    out_d[key]=[f"a video of a {a['name']}, a {dv['activities'][[x for x in dv['activities'] if dv['activities'][x]['short']==a['activity']][0]]['name']} in springboard or platform diving"]+ask(p)
    print(f"[dive {cid}] {key} ({time.time()-t0:.0f}s): {out_d[key][1][:80]}", flush=True)
json.dump(out_d, open(f'{OUT}/fd_dive_ensemble.json','w'), indent=1)
out_s={}
for cid,a in sv['actions'].items():
    key=f"({a['activity']}) {a['name']}"; fam=sv['phrases'][a['phrase']]['name'].replace('_',' ')
    p=(f"Context: springboard/platform diving; the sub-action '{a['name']}' is a {fam} phase of a dive.\nList exactly {N} short visual phases of this sub-action alone, in temporal order. "
       f"Each phase must be one sentence describing only what is VISIBLE (body position, rotation, twist, arm/leg configuration, motion relative to the board and water). Number them 1-{N}.")
    out_s[key]=[f"a video of the diving sub-action {a['name']}, a {fam} phase"]+ask(p)
    print(f"[sub {cid}] {key} ({time.time()-t0:.0f}s): {out_s[key][1][:80]}", flush=True)
json.dump(out_s, open(f'{OUT}/fd_sub_ensemble.json','w'), indent=1); print('DONE', time.time()-t0)
