#!/usr/bin/env python
"""FineDiving two-level hierarchy assets (dive <- ordered sub-actions <- phases), champion formats.
Outputs (configs/finediving/): fd_dive_verbalizer.json (52 dive classes c00..c51, phrases = identity p00..p51,
activities = 6 dive groups; actions[c].members = ORDERED sub-action ids of fd_sub_verbalizer), fd_sub_verbalizer.json
(28 sub-action types, dense ids, phrases = 5 families), fd_dive_attribute_table_s{seed}.json (code-derived binary
attributes + held-out split), fd_dive_duration_prior_s{seed}.json (log-normal per seen class, Jaccard-weighted for held-out).
"""
import json, math, random, collections, argparse, os
D='/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
ap=argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--n_held', type=int, default=13); ap.add_argument('--out', default='configs/finediving'); a=ap.parse_args()
GROUPS=[('a0','FWD','forward dive'),('a1','BCK','back dive'),('a2','REV','reverse dive'),('a3','INW','inward dive'),('a4','TWI','twisting dive'),('a5','ARM','armstand dive')]
POS={'a':'straight','b':'pike','c':'tuck','d':'free'}
def load_lines(p): return [json.loads(l) for l in open(p) if l.strip()]
train=load_lines(f'{D}/finediving_train_label.txt'); test=load_lines(f'{D}/finediving_test_label.txt')
codes=sorted({x['activity_id'] for e in train+test for x in e['raw_value']}); assert len(codes)==52
mapping=json.load(open(f'{D}/action_type_mapping.json')); assert [mapping[c] for c in codes]==[f'a{i:02d}' for i in range(52)], 'mapping order != sorted order'
subs=json.load(open(f'{D}/action_to_subaction_mapping.json')); rawv=json.load(open(f'{D}/verbalizer.json'))
# ---- sub-action verbalizer (dense ids in raw c-order) ----
raw_sub=[(k,v) for k,v in sorted(rawv['actions'].items())]; sub_name2id={}; sub_actions={}
fam=rawv['phrases']  # p00..p04 families
fam_short={'p00':'TKO','p01':'SOMP','p02':'SOMT','p03':'TWST','p04':'ENT'}
for i,(k,v) in enumerate(raw_sub):
    cid=f'c{i:02d}'; sub_name2id[v['name']]=cid
    sub_actions[cid]={'id':cid,'name':humanize(v['name']) if False else v['name'],'raw_id':k,'phrase':v['phrase'],'activity':fam_short[v['phrase']]}
def hum_sub(n):
    n=n.replace('Soms.Pike','somersaults pike').replace('Som.Pike','somersault pike').replace('Soms.Tuck','somersaults tuck').replace('Som.Tuck','somersault tuck')
    n=n.replace('Twists','twists').replace('Twist','twist').replace('Arm.Forward','armstand forward takeoff').replace('Arm.Back','armstand back takeoff').replace('Arm.Reverse','armstand reverse takeoff')
    for w in ('Forward','Back','Reverse','Inward'):
        if n==w: n=f'{w.lower()} takeoff'
    if n=='Entry': n='entry into the water'
    return n
for cid,v in sub_actions.items(): v['name']=hum_sub(v['name'])
sub_verb={'dataset':'finediving_sub','activities':{f'a{i}':{'id':f'a{i}','name':fam[p]['name'].replace('_',' '),'short':fam_short[p]} for i,p in enumerate(sorted(fam))},
          'phrases':{p:{'id':p,'name':fam[p]['name'],'activity':f'a{i}'} for i,p in enumerate(sorted(fam))},'actions':sub_actions}
# ---- dive verbalizer ----
def decode(code):
    g=int(code[0])-1; pos=POS[code[-1]]; digits=code[:-1]
    if g in (0,1,2,3):   # 1xy: y half-somersaults, x=1 flying
        flying = digits[1]=='1'; som=int(digits[2])/2; tw=0.0
    elif g==4:           # 5xyz
        flying=False; som=int(digits[2])/2; tw=int(digits[3])/2
    else:                # 6xyz armstand
        flying=False; som=int(digits[2])/2 if len(digits)>2 else 0; tw=int(digits[3])/2 if len(digits)>3 else 0
    return g,pos,som,tw,flying
dive_actions={}; dive_phr={}
for i,c in enumerate(codes):
    g,pos,som,tw,fly=decode(c); aid,short,gname=GROUPS[g]
    DIRS={'1':'forward','2':'back','3':'reverse','4':'inward'}
    if g==4:   gname=f"{DIRS.get(c[1],'')} twisting dive".strip()      # 5xyz: x = somersault direction
    elif g==5: gname=f"armstand {DIRS.get(c[1],'')} dive".replace('  ',' ')   # 6xyz: x = direction
    parts=[gname]
    if fly: parts.append('flying')
    if som>0: parts.append(f'{som:g} somersault{"s" if som!=1 else ""}')
    if tw>0: parts.append(f'{tw:g} twist{"s" if tw!=1 else ""}')
    parts.append(f'{pos} position')
    name=', '.join(parts)+f' (dive {code_up})' if False else ', '.join(parts)
    members=[sub_name2id[s] for s in subs[c]]
    cid=f'c{i:02d}'; pid=f'p{i:02d}'
    dive_actions[cid]={'id':cid,'name':name,'code':c,'phrase':pid,'activity':short,'members':members,'member_names':[sub_actions[m]['name'] for m in members]}
    dive_phr[pid]={'id':pid,'name':name,'activity':aid,'set':i}
dive_verb={'dataset':'finediving_dive','activities':{a:{'id':a,'name':n,'short':s} for a,s,n in GROUPS},'phrases':dive_phr,'actions':dive_actions,
           '_note':'detection classes = 52 dive codes (sorted); phrases = identity; activities = 6 dive groups (parent for coarse supervision); members = ORDERED sub-actions (fd_sub_verbalizer ids)'}
# ---- attribute table + split ----
attr_names=[f'group_{n.split()[0]}' for _,_,n in GROUPS]+[f'pos_{p}' for p in ('straight','pike','tuck','free')]+['som_le_1','som_1.5','som_2','som_2.5','som_3','som_3.5','som_ge_4','twist_0','twist_0.5_1','twist_1.5_2','twist_ge_2.5','flying','armstand_takeoff']
def attrs(code):
    g,pos,som,tw,fly=decode(code); v=[0]*len(attr_names)
    v[g]=1; v[6+['straight','pike','tuck','free'].index(pos)]=1
    v[10+ (0 if som<=1 else 1 if som==1.5 else 2 if som==2 else 3 if som==2.5 else 4 if som==3 else 5 if som==3.5 else 6)]=1
    v[17+ (0 if tw==0 else 1 if tw<=1 else 2 if tw<=2 else 3)]=1
    v[21]=int(fly); v[22]=int(g==5); return v
tr_cnt=collections.Counter(x['activity_id'] for e in train for x in e['raw_value']); te_cnt=collections.Counter(x['activity_id'] for e in test for x in e['raw_value'])
elig=[c for c in codes if te_cnt[c]>=4 and tr_cnt[c]>=8]
rng=random.Random(a.seed); held=[]
while True:
    cand=rng.sample(elig, a.n_held); groups_left=collections.Counter(decode(c)[0] for c in codes if c not in cand)
    if all(groups_left[g]>=1 for g in set(decode(c)[0] for c in codes)): held=sorted(cand); break
held_ids=[f'c{codes.index(c):02d}' for c in held]
table={'attribute_names':attr_names,'actions':{f'c{i:02d}':{'name':dive_actions[f'c{i:02d}']['name'],'code':c,'attrs':attrs(c),'members':dive_actions[f'c{i:02d}']['members']} for i,c in enumerate(codes)},
       'held_out_classes':held_ids,'note':f'FineDiving dive-level split seed {a.seed}: {a.n_held} of 52 dive codes held out (eligible: >=4 test and >=8 train instances); train/test instances of held-out: '+', '.join(f'{c}:{tr_cnt[c]}/{te_cnt[c]}' for c in held)}
# ---- duration prior ----
dur=collections.defaultdict(list)
for e in train:
    for x in e['raw_value']:
        s,t=[float(v.strip('<>').replace(' seconds','')) for v in x['span']]; dur[codes.index(x['activity_id'])].append(max(t-s,1e-3))
SF=0.10; stats={}; heldi=[int(h[1:]) for h in held_ids]; seen=[i for i in range(52) if i not in heldi]
for c in seen:
    logs=[math.log(d) for d in dur[c]]; mu=sum(logs)/len(logs); var=sum((x-mu)**2 for x in logs)/(len(logs)-1) if len(logs)>1 else 0.0
    stats[c]={'mu':mu,'sigma':max(math.sqrt(var),SF),'n':len(logs),'kind':'seen'}
A={i:set(j for j,v in enumerate(attrs(codes[i])) if v) for i in range(52)}
for h in heldi:
    ws=[len(A[h]&A[s])/max(len(A[h]|A[s]),1) for s in seen]; Z=sum(ws) or 1.0; ws=[w/Z for w in ws]
    mu=sum(w*stats[s]['mu'] for w,s in zip(ws,seen)); m2=sum(w*(stats[s]['sigma']**2+stats[s]['mu']**2) for w,s in zip(ws,seen))
    stats[h]={'mu':mu,'sigma':max(math.sqrt(max(m2-mu*mu,0.0)),SF),'n':0,'kind':'held_out'}
prior={'meta':{'labels':f'{D}/finediving_train_label.txt','attr_table':f'{a.out}/fd_dive_attribute_table_s{a.seed}.json','sigma_floor':SF,'num_seen':len(seen),'num_held_out':len(heldi),'held_out':heldi,'level':'dive','note':'log-normal over dive durations (s); held-out = attribute-Jaccard-weighted seen mixture'},
       'classes':{str(c):stats[c] for c in range(52)}}
os.makedirs(a.out, exist_ok=True)
json.dump(dive_verb, open(f'{a.out}/fd_dive_verbalizer.json','w'), indent=1); json.dump(sub_verb, open(f'{a.out}/fd_sub_verbalizer.json','w'), indent=1)
json.dump(table, open(f'{a.out}/fd_dive_attribute_table_s{a.seed}.json','w'), indent=1); json.dump(prior, open(f'{a.out}/fd_dive_duration_prior_s{a.seed}.json','w'), indent=1)
print('dive classes:', [(k,v['code'],v['name'],v['member_names']) for k,v in list(dive_actions.items())[:3]])
print('sub-actions:', [(k,v['name'],v['phrase']) for k,v in list(sub_actions.items())[:5]], '... n=', len(sub_actions))
print('held-out (%d):'%len(held), [(h, dive_actions[h]['code'], dive_actions[h]['name'], f"train {tr_cnt[dive_actions[h]['code']]} test {te_cnt[dive_actions[h]['code']]}") for h in held_ids])
print('seen per group:', collections.Counter(decode(codes[i])[0] for i in seen), '| held per group:', collections.Counter(decode(codes[i])[0] for i in heldi))
