import csv,ast,os
R='/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving/'
A=R+'raw/FineDiving/Released_FineDiving_Dataset/'
te={(r['video'],r['clip_id']) for r in csv.DictReader(open(A+'train_test_split/test_split.csv'))}
co={(r['video'],r['clip_id']):r for r in csv.DictReader(open(A+'Annotations/FineDiving_coarse_annotation.csv'))}
fg={(r['video'],r['clip_id']):r for r in csv.DictReader(open(A+'Annotations/FineDiving_fine-grained_annotation_corrected.csv'))}
tab={int(r['id']):r['sub_action_type'] for r in csv.DictReader(open(A+'Annotations/Sub_action_Types_Table_filtered.csv'))}
seen={}
for k in sorted(te):
    if k not in fg: continue
    fl=ast.literal_eval(fg[k]['frames_labels']); seq=[fl[0]]
    for x in fl[1:]:
        if x!=seq[-1]: seq.append(x)
    names=tuple(tab.get(x,x) for x in seq); c=co[k]
    s,e=int(c['start_frame']),int(c['end_frame'])
    d=R+'RGB/'+k[0]
    if len(names)!=3 or not os.path.isdir(d): continue
    ok=all(os.path.exists(f"{d}/{f:06d}.jpg") for f in range(s-1,e,4))
    key=names
    if key not in seen and ok:
        seen[key]=1; print(k,fg[k]['action_type'],names,s,e,len(fl))
