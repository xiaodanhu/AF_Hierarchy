import json,os,sys,csv,ast
sys.path.insert(0,os.path.dirname(__file__))
from common import *
from matplotlib.patches import Rectangle
R='/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving/'
A=R+'raw/FineDiving/Released_FineDiving_Dataset/Annotations/'
co={(r['video'],r['clip_id']):r for r in csv.DictReader(open(A+'FineDiving_coarse_annotation.csv'))}
fg={(r['video'],r['clip_id']):r for r in csv.DictReader(open(A+'FineDiving_fine-grained_annotation_corrected.csv'))}
tab={int(r['id']):r['sub_action_type'] for r in csv.DictReader(open(A+'Sub_action_Types_Table_filtered.csv'))}
S=json.load(open('/data3/xiaodan8/actionformer6/configs/finediving/fd_sub_ensemble.json'))
KEYS={k.split(') ',1)[1]:k for k in S}
def key_of(name,pos):
    # annotated name -> fd_sub_ensemble key
    if name in('Forward','Back','Reverse','Inward'): return KEYS[name.lower()+' takeoff']
    if name.startswith('Arm.'): return KEYS['armstand '+name[4:].lower()+' takeoff']
    if name=='Entry': return KEYS['entry into the water']
    if 'Twist' in name: return KEYS[name.lower()]
    n,r=name.split(' ',1); kind=r.split('.')[1].lower(); n_=n
    word='somersault' if n=='1' else 'somersaults'
    return KEYS[f"{n} {word} {kind}"]
DIVES=[(('09','6'),'forward'),(('07','3'),'twisting (armstand forward)')]
H=float(os.environ.get('HH','9.0')); fig=new_fig(H); y=0.05; out=[]
for pi,(k,typ) in enumerate(DIVES):
    fl=ast.literal_eval(fg[k]['frames_labels']); N=len(fl)
    s0=int(co[k]['start_frame']); code=fg[k]['action_type']
    seq=[fl[0]]; b=[0]
    for i,x in enumerate(fl):
        if x!=seq[-1]: seq.append(x); b.append(i)
    b.append(N); names=[tab[x] for x in seq]; K=len(names); assert K==3
    keys=[key_of(n,i) for i,n in enumerate(names)]
    # five phase sentences per sub-action (K'=5), picked from the
    # ten generated ones (1-based phase indices into fd_sub_ensemble.json); fd_ten_phases.py = all ten
    PICK={('09','6'):[[1,2,5,8,10],[1,4,5,8,10],[1,4,5,8,10]], ('07','3'):[[1,3,5,7,10],[1,3,5,8,10],[1,2,5,8,10]]}
    sent=[[S[kk][j] for j in PICK[k][i]] for i,kk in enumerate(keys)]
    out.append((k,code,typ,names,keys,s0,N,b))
    fig.text(0.05/W,1-y/H,f"({'ac'[pi]}) {typ} dive, type {code}, video {k[0]} dive {k[1]}, frames {s0}-{s0+N-1}",fontsize=7.5,fontweight='bold',va='top')
    y+=0.17; ox=0.05; wi=W-0.1; nf=12; tw=wi/nf
    fr=Image.open(f"{R}RGB/{k[0]}/{s0-1:06d}.jpg"); th=tw*fr.size[1]/fr.size[0]
    ax=ax_in(fig,ox,y,wi,th,H); ax.set_xlim(0,N); ax.set_ylim(1,0)
    for j in range(nf):
        ic=(j+0.5)*N/nf; q=int(round((ic-0.5)/4.0))*4  # nearest stored frame (stride 4), file = start-1+i
        q=min(max(q,0),(N-1)//4*4); f=f"{R}RGB/{k[0]}/{s0-1+q:06d}.jpg"; assert os.path.exists(f)
        ax.imshow(np.asarray(Image.open(f).convert('RGB')),extent=(j*N/nf,(j+1)*N/nf,1,0),aspect='auto')
    ax.set_xlim(0,N); ax.set_ylim(1,0)
    y+=th+0.02
    bh=0.19; a1=ax_in(fig,ox,y,wi,bh,H); a1.set_xlim(0,N); a1.set_ylim(1,0)
    for i in range(K):
        a1.add_patch(Rectangle((b[i],0),b[i+1]-b[i],1,fc='0.92' if i%2==0 else 'white',ec='none'))
        a1.text((b[i]+b[i+1])/2,0.52,names[i],ha='center',va='center',fontsize=6.5)
    for x in b: a1.plot([x,x],[0,1],color='black',lw=0.8)
    a1.plot([0,N],[0,0],color='black',lw=0.5); a1.plot([0,N],[1,1],color='black',lw=0.5)
    fig.text((ox-0.0)/W,1-(y-0.005)/H,'',fontsize=5)
    y+=bh+0.04
    a2=ax_in(fig,ox,y,wi,bh,H); a2.set_xlim(0,N); a2.set_ylim(1,0)
    for i in range(K):
        a2.text((i+0.5)*N/K,0.52,str(i+1),ha='center',va='center',fontsize=6.5)
    for i in range(K+1): a2.plot([i*N/K]*2,[0,1],color='black',lw=0.6,ls=(0,(3,2)))
    a2.plot([0,N],[0,0],color='black',lw=0.4); a2.plot([0,N],[1,1],color='black',lw=0.4)
    y+=bh+0.02
    fig.text(0.05/W,1-y/H,"upper bar: annotated sub-actions (solid); lower bar: 3 equal chunks (dashed). Columns: the five phase sentences of sub-action k, aligned with chunk k.",fontsize=5.8,va='top',color='0.25')
    y+=0.14
    cw=wi/K; fs=6.0; lh=fs*1.15/72; yb=y
    for c in range(K):
        x=ox+c*cw; yy=y
        fig.text((x+0.03)/W,1-yy/H,f"{names[c]}",fontsize=6.5,fontweight='bold',va='top'); yy+=0.14
        for j,sn in enumerate(sent[c]):
            ls=textwrap.wrap(f"{j+1}. "+sn,48,subsequent_indent='    ')
            fig.text((x+0.03)/W,1-yy/H,"\n".join(ls),fontsize=fs,va='top',linespacing=1.15)
            yy+=len(ls)*lh+0.03
        yb=max(yb,yy)
    y=yb
    y+=0.1
print(out,y)
assert y<=9
if os.environ.get('HH') is None: print('REDO',y); sys.exit(0)
D='/data3/xiaodan8/actionformer6/tools/figures/'
fig.savefig(D+'tree_examples_finediving.pdf'); fig.savefig(D+'tree_examples_finediving.png',dpi=150)
