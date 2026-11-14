import json,math,os,sys,subprocess,tempfile
TMP=tempfile.mkdtemp(prefix='thfig_')
sys.path.insert(0,os.path.dirname(__file__))
from common import *
from matplotlib.patches import Rectangle
db=json.load(open('/data3/xiaodan8/thumos/annotations/thumos14.json'))['database']
S=json.load(open('/data3/xiaodan8/tifad/configs/thumos14_qwen_ensemble.json'))
FR='/data3/xiaodan8/thumos/frames_1fps_336/'
INST=[('PoleVault','video_test_0001207',600.4,611.5),('LongJump','video_test_0001369',196.0,199.2),('HighJump','video_test_0000538',133.0,136.7),('BaseballPitch','video_test_0000714',141.4,144.7)]   # four instances from tools/verify_kit/thumos_selection.json (random draws, unusable ones replaced): the three classes of the submitted figure plus the first other class in kit order 
if len(sys.argv)>1: INST=json.loads(sys.argv[1])
K=10
def inst_info(c,v,s,e):
    a=[x for x in db[v]['annotations'] if x['label']==c and abs(x['segment'][0]-s)<0.06 and abs(x['segment'][1]-e)<0.06]
    return a[0]['segment']
import sys as _s
H=float(os.environ.get('HH','9.0')); fig=new_fig(H); y=0.05; out=[]
for pi,(c,v,s,e) in enumerate(INST):
    s,e=inst_info(c,v,s,e)
    # one frame at the center of each of the K equal chunks (ffmpeg), so frame k
    # shows what sentence k is aligned to ; earlier: twelve frames at equal steps / the 1-fps store
    NF=K; L=e-s; ns=[]
    for j in range(NF):
        t=s+(j+0.5)*L/NF; fp=f"{TMP}/{pi}_{j}.png"
        subprocess.run(['ffmpeg','-y','-v','error','-ss',f'{t:.3f}','-i',f'/data3/xiaodan8/thumos/video/{v}.mp4','-frames:v','1','-vf','scale=320:-2',fp],check=True); ns.append(fp)
    sent=S[c][1:1+K]; assert len(sent)==K
    fig.text(0.05/W,1-y/H,f"({'abcd'[pi]}) {c}   {v}   [{s:.1f} s, {e:.1f} s]   one frame at the center of each chunk",fontsize=7.5,fontweight='bold',va='top')
    y+=0.17
    ox=0.05; wi=W-0.1; tw=wi/NF
    im0=Image.open(ns[0]); fh=tw*im0.size[1]/im0.size[0]
    ax=ax_in(fig,ox,y,wi,fh,H); ax.set_xlim(s,e); ax.set_ylim(1,0)
    for j,fp in enumerate(ns):
        ax.imshow(np.asarray(Image.open(fp).convert('RGB')),extent=(s+j*L/NF,s+(j+1)*L/NF,1,0),aspect='auto',interpolation='bilinear')
    ax.set_xlim(s,e); ax.set_ylim(1,0)
    for k in range(1,K):
        ax.plot([s+k*(e-s)/K]*2,[0,1],color='white',lw=0.4,ls=(0,(2,2)),alpha=0.9)
    y+=fh+0.02
    bh=0.16; ab=ax_in(fig,ox,y,wi,bh,H); ab.set_xlim(s,e); ab.set_ylim(1,0)
    for k in range(K):
        a,b=s+k*(e-s)/K,s+(k+1)*(e-s)/K
        ab.add_patch(Rectangle((a,0),b-a,1,fc='0.92' if k%2==0 else 'white',ec='black',lw=0.5))
        ab.text((a+b)/2,0.55,str(k+1),ha='center',va='center',fontsize=6.5)
    y+=bh+0.03
    ax2=ax_in(fig,ox,y,wi,0.1,H); ax2.set_xlim(s,e)
    ax2.annotate('',xy=(e,0.5),xytext=(s,0.5),arrowprops=dict(arrowstyle='<->',lw=0.4,shrinkA=0,shrinkB=0),annotation_clip=False)
    ax2.text((s+e)/2,0.5,f" {e-s:.1f} s, 10 equal chunks of {(e-s)/K:.2f} s ",ha='center',va='center',fontsize=5.8,bbox=dict(fc='white',ec='none',pad=0.5))
    y+=0.14
    y=sentences(fig,H,y,sent)+0.12
    out.append((c,v,s,e,len(ns)))
print(out,'height used',y)
assert y<=10.5   # full-page figure; scaled to \linewidth in the appendix
if os.environ.get('HH') is None: print('REDO',y); sys.exit(0)
fig.set_size_inches(W,H)
D='/data3/xiaodan8/actionformer6/tools/figures/'
fig.savefig(D+'tree_examples_thumos.pdf'); fig.savefig(D+'tree_examples_thumos.png',dpi=150)
