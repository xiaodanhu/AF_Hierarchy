import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'pdf.fonttype':42,'font.size':7,'axes.linewidth':0.5})
import textwrap, numpy as np
from PIL import Image
W=6.5
def new_fig(H): 
    fig=plt.figure(figsize=(W,H)); return fig
def ax_in(fig,x,y,w,h,H):
    # x,y,w,h in inches from top-left
    ax=fig.add_axes([x/W,1-(y+h)/H,w/W,h/H]); ax.set_axis_off(); return ax
def sentences(fig,H,y0,sents,fs=6.1,x0=0.05,wrap=67,startidx=1,single=False):   # 6.3pt/70 chars overran the column (FineGym (d) line 4 collided with the right column)
    # two columns, returns bottom y
    K=len(sents); half=(K+1)//2; colw=(W-0.1)/2; lh=fs*1.18/72
    bottoms=[]
    for c,(lo,hi) in enumerate([(0,K)] if single else [(0,half),(half,K)]):
        y=y0
        for k in range(lo,hi):
            lines=textwrap.wrap(sents[k],wrap,subsequent_indent='   ')
            txt=f"{k+1}  "+lines[0] if False else None
            lines=textwrap.wrap(f"{k+1}. "+sents[k],wrap,subsequent_indent='    ')
            fig.text((x0+c*colw)/W,1-y/H,"\n".join(lines),fontsize=fs,va='top',ha='left',linespacing=1.15)
            y+=len(lines)*lh*1.0+0.045
        bottoms.append(y)
    return max(bottoms)
