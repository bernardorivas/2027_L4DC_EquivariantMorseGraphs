"""Compact O(3) quotient portrait using the pinned exact flow."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sphere_flow_figure import RC, ROOT
import o3_geometry as exact
from paper_figure_style import O3_COLORS

def draw(ax):
    ax.set(aspect='equal',xlim=(-1.16,1.16),ylim=(-.22,1.14),xlabel='$a$',ylabel='$r$')
    ax.set_xticks([-1,0,1]);ax.set_yticks([0,1]);ax.tick_params(length=3,width=.65,pad=3)
    ax.spines['bottom'].set_position(('data',-.22))
    def path(a,r,fraction=.58,lw=.85):
        p=np.column_stack((a,r));ax.plot(a,r,color='.52',lw=lw,zorder=1)
        # Locate arrow by arc length rather than uneven flow time samples.
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        s=fraction*length[-1]
        start=np.array([np.interp(s-.045,length,p[:,k]) for k in (0,1)])
        end=np.array([np.interp(s,length,p[:,k]) for k in (0,1)])
        ax.annotate('',xy=end,xytext=start,zorder=2,arrowprops=dict(arrowstyle='-|>',lw=.8,color='.48',mutation_scale=8,shrinkA=0,shrinkB=0))
    t=np.linspace(-5,6,700)
    for sign in (-1,1):
        for a,r in [(.39,.22),(.34,.59)]:
            x,y=exact.flow(sign*a,r,t);path(x,y,.60)
        angle=np.linspace(np.pi/2,0 if sign>0 else np.pi,300)
        path(np.cos(angle),np.sin(angle),.48,1.0)
        path(np.linspace(0,sign,200),np.zeros(200),.55)
    path(np.zeros(200),np.linspace(0,1,200),.58,1.0)
    label_positions={'A-':(-1.,-.10),'A+':(1.,-.10),'O':(0.,-.10),'C':(0.,1.10)}
    for role,(a,r) in exact.ANCHORS.items():
        ax.plot(a,r,'o',color=O3_COLORS[role],ms=4,zorder=5)
        ax.text(*label_positions[role], '$'+exact.MATH[role]+'$', ha='center', va='center', color=O3_COLORS[role], zorder=6)

def render():
    out=ROOT/'artifacts/o3-flow';out.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(RC):
        fig,ax=plt.subplots(figsize=(3.5,2.1));draw(ax)
        for ext in ('pdf','svg','png'):fig.savefig(out/f'quotient-flow.{ext}',bbox_inches='tight',pad_inches=.035,dpi=220)
        plt.close(fig)
    return out
if __name__=='__main__':print(render())
