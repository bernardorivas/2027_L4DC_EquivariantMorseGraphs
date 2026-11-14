"""Render attitude interval preimages with Example 1's transparent sphere."""
import numpy as np
import matplotlib.pyplot as plt
from sphere_pullback import draw_transparent_sphere
from common import figure_style as style

def sphere_panel(phase, coordinate, path, svg=True):
    boxes=np.asarray(phase['boxes'])
    rows=[]
    for bounds,node in zip(boxes,phase['membership']):
        z=np.interp(bounds,coordinate['values'],coordinate['knots']).tolist()
        if z[1]<=z[0]:continue
        if node>=0:rows.append(dict(z=z,node=int(node)))
    def merge_adjacent(bands):
        merged=[]
        for row in sorted(bands,key=lambda r:r['z'][0]):
            if merged and merged[-1]['node']==row['node'] and abs(merged[-1]['z'][1]-row['z'][0])<1e-10:
                merged[-1]['z'][1]=row['z'][1]
            else:merged.append(dict(node=row['node'],z=list(row['z'])))
        return merged
    rows=merge_adjacent(rows)
    palette={r['node']:style.morse(r['node']) for r in rows}
    with plt.rc_context({'font.family':'serif','mathtext.fontset':'cm','font.size':13}):
        fig,(ax,strip)=plt.subplots(2,1,figsize=(4.6,4.8),gridspec_kw={'height_ratios':[4.5,1]})
        draw_transparent_sphere(ax,rows,palette)
        elevation=np.deg2rad(12)
        for row in rows:
            lo,hi=row['z']; color=palette[row['node']]
            pole=lo<=-1 or hi>=1
            z=(-1 if lo<=-1 else 1) if pole else (lo+hi)/2
            label='$S$' if lo<=-1 else ('$N$' if hi>=1 else '$C$')
            longitude=1.08 if not pole else 0
            radius=np.sqrt(max(0,1-z*z))
            point=(radius*np.sin(longitude),z*np.cos(elevation)-radius*np.cos(longitude)*np.sin(elevation))
            target=(0,1.13*np.sign(z)) if pole else (1.22,point[1]+.19)
            ax.annotate(label,xy=point,xytext=target,color=color,fontsize=16,ha='center',va='center',zorder=10,
                arrowprops=dict(arrowstyle='-' if pole else '->',connectionstyle='arc3,rad=0' if pole else 'arc3,rad=.3',color=color,lw=.9))
        ax.set(aspect='equal',xlim=(-1.08,1.43),ylim=(-1.22,1.22));ax.axis('off')
        strip.axhline(.5,color='.65',lw=.8)
        for row in rows:
            strip.axvspan(*row['z'],ymin=.30,ymax=.70,color=palette[row['node']],alpha=.85,lw=0)
            lo,hi=row['z'];name='S' if lo<=-1 else ('N' if hi>=1 else 'C')
            strip.text((lo+hi)/2,.78,'$'+name+'$',ha='center',color=palette[row['node']],fontsize=13)
        strip.set(xlim=(-1.04,1.04),ylim=(0,1),xticks=[-1,0,1],yticks=[],xlabel='Latent sphere height')
        for spine in strip.spines.values():spine.set_visible(False)
        strip.tick_params(axis='x',length=0)
        fig.tight_layout(pad=.4)
        for ext in ['pdf','svg'] if svg else ['pdf']:fig.savefig(path.with_suffix('.'+ext),bbox_inches='tight',dpi=220)
        plt.close(fig)
