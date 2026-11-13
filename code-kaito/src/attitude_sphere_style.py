"""Render attitude interval preimages with Example 1's transparent sphere."""
import numpy as np
import matplotlib.pyplot as plt
from sphere_pullback import draw_transparent_sphere
from common import figure_style as style

def sphere_panel(phase, coordinate, path, svg=True):
    boxes=np.asarray(phase['boxes'])
    rows=[];pale=[]
    for bounds,node,basin in zip(boxes,phase['membership'],phase['basin_node']):
        z=np.interp(bounds,coordinate['values'],coordinate['knots']).tolist()
        if z[1]<=z[0]:continue
        if basin>=0:pale.append(dict(z=z,node=int(basin)))
        if node>=0:rows.append(dict(z=z,node=int(node)))
    def merge_adjacent(bands):
        merged=[]
        for row in sorted(bands,key=lambda r:r['z'][0]):
            if merged and merged[-1]['node']==row['node'] and abs(merged[-1]['z'][1]-row['z'][0])<1e-10:
                merged[-1]['z'][1]=row['z'][1]
            else:merged.append(dict(node=row['node'],z=list(row['z'])))
        return merged
    rows=merge_adjacent(rows);pale=merge_adjacent(pale)
    palette={r['node']:style.morse(r['node']) for r in rows}
    light={r['node']:style.tint(r['node']) for r in pale}
    fig,ax=plt.subplots(figsize=(3.4,3.6))
    if pale:draw_transparent_sphere(ax,pale,light)
    draw_transparent_sphere(ax,rows,palette)
    elevation=np.deg2rad(12)
    for row in rows:
        if row['z'][0]>-1 and row['z'][1]<1:
            height=np.mean(row['z'])
            ax.text(1.02,height*np.cos(elevation),'$C$',color=palette[row['node']],fontsize=13,ha='left',va='center')
    for sign,label in [(1,'$N$'),(-1,'$S$')]:
        ax.plot(0,sign*np.cos(elevation),'o',ms=3,mfc='white',mec='.3',alpha=1 if sign>0 else .4,zorder=5)
        ax.text(0,sign*1.07,label,ha='center',va='center',fontsize=13)
    ax.set(aspect='equal',xlim=(-1.12,1.2),ylim=(-1.14,1.14));ax.axis('off')
    fig.tight_layout(pad=.3)
    for ext in ['pdf','svg'] if svg else ['pdf']:fig.savefig(path.with_suffix('.'+ext),bbox_inches='tight',dpi=220)
    plt.close(fig)
