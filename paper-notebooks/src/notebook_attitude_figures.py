"""Notebook presentation of computed attitude graphs and latent-sphere cells.

Colors and the 12-degree sphere projection follow the manuscript's Example 3.
Inputs are computed cell records, never manuscript images or substituted results.
"""
from pathlib import Path
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

COLORS = {'S': '#D55E00', 'C': '#0072B2', 'A': '#CC79A7'}
FONT = {'font.family': 'serif', 'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 11, 'axes.titlesize': 12,
        'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10}

def color(policy, name):
    return ('#0072B2' if policy == 'nominal' else '#009E73') if name == 'N' else COLORS.get(name, '#CC79A7')

def graph(nodes, edges, names, policy, indices, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['digraph {', 'graph [margin="0",pad="0.04",ranksep="0.30",nodesep="0.12"];',
             'node [shape=ellipse,style=filled,fontname="Times-Roman",fontsize=15,margin="0.065,0.04",penwidth=0.8];',
             'edge [color="#4d4d4d",penwidth=0.9,arrowsize=0.7];']
    for node in nodes:
        name=names[node]; ink=color(policy,name)
        rgb=[int(ink[i:i+2],16) for i in (1,3,5)]
        tint='#' + ''.join(f'{round(.4*v+.6*255):02x}' for v in rgb)
        label=f"{name}: ({', '.join(indices[node])})"
        lines.append(f'{node} [label="{label}",color="{ink}",fillcolor="{tint}"];')
    lines += [f'{a} -> {b};' for a,b in edges] + ['}']
    source='\n'.join(lines)
    path.with_suffix('.dot').write_text(source+'\n')
    for ext in ('pdf','svg'):
        subprocess.run(['dot', '-T'+ext, '-o', str(path.with_suffix('.'+ext))], input=source,text=True,check=True)
    return path.with_suffix('.pdf')

def sphere(boxes, membership, knots, values, names, policy, path, basin_labels=None, strip=False):
    boxes=np.asarray(boxes); heights=np.interp(boxes,values,knots)
    def bands(labels):
        out=[]
        for bounds,node in zip(heights,labels):
            if node<0:continue
            if out and out[-1][2]==node and abs(out[-1][1]-bounds[0])<1e-12:
                out[-1]=(out[-1][0],float(bounds[1]),int(node))
            else:out.append((float(bounds[0]),float(bounds[1]),int(node)))
        return out
    morse=bands(membership); basin=[] if basin_labels is None else bands(basin_labels)
    with plt.rc_context(FONT):
        fig=plt.figure(figsize=(3.1,3.7 if strip else 3.0))
        ax=fig.add_axes([.02,.26 if strip else .02,.96,.72 if strip else .96])
        ax.add_patch(Circle((0,0),1,facecolor='#fafafa',edgecolor='.65',lw=.8,zorder=0))
        t=np.linspace(-1,1,801);u,v=np.meshgrid(t,t);rr=u*u+v*v
        depth=np.sqrt(np.maximum(0,1-rr)); elevation=np.deg2rad(12)
        for sign,alpha in [(-1,.05),(1,.16)]:
            height=np.ma.array(v*np.cos(elevation)+sign*depth*np.sin(elevation),mask=rr>1)
            for lo,hi,node in basin:ax.contourf(u,v,height,levels=[lo,hi],colors=[color(policy,names[node])],alpha=alpha,zorder=1)
        for sign,alpha,layer in [(-1,.24,1),(1,.85,3)]:
            height=np.ma.array(v*np.cos(elevation)+sign*depth*np.sin(elevation),mask=rr>1)
            if sign==1:ax.add_patch(Circle((0,0),1,facecolor='#e6e9ed',edgecolor='none',alpha=.2,zorder=2))
            for lo,hi,node in morse:
                ax.contourf(u,v,height,levels=[lo,hi],colors=[color(policy,names[node])],alpha=.13 if sign<0 and lo<=-1 else alpha,zorder=layer)
        ax.add_patch(Circle((0,0),1,facecolor='none',edgecolor='.65',lw=.7,zorder=4))
        for lo,hi,node in morse:
            name=names[node];ink=color(policy,name)
            if name in ('N','S'):
                ax.text(0,1.13 if name=='N' else -1.13,'$'+name+'$',color=ink,fontsize=12,ha='center',va='center',zorder=10)
            else:
                z=(lo+hi)/2;r=np.sqrt(1-z*z);a=1.13
                point=(r*np.sin(a),z*np.cos(elevation)-r*np.cos(a)*np.sin(elevation))
                ax.annotate('$'+name+'$',xy=point,xytext=(1.26,point[1]+.16),color=ink,fontsize=12,ha='center',va='center',zorder=10,
                            arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.36',color=ink,lw=.8,mutation_scale=9,shrinkA=4,shrinkB=1))
        ax.set(xlim=(-1.2,1.45),ylim=(-1.25,1.25),aspect='equal');ax.axis('off')
        if strip:
            line=fig.add_axes([.12,.13,.76,.10]);line.axhline(.5,color='.65',lw=.8)
            for bounds,node in zip(boxes,membership):
                if node>=0:line.axvspan(*bounds,ymin=.3,ymax=.7,color=color(policy,names[node]),lw=0)
            line.set(xlim=(-1.04,1.04),ylim=(0,1),xticks=[-1,0,1],yticks=[],xlabel='$u$')
            for spine in line.spines.values():spine.set_visible(False)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('pdf','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=200)
        plt.close(fig)
    return path.with_suffix('.pdf')

def show(paths, titles, width=900):
    """Keep native panel proportions and a common graph font scale at notebook size."""
    import io,tempfile
    from PIL import Image as PILImage
    from IPython.display import Image,display
    panels=[]
    with tempfile.TemporaryDirectory() as tmp:
        for i,path in enumerate(paths):
            stem=Path(tmp)/str(i)
            subprocess.run(['pdftoppm','-r','100','-png','-singlefile',str(path),str(stem)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            panels.append(PILImage.open(stem.with_suffix('.png')).convert('RGB'))
        # Fixed scale for every graph, including the two-node and three-node cases.
        with plt.rc_context(FONT):
            height = 240 if all(Path(p).with_suffix(".dot").exists() for p in paths) else 360
            fig=plt.figure(figsize=(width/100,height/100))
            cell=width/len(paths)
            for i,(panel,title) in enumerate(zip(panels,titles)):
                scale = .72 if Path(paths[i]).with_suffix(".dot").exists() else min(.9, 260/panel.height)
                if panel.width*scale > cell-16:
                    raise ValueError("Panel exceeds its column; increase the display width.")
                w,h=panel.width*scale,panel.height*scale
                ax=fig.add_axes([(i*cell+(cell-w)/2)/width,(.54*height-h/2)/height,w/width,h/height])
                ax.imshow(panel);ax.axis('off')
                fig.text((i+.5)/len(paths),.055,title,ha='center',va='center',fontsize=11)
            buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=160);plt.close(fig)
    display(Image(data=buffer.getvalue(),width=width))


def show_plot(fig):
    """Display a plot at a readable width independent of its export resolution."""
    import io
    from IPython.display import Image, display
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    width = min(900, round(90 * fig.get_size_inches()[0]))
    display(Image(data=buffer.getvalue(), width=width))
    plt.close(fig)
