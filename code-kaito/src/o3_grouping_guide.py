"""Explain saved 64-cell O(3) reachability and aggregation with one role palette."""
from pathlib import Path
import json
import hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def show_grouping_guide(fit, study):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle, Patch
    from graphviz import Digraph
    from PIL import Image
    from IPython.display import display, Markdown
    from notebook_support import table
    from paper_figure_style import O3_COLORS, O3_IDS, style
    folder = ROOT / study['study']
    out = ROOT / study['checks']
    records = json.loads((ROOT/'build/o3-certification/regions.json').read_text())
    record = next(r for r in records if r['case']=='learned')
    assert record['checkpoint_sha256']==hashlib.sha256((fit/'models.npz').read_bytes()).hexdigest()
    group = json.loads((ROOT/'build/o3-aggregation/result-64.json').read_text())
    assert group['path_closed_disjoint'] and group['index_pairs_validated']
    comparison=json.loads((folder/'comparison.json').read_text())
    native=next(r for r in comparison['resolutions'] if r['n']==64)
    roles={int(k):v for k,v in record['roles'].items()}
    names={'A-':'A−','A+':'A+','O':'O','C':'C'}
    group_roles={0:'A-',1:'O',2:'C',3:'A+'}
    assert all(roles[node]==group_roles[int(k)] for k,ids in group['groups'].items() for node in ids)
    with np.load(ROOT/'build/o3-certification/learned-basin-labels.npz') as d:
        boxes=d['boxes']; singleton=d['singleton']; image=d['image_cells']; paths=d['path_cells']
    cells=np.loadtxt(ROOT/'build/o3-certification/learned-morse-cells.csv',delimiter=',',ndmin=2)
    with np.load(folder/'grid-64.npz') as d:lower=d['lower'];upper=d['upper']
    included=np.zeros(len(boxes),bool);included[image]=True
    # Keep the stored one-step-image convention: no new basin computation.
    face=np.full(len(boxes),'white',dtype=object)
    for i,target in enumerate(singleton):
        role=roles.get(int(target))
        if included[i] and role in ('A-','A+'):
            face[i]=style.tint(O3_IDS[role])
    face[paths]=style.tint(O3_IDS['C'])
    face[~included]='#dddddd'
    import os, subprocess
    env=dict(os.environ, MPLBACKEND='Agg', JAX_ENABLE_X64='true')
    preimage=out/'physical-basin-preimages.npz'
    subprocess.run([str(ROOT/'.venv-emlp/bin/python'),str(ROOT/'src/o3_physical_preimages.py'),str(fit),str(preimage)],env=env,check=True,capture_output=True)
    with np.load(preimage) as d:a=d['a'];r=d['r'];q=d['encoded']
    side=(upper-lower)/64
    color_grid=np.full((64,64),'#dddddd',dtype=object)
    ij=np.floor(((boxes[:,:2]+boxes[:,2:])/2-lower)/side).astype(int)
    color_grid[ij[:,0],ij[:,1]]=face
    for b in cells:
        ijcell=np.floor(((b[:2]+b[2:4])/2-lower)/side).astype(int)
        color_grid[tuple(ijcell)]=O3_COLORS[roles[int(b[4])]]
    index=np.floor((q-lower)/side).astype(int)
    valid=np.all((index>=0)&(index<64),axis=-1)
    physical=np.full(q.shape[:2],'#dddddd',dtype=object)
    physical[valid]=color_grid[index[valid,0],index[valid,1]]
    from matplotlib.colors import to_rgba
    rgba=np.array([to_rgba(c) for c in physical.ravel()]).reshape(*physical.shape,4)
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained',gridspec_kw={'width_ratios':[1.3,1.3,1]})
        ax=axes[0];ax.set_facecolor('#dddddd')
        patches=[Rectangle(b[:2],*(b[2:]-b[:2])) for b in boxes]
        ax.add_collection(PatchCollection(patches,facecolors=face,edgecolors='none',antialiaseds=False))
        for b in cells:
            k=int(b[4]);ax.add_patch(Rectangle(b[:2],*(b[2:4]-b[:2]),facecolor=O3_COLORS[roles[k]],edgecolor='none'))
        for ids,label,offset in [([0],'A− · 0',(-25,28)),([1],'A+ · 1',(22,28)),([3,5],'O · 3, 5',(-42,40)),([2,4],'C · 2, 4',(25,24))]:
            selected=cells[np.isin(cells[:,4],ids)];center=(selected[:,:2].min(0)+selected[:,2:4].max(0))/2
            color=O3_COLORS[roles[ids[0]]]
            ax.annotate(label,center,xytext=offset,textcoords='offset points',color=color,
                        arrowprops={'arrowstyle':'-','color':color,'lw':.8})
        ax.set(xlim=(lower[0],upper[0]),ylim=(-.04,min(upper[1],cells[:,3].max()+.25)),xlabel='$q_1$',ylabel='$q_2$',title='Learned quotient: regions and native cells')
        ax=axes[1]
        ax.imshow(rgba,origin='lower',extent=[a[0],a[-1],r[0],r[-1]],aspect='auto',interpolation='nearest')
        ax.set(xlabel='Physical scalar $a$',ylabel='Physical radius $r$',title='Physical inverse images under the encoder')
        for x,y,label,role in [(-1,0,'A− · 0','A-'),(1,0,'A+ · 1','A+'),(0,0,'O · 3, 5','O'),(0,1,'C · 2, 4','C')]:
            ax.annotate(label,(x,y),xytext=(8,18),textcoords='offset points',color=O3_COLORS[role],fontsize=9)
        ax=axes[2];ax.axis('off');ax.set_title('Color and grouping key',loc='left')
        legend=[Patch(facecolor=style.tint(O3_IDS[r]),edgecolor=O3_COLORS[r],label=f'{names[r]}: native '+', '.join(str(k) for k,v in roles.items() if v==r)) for r in ('A-','A+','O','C')]
        legend += [Patch(facecolor='white',edgecolor='.6',label='White: unresolved in this display'),Patch(facecolor='#dddddd',label='Gray: outside one-step image cover')]
        ax.legend(handles=legend,loc='upper left',frameon=False,labelspacing=1.1,fontsize=10)
        ax.text(0,.27,'Dark cells: recurrent components\nPale blue/orange: single-sink reachability\nPale green: O-to-C connecting paths\n\n64 × 64 grid; saved checkpoint unchanged.',transform=ax.transAxes,va='top',fontsize=10,linespacing=1.6)
        for ext in ('png','pdf','svg'):fig.savefig(out/f'basin-group-guide.{ext}',dpi=180,bbox_inches='tight')
        plt.show()
        for kind in ('native','grouped'):
            graph=Digraph();graph.attr(dpi='180',ranksep='.25',nodesep='.3',bgcolor='white')
            if kind=='native':
                nodes={node['id']:roles[node['id']] for node in native['rectangle']['nodes']}
                labels={k:f'{k} · {names[r]}' for k,r in nodes.items()};edges=native['rectangle']['edges']
            else:
                nodes=group_roles;labels={k:f"{names[r]} · {{{', '.join(map(str,group['groups'][str(k)]))}}}" for k,r in nodes.items()};edges=group['edges']
            for k,r in nodes.items():graph.node(str(k),labels[k],style='filled',fillcolor=style.tint(O3_IDS[r]),color=O3_COLORS[r],fontname='Helvetica',fontsize='14')
            for a,b in edges:graph.edge(str(a),str(b))
            for ext in ('png','pdf','svg'):(out/f'{kind}-group-guide.{ext}').write_bytes(graph.pipe(format=ext))
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        images={kind:Image.open(out/f'{kind}-group-guide.png') for kind in ('native','grouped')}
        width=max(im.width for im in images.values());height=max(im.height for im in images.values())
        for ax,kind,title in zip(axes,['native','grouped'],['Six native components','Four groups, including connecting cells']):
            im=images[kind];ax.imshow(im,extent=(-im.width/2,im.width/2,im.height,0));ax.set(xlim=(-width/2,width/2),ylim=(height,0));ax.axis('off');ax.set_title(title,fontsize=11)
        for ext in ('png','pdf','svg'):fig.savefig(out/f'grouping-graphs.{ext}',dpi=180,bbox_inches='tight')
        plt.show()
    table(['Group','Native IDs','Cells after path closure','Recomputed rectangle index'],
          [(names[group_roles[int(k)]],str(ids),group['group_cell_counts'][k],str(tuple(group['indices'][k]))) for k,ids in group['groups'].items()])
    display(Markdown('**Grouping retains every native component.** O combines 3 and 5. C combines 2 and 4 and their connecting cells. '
        'The two sinks remain separate. CMGDB checks the grouped index pairs and recomputes their indices on the rectangular domain. '
        'The physical panel classifies each sampled (a,r) by E(a,r,0,0), using the same cell labels. Each point represents the full O(3) orbit with vector norm r, not a decoder output. The basin panel uses the saved restricted graph and one-step-image cover; its colors do not transfer rectangle indices to Atlas. '
        'Pale green paths are not a third attracting basin. White means unresolved here, not absence of attraction. '
        'These sampled graph regions are not certified basins of the continuous flow.'))
