"""Native analytical time-ten CMGDB reference beside the joint learned map."""
from pathlib import Path
import json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def analytical_graphs(folder,levels=(7,8,9),tau=10.):
    import CMGDB
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for level in levels:
        n=2**level;lower=np.array([-1.5,0.]);upper=np.array([1.5,1.5]);side=(upper-lower)/n
        # Same 5-by-5 source sampling and 1e-9 inflation as the learned comparison.
        a,r=np.meshgrid(np.linspace(-1.5,1.5,4*n+1),np.linspace(0,1.5,4*n+1),indexing='ij')
        t=np.exp(2*tau)-1;d=1+t*a*a
        values=np.stack((np.exp(tau)*a/np.sqrt(d),np.exp(tau)*r/np.sqrt(d*(d+t*r*r))),axis=-1)
        lo=np.full((n,n,2),np.inf);hi=-lo.copy()
        for i,j in itertools.product(range(5),repeat=2):
            v=values[i:4*n+i:4,j:4*n+j:4];np.minimum(lo,v,out=lo);np.maximum(hi,v,out=hi)
        full=np.concatenate((lo-1e-9,hi+1e-9),axis=-1);full[...,1]=np.maximum(0,full[...,1])
        def batch(boxes):
            b=np.asarray(boxes);ij=np.floor(((b[:,:2]+b[:,2:])/2-lower)/side).astype(int)
            return full[ij[:,0],ij[:,1]].tolist()
        model=CMGDB.Model(2*level,2*level,2*level,10000,lower.tolist(),upper.tolist(),lambda b:batch([b])[0]);model.set_batch_map(batch)
        morse,phase=CMGDB.ComputeConleyMorseGraph(model);graph=CMGDB.PlotMorseGraph(morse)
        for ext in ('svg','png','pdf'):(folder/f'analytical-{n}.{ext}').write_bytes(graph.pipe(format=ext))
        (folder/f'analytical-{n}.dot').write_text(graph.source)
        rows.append(dict(n=n,components=len(list(morse.vertices())),edges=[list(e) for e in morse.edges()],
            nodes=[dict(id=int(k),annotations=list(morse.annotations(k)),cells=len(morse.morse_set(k))) for k in morse.vertices()]))
    result=dict(time=tau,samples_per_axis=5,inflation=1e-9,certified=False,resolutions=rows)
    (folder/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');return result


def show_joint_comparison(reference,fit,study,diagnostics=True):
    from notebook_support import gallery,table
    from quotient_domain_notebook import _domain_figure
    from IPython.display import display,HTML
    folder=ROOT/study['study'];dest=folder/'analytical-matched-time'
    cfg=json.loads((fit/'config.json').read_text());tau=cfg['time_step']*cfg['cmgdb_iterations']
    time_label=f'{tau:.4g}'
    result=json.loads((folder/'comparison.json').read_text());levels=[int(np.log2(r['n'])) for r in result['resolutions']]
    if not (dest/'comparison.json').exists():analytical_graphs(dest,levels,tau=tau)
    exact=json.loads((dest/'comparison.json').read_text())
    table(['Grid per axis',f'Analytical time {time_label}','Learned rectangle','Learned encoder image'],
        [(a['n'],a['components'],b['rectangle']['components'],b['restricted']['components']) for a,b in zip(exact['resolutions'],result['resolutions'])])
    n=result['resolutions'][-1]['n']
    from o3_morse_regions import show_regions
    show_regions(fit)
    from o3_aggregation import show_aggregation
    show_aggregation(fit,show_graphs=False,show_placement=False)
    if diagnostics:show_joint_diagnostics(fit,study)
    fit_result=json.loads((fit/'result.json').read_text())
    record=dict(fit=str(fit.relative_to(ROOT)),checkpoint_sha256=fit_result['checkpoint_sha256'],
        metrics=fit_result['metrics'],domain_study=study,reference_figures=str(reference.relative_to(ROOT)),
        analytical_matched_time=str(dest.relative_to(ROOT)),synthetic_observations=False)
    (ROOT/'build/notebooks/o3/latest.json').write_text(json.dumps(record,indent=2)+'\n')

def show_joint_diagnostics(fit, study):
    """Show actual cell geometry, shared node IDs, and domain-specific indices."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from graphviz import Digraph
    from PIL import Image
    from IPython.display import display, Markdown
    from notebook_support import table
    folder = ROOT / study['study']
    checks = ROOT / study['checks']
    result = json.loads((folder / 'comparison.json').read_text())
    last = result['resolutions'][-1]
    nodes = last['restricted']['nodes']
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#756BB1', '#8C6D31']
    c = result['point_checks']
    display(Markdown(f"**Sampled domain check:** {c['one_step_outside']} of {c['samples']:,} "
                     "encoded starts leave the sampled image after one learned step. "
                     "The outline below is the sampled encoder boundary, not a basin boundary."))
    with np.load(folder / 'domain.npz') as data:
        boundary = data['boundary']
    roots = c['roots']
    graph = Digraph()
    graph.attr(rankdir='TB', bgcolor='white', nodesep='.3', ranksep='.3', dpi='180')
    for node in nodes:
        k = node['id']
        graph.node(str(k), str(k), style='filled', fillcolor=colors[k]+'33',
                   color=colors[k], fontname='Helvetica', fontsize='14', width='.4', height='.3')
    for u,v in last['restricted']['edges']:
        graph.edge(str(u),str(v))
    for ext in ('svg','pdf','png'):
        (checks / f'native-topology.{ext}').write_bytes(graph.pipe(format=ext))
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
        ax = axes[0,0]
        ax.add_patch(Polygon(boundary,facecolor='#f0f3f5',edgecolor='.5',lw=.8))
        for node in nodes:
            k=node['id']; lo=np.array(node['lower']); hi=np.array(node['upper'])
            center=(lo+hi)/2
            ax.scatter(*center,s=22,color=colors[k])
        for root, name in zip(roots, ['left sink','origin','upper saddle','right sink']):
            ax.annotate(name,root['q'],xytext=(-40,9) if name=='right sink' else (6,9),textcoords='offset points',fontsize=8)
        ax.autoscale_view(); ax.set(title='Locations in the encoder image',xlabel='$q_1$',ylabel='$q_2$')
        # Magnify each neighborhood without enlarging the cells in data coordinates.
        for ax, root, title in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1]],
                                  roots,['Left sink','Origin: adjacent components','Upper saddle: adjacent components','Right sink']):
            q=np.array(root['q']); half=.025
            for node in nodes:
                lo=np.array(node['lower']); hi=np.array(node['upper']); k=node['id']
                if np.all(hi>=q-half) and np.all(lo<=q+half):
                    ax.add_patch(Rectangle(lo,*(hi-lo),facecolor=colors[k]+'55',edgecolor=colors[k],lw=1.5))
                    ax.text(*(lo+hi)/2,str(k),ha='center',va='center',fontsize=11)
            ax.plot(*q,marker='+',color='black',ms=9,mew=1.3)
            ax.set(xlim=(q[0]-half,q[0]+half),ylim=(max(-.003,q[1]-half),q[1]+half),
                   title=title,xlabel='$q_1$',ylabel='$q_2$')
            ax.tick_params(labelsize=8);ax.ticklabel_format(useOffset=False)
        ax=axes[1,2];ax.imshow(Image.open(checks/'native-topology.png'));ax.axis('off')
        ax.set_title('Restricted graph: same cell IDs')
        for ext in ('png','pdf','svg'):
            fig.savefig(checks/f'native-cell-guide.{ext}',dpi=170,bbox_inches='tight')
        plt.show()
    display(Markdown("**How to read this figure.** Colored rectangles are the actual recurrent cells, "
        "not point markers. Their numbers match the directed graph. A black + marks a numerically found "
        "fixed point. Adjacent components without a + remain distinct native components. "
        "The close-ups change the axis scale, not the computed cells."))
    same_edges = sorted(last['restricted']['edges']) == sorted(last['rectangle']['edges'])
    same_cells = all(a['id']==b['id'] and a['lower']==b['lower'] and a['upper']==b['upper']
                     for a,b in zip(nodes,last['rectangle']['nodes']))
    display(Markdown(f"At {last['n']} × {last['n']}, the two computations have "
        + ("the same node cells and directed edges. " if same_edges and same_cells else "different native graph data. ")
        + "The restricted domain uses Atlas and has no index annotations. The table reports only the rectangle computation's indices."))
    table(['Cell ID','Restricted cell count','Contains a found fixed point','Rectangle index (degrees 0, 1, 2)'],
          [(a['id'],a['cell_count'],'yes' if a['roots'] else 'no',str(tuple(b['annotations'])))
           for a,b in zip(nodes,last['rectangle']['nodes'])])

if __name__=='__main__':
    import sys
    analytical_graphs(Path(sys.argv[1]))
