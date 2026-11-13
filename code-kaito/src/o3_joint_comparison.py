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

def show_joint_diagnostics(fit,study):
    """Display sampled image membership and the two native graph domains."""
    from notebook_support import gallery
    from quotient_domain_notebook import _domain_figure
    from IPython.display import display,HTML
    folder=ROOT/study['study']
    result=json.loads((folder/'comparison.json').read_text())
    n=result['resolutions'][-1]['n']
    c=result['point_checks']
    display(HTML(f"<p><b>Domain check:</b> {c['initial_outside']} of {c['samples']} encoded starts lie outside the sampled image; "
        f"{c['one_step_outside']} leave it after one learned step. "
        "These sampled cell hulls are numerical evidence, not certified enclosures. The unaggregated graphs remain available below.</p>"))
    checks=ROOT/study['checks']
    _domain_figure(folder,checks,result)
    gallery([('Encoder image: native graph, no indices',folder/f'restricted-{n}.svg'),
             ('Rectangle: native graph and indices',folder/f'rectangle-{n}.svg')],fit_height=420)
    display(HTML('<p>Rectangle indices include cells outside the sampled encoder image and are not transferred to the Atlas restriction.</p>'))

if __name__=='__main__':
    import sys
    analytical_graphs(Path(sys.argv[1]))
