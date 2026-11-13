"""One notebook call for fixed-model domain restriction and a compact comparison."""
from pathlib import Path
import json,tempfile
import numpy as np
from IPython.display import HTML,display
from IPython.utils.capture import capture_output
from notebook_support import gallery,table
from quotient_domain import ROOT,identity,digest


def register_domain_study(fit,folder,checks):
    result=json.loads((folder/'comparison.json').read_text())
    levels=[int(np.log2(r['n'])) for r in result['resolutions']]
    assert result['identity']==identity(fit,levels)
    assert (checks/'verification.json').is_file()
    record=dict(study=str(folder.relative_to(ROOT)),checks=str(checks.relative_to(ROOT)),
                identity=result['identity'],checks_code=digest(ROOT/'src/quotient_domain_checks.py'))
    (ROOT/f"build/quotient-domain/{result['system']}-latest.json").write_text(json.dumps(record,indent=2)+'\n')
    return record


def quotient_domain_study(fit,levels=None,recompute=False):
    cfg=json.loads((fit/'config.json').read_text());system=cfg['system']
    levels=tuple(levels or ((7,8,9,10) if system=='sphere' else (7,8,9)))
    parent=ROOT/'build/quotient-domain';parent.mkdir(parents=True,exist_ok=True)
    pointer=parent/f'{system}-latest.json'
    if pointer.exists() and not recompute:
        record=json.loads(pointer.read_text())
        if record['identity']==identity(fit,levels) and record['checks_code']==digest(ROOT/'src/quotient_domain_checks.py'):
            return record
    from quotient_domain import compute
    from quotient_domain_checks import verify
    if not (fit/'cmgdb.json').exists():
        from example_plots import learned_cmgdb
        with capture_output():learned_cmgdb(fit)
    folder=Path(tempfile.mkdtemp(prefix=system+'-',dir=parent));checks=folder/'checks';checks.mkdir()
    with capture_output():
        compute(fit,folder,list(levels));verify(folder,checks)
    return register_domain_study(fit,folder,checks)


def _domain_figure(folder,checks,result):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Rectangle
    sphere=result['system']=='sphere';last=result['resolutions'][-1]
    with np.load(folder/'domain.npz') as data:
        boundary=data['boundary'];encoded=data['encoded'];images=data['encoded_next'];inside=data['step_inside']
    roots=result['point_checks']['roots']
    with plt.rc_context({'font.family':'serif','mathtext.fontset':'cm','font.size':11}):
        if sphere:
            fig,ax=plt.subplots(figsize=(6,2.3),layout='constrained')
            ax.plot(encoded[:,0],images[:,0]-encoded[:,0],color='#0072B2')
            ax.axhline(0,color='.5',lw=.7)
            ax.scatter([r['q'][0] for r in roots],np.zeros(len(roots)),s=22,color='black')
            ax.set(xlabel='$q$',ylabel='$f(q)-q$',title='Fixed points',xlim=(-1,1))
        else:
            fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
            ax=axes[0]
            ax.add_patch(Polygon(boundary,facecolor='#e3edf5',edgecolor='.3',lw=.8))
            ax.scatter(encoded[~inside,0],encoded[~inside,1],s=2,color='#D55E00',label='One-step exit starts')
            for valid,marker,label in [(True,'o','Fixed point in image'),(False,'x','Exterior fixed points')]:
                q=np.array([r['q'] for r in roots if r['inside']==valid])
                if len(q):ax.scatter(q[:,0],q[:,1],s=40,c='black',marker=marker,label=label)
            ax.autoscale_view();ax.set(xlabel='$q_1$',ylabel='$q_2$',title='Encoder image')
            ax.legend(frameon=False,fontsize=8)
            ax=axes[1]
            cells=np.loadtxt(folder/f"restricted-{last['n']}-cells.csv",delimiter=',',ndmin=2)
            palette=['#1f77b4','#e6550d','#31a354','#d62728','#9467bd','#8c564b']
            for x0,y0,x1,y1,k in cells:
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=palette[int(k)%len(palette)],alpha=.65,edgecolor='white',lw=.5))
            q=np.array([r['q'] for r in roots if r['inside']])
            ax.scatter(q[:,0],q[:,1],s=28,c='black',zorder=3)
            ax.autoscale_view();ax.set(xlabel='$q_1$',ylabel='$q_2$',title='Restricted Morse cells')
        for ext in ('pdf','png'):fig.savefig(checks/f'domain-diagnostic.{ext}',dpi=180,bbox_inches='tight')
        display(fig);plt.close(fig)


def show_domain_comparison(reference,fit,study,show_graphs=True):
    folder=ROOT/study['study'];checks=ROOT/study['checks']
    result=json.loads((folder/'comparison.json').read_text())
    verified=json.loads((checks/'verification.json').read_text())
    sphere=result['system']=='sphere';rows=result['resolutions'];last=rows[-1]
    table(['Grid','Rectangle components','Encoder-image components'],
          [(str(r['n']) if sphere else f"{r['n']} × {r['n']}",r['rectangle']['components'],r['restricted']['components']) for r in rows])
    analytical=reference/('morse/explicit-graph.png' if sphere else 'explicit-graph.png')
    if sphere:
        learned=folder/f"aggregate-{last['n']}.svg"
        title='Learned quotient (grouped)'
    else:
        aggregate=verified.get('local_aggregate')
        learned=checks/f"aggregate-{last['n']}.svg" if aggregate else folder/f"restricted-{last['n']}.svg"
        title='Restricted learned quotient (local group)' if aggregate else 'Restricted learned quotient'
    if show_graphs:
        gallery([('Analytical quotient'+('' if sphere else ' (grouped)'),analytical),(title,learned)],fit_height=340,
                image_max_width={title:230} if not sphere and aggregate else None)
    if sphere:
        match=all(r['aggregate']['matches_analytical'] for r in rows)
        display(HTML('<p><b>'+('The grouped graphs and indices agree on every tested grid.' if match else
            'The grouped graphs or indices differ from the analytical reference.')+'</b> '
            f"Point-map checks find {len(result['point_checks']['roots'])} fixed points. "
            'Local grouping includes connecting cells and recomputes their indices. '
            'The native graphs remain available below.</p>'))
    else:
        c=result['point_checks']
        display(HTML(f"<p>The restricted graph has {last['restricted']['components']} native components at the finest grid; "
            'the analytical grouped graph has four. '
            f"{c['one_step_outside']} of {c['samples']} sampled encoded states leave the encoder image in one step. "
            f"{sum(r['inside'] for r in c['roots'])} of {len(c['roots'])} numerically found fixed points lie inside the image. "
            +('The nearby components form one checked interior group with a recomputed index. ' if aggregate else '')+
            'A matching node count alone does not establish matching dynamics.</p>'))
    display(HTML('<p>These are numerical results for the fixed checkpoint and a sampled encoder image. '
        'Escaping images are not projected back. Graph agreement and sampled drift do not certify the original ODE.</p>'))
    with capture_output() as captured:
        _domain_figure(folder,checks,result)
        for i in range(0,len(rows),2):
            gallery([(f"Native restricted graph: {r['n']}"+('' if sphere else '²'),
                folder/f"{'rectangle' if sphere else 'restricted'}-{r['n']}.svg") for r in rows[i:i+2]],max_height=450)
        if not sphere:
            display(HTML('<p>The native Atlas restriction supplies Morse graphs without index annotations. '
                'The displayed local group index is recomputed using the ordinary CMGDB index-pair routine on identical interior cell images. '
                'The graph was independently checked by SciPy strongly connected components.</p>'))
            table(['Fixed point','In encoder image','Residual','Distance to image (numerical inversion)'],
                [('Saddle' if r['unstable_directions']==1 else 'Attractor',r['inside'],f"{r['residual']:.2g}",f"{r['inverse_check']['residual']:.4g}") for r in result['point_checks']['roots']])
    html=''.join(item.data.get('text/html','') for item in captured.outputs)
    # capture_output stores Matplotlib PNG as image/png, not HTML.
    diagnostic=checks/'domain-diagnostic.png'
    with capture_output() as captured:
        gallery([('Point-map and domain checks',diagnostic)])
    image_html=''.join(item.data.get('text/html','') for item in captured.outputs)
    display(HTML('<details><summary>Native graphs and domain checks</summary>'+image_html+html+'</details>'))
    latest=ROOT/f"build/notebooks/{result['system']}/latest.json"
    record={'metrics':json.loads((fit/'result.json').read_text()).get('metrics',{}),
            'learned_cmgdb':json.loads((fit/'cmgdb.json').read_text()) if (fit/'cmgdb.json').exists() else {}}
    record.update(experiment='raw-state direct quotient',fit=str(fit.relative_to(ROOT)),
        reference_figures=str(reference.relative_to(ROOT)),checkpoint_sha256=result['identity']['inputs']['models.npz'],
        domain_study=study,synthetic_observations=False)
    latest.write_text(json.dumps(record,indent=2)+'\n')
