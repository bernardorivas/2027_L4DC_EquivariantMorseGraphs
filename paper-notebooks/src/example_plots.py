"""Notebook figures and native CMGDB calls; no neural-network implementation."""
from contextlib import redirect_stdout
import io
import json
import numpy as np
from notebook_support import gallery, new_run, render_panels


def paper_reference(root, examples, data, system):
    """Recompute the released examples; save all six original paper panels."""
    import replay
    run = new_run(root,system)
    with redirect_stdout(io.StringIO()):
        checks = (replay.replay_sphere if system=='sphere' else replay.replay_o3)(
            data/system,run/system,recompute_explicit=True)
        figures = render_panels(examples,run,system)
    (run/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    return figures


def show_analytical(reference, system):
    if system == 'sphere':
        from sphere_flow_figure import render
        panels = render()
        gallery([('', panels/'sphere-flow-comparison.svg')])
    else:
        from o3_flow_figure import render
        panels = render()
        gallery([('', panels/'quotient-flow.svg')])


def learned_cmgdb(run):
    """Actual CMGDB graph of sampled image hulls; exploratory, NOT a certificate."""
    import CMGDB
    import matplotlib.pyplot as plt
    import figure_style as style
    result = json.loads((run/'result.json').read_text())
    sphere = result['config']['system']=='sphere'
    with np.load(run/'evaluation.npz') as data:
        boxes, images = data['boxes'],data['image_boxes']
        lower, upper = data['domain_lower'].tolist(),data['domain_upper'].tolist()
    lookup = {tuple(np.round(box,12)): image.tolist() for box,image in zip(boxes,images)}
    assert len(lookup)==len(boxes)
    def box_map(box): return lookup[tuple(np.round(box,12))]
    depth = result['config']['grid_depth']*(1 if sphere else 2)
    model = CMGDB.Model(depth,depth,depth,10000,lower,upper,box_map)
    model.set_batch_map(lambda batch: [box_map(box) for box in batch])
    morse, phase = CMGDB.ComputeConleyMorseGraph(model)
    assert phase.num_vertices()==len(boxes)
    graph = CMGDB.PlotMorseGraph(morse,clist=style.MORSE)
    (run/'learned-morse.dot').write_text(graph.source)
    for format in ('svg','pdf','png'):
        (run/f'learned-morse.{format}').write_bytes(graph.pipe(format=format))
    CMGDB.SaveMorseSets(morse,str(run/'learned-morse-cells.csv'))
    if sphere:
        fig,ax = CMGDB.PlotMorseSets1D(morse,clist=style.MORSE,xlabel='$\\zeta$',label_sets=True,
                                      axis_arrow=False,fig_w=7,fig_h=1.5,show=False)
    else:
        fig,ax = CMGDB.PlotMorseSets(morse,clist=style.MORSE,xlabel='$a_\\ell$',ylabel='$r_\\ell$',
                                   xlim=lower[0:1]+upper[0:1],ylim=lower[1:2]+upper[1:2],fig_w=5,fig_h=4,show=False)
    fig.savefig(run/'learned-morse-cells.pdf',bbox_inches='tight')
    fig.savefig(run/'learned-morse-cells.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    dim=len(lower)
    record = dict(certified=False,input_method=result['cmgdb_input'],domain_lower=lower,domain_upper=upper,
                  native_nodes=len(list(morse.vertices())),edges=[list(e) for e in morse.edges()],
                  cells=phase.num_vertices(),outgoing_image_cells=int(np.any((images[:,:dim]<lower)|(images[:,dim:]>upper),axis=1).sum()),
                  nodes=[dict(node=int(n),annotations=list(morse.annotations(n)),cell_count=len(morse.morse_set(n))) for n in morse.vertices()],
                  checkpoint_sha256=result['checkpoint_sha256'])
    (run/'cmgdb.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
