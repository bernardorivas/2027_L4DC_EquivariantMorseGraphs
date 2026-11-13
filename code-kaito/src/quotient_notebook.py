"""Compact views of raw-state quotient training and native CMGDB results."""
from pathlib import Path
import json
import numpy as np
from IPython.display import display, HTML, Image
from IPython.utils.capture import capture_output
from notebook_support import gallery, table

ROOT=Path(__file__).resolve().parents[1]


def show_quotient_fit(run):
    import matplotlib.pyplot as plt
    result=json.loads((run/'result.json').read_text())
    sphere=result['config']['system']=='sphere'
    with np.load(run/'evaluation.npz') as data:
        d={k:data[k] for k in data.files}
    with plt.rc_context({'font.family':'serif','mathtext.fontset':'cm','font.size':12}):
        fig,axes=plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
        physical,encoded=d['physical_quotient'],d['encoded_quotient']
        if sphere:
            axes[0].plot(physical[:,0],encoded[:,0],color='#0072B2')
            axes[0].set(xlabel='$z$',ylabel='$q$',title='Learned quotient encoder')
            axes[1].plot(d['quotient_grid'][:,0],d['quotient_images'][:,0],color='#D55E00',label='Learned')
            axes[1].plot(encoded[:,0],d['encoded_next'][:,0],'--',color='.3',label='Encoded ODE')
            axes[1].plot([-1,1],[-1,1],':',color='.65')
            axes[1].set(xlabel='$q$',ylabel='$f^{'+str(result['quotient_iterations'])+'}(q)$',title='Quotient dynamics')
            axes[1].legend(frameon=False,fontsize=9)
            truth=d['raw_states'][:,2:3]
        else:
            grid=encoded.reshape(13,19,2)
            for line in list(grid)+list(grid.transpose(1,0,2)):
                axes[0].plot(line[:,0],line[:,1],color='.75',lw=.6)
            points=axes[0].scatter(encoded[:,0],encoded[:,1],c=physical[:,0],cmap='coolwarm',s=5,vmin=-1.5,vmax=1.5)
            fig.colorbar(points,ax=axes[0],label='$a$',shrink=.8)
            axes[0].set(xlabel='$q_1$',ylabel='$q_2$',title='Learned quotient encoder')
            q,image=d['quotient_grid'],d['quotient_images']
            delta=image-q
            axes[1].quiver(q[:,0],q[:,1],delta[:,0],delta[:,1],angles='xy',scale_units='xy',color='#D55E00',width=.004)
            axes[1].set(xlabel='$q_1$',ylabel='$q_2$',title='Quotient dynamics')
            truth=np.column_stack((d['raw_states'][:,0],np.linalg.norm(d['raw_states'][:,1:],axis=1)))
        for i in range(truth.shape[1]):
            axes[2].scatter(truth[:,i],d['decoded_quotient'][:,i],s=4,alpha=.4,label=('$a$' if i==0 else '$r$'))
        lo=min(truth.min(),d['decoded_quotient'].min());hi=max(truth.max(),d['decoded_quotient'].max())
        axes[2].plot([lo,hi],[lo,hi],'--',color='.35',lw=1)
        axes[2].set(xlabel='True orbit coordinate',ylabel='Decoded',title='Reconstruction up to symmetry')
        if not sphere:axes[2].legend(frameon=False,fontsize=9)
        for ext in ('pdf','png'):fig.savefig(run/f'quotient-fit.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig);display(Image(filename=str(run/'quotient-fit.png')))
    metrics=result['metrics']
    table(['Held-out orbit RMS / state scale','Reconstruction','One step','Ten steps'],
          [('Raw states',*[f'{100*metrics[k]:.2f}%' if metrics[k] is not None else 'Nonfinite' for k in
             ('reconstruction_orbit_rms','prediction_orbit_rms','rollout_orbit_rms')])])
    display(HTML('<p>The error is the minimum Euclidean state distance over rotations, divided by '
                 +('the sphere radius 1.' if sphere else 'the state scale 1.5. Arrows use a common display scale.')+'</p>'))


def show_quotient_cmgdb(reference,run):
    result=json.loads((run/'result.json').read_text())
    cmg=json.loads((run/'cmgdb.json').read_text())
    sphere=result['config']['system']=='sphere'
    graph=reference/('morse/explicit-graph.png' if sphere else 'explicit-graph.png')
    gallery([('Analytical quotient'+('' if sphere else ' (grouped)'),graph),
             ('Direct EMLP quotient — sampled',run/'learned-morse.png')],
            max_height=550,fit_height=550 if cmg['native_nodes']<=8 else None)
    display(HTML(f"<p>The learned graph has {cmg['native_nodes']} native components. "
                 f"{cmg['outgoing_image_cells']} of {cmg['cells']} sampled cell images leave the grid domain. "
                 'These annotations belong to the sampled learned map; agreement with the original ODE is not certified.</p>'))
    expected=5 if sphere else 4
    if cmg['native_nodes']!=expected:
        display(HTML(f'<p><b>Comparison:</b> the displayed analytical graph has {expected} nodes'
                     +(' (after its documented grouping)' if not sphere else '')
                     +'. This direct fit does not yet reproduce that displayed graph. Extra nodes have not been removed or declared artifacts.</p>'))
    if not sphere:
        display(HTML('<p>The grid rectangle contains sampled encodings. Its points outside the encoded state domain are a learned extension, '
                     'so its complete Morse graph need not describe only physically realizable encoded states.</p>'))
    with capture_output() as captured:
        if sphere:
            files=[('Sphere','flow/sphere.png'),('Height map','flow/height-map.png'),('Analytical cells','morse/explicit-cells.png'),('Paper-fit cells','morse/learned-cells.png'),('Analytical graph','morse/explicit-graph.png'),('Paper-fit graph','morse/learned-graph.png')]
        else:
            files=[('Quotient flow','quotient-flow.png'),('Analytical cells','explicit-cells.png'),('Paper-fit cells','learned-cells.png'),('Analytical graph','explicit-graph.png'),('Paper-fit graph','learned-graph.png'),('Full-space index','original-graph.png')]
        for i in range(0,len(files),2):gallery([(title,reference/file) for title,file in files[i:i+2]])
    html=''.join(item.data.get('text/html','') for item in captured.outputs)
    display(HTML('<details><summary>Original paper panels — historical reference, separate fits</summary>'+html+'</details>'))
    record=dict(experiment='raw-state direct quotient',fit=str(run.relative_to(ROOT)),
        reference_figures=str(reference.relative_to(ROOT)),checkpoint_sha256=result['checkpoint_sha256'],
        metrics=result['metrics'],learned_cmgdb=cmg,synthetic_observations=False)
    (ROOT/'build/notebooks'/result['config']['system']/'latest.json').write_text(json.dumps(record,indent=2)+'\n')
