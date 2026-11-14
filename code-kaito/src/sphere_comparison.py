"""Three-row sphere comparison from saved cells and computed quotient indices."""
from pathlib import Path
import json,re,subprocess,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NAMES={0:'S',1:'A−',2:'C',3:'A+',4:'N'}
from paper_figure_style import style, SPHERE_IDS, SPHERE_COLORS as COLORS, save_graph, size_panel_title

def show_sphere_comparison(fit,study):
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    from IPython.utils.capture import capture_output
    from notebook_support import gallery
    from sphere_pullback import show_sphere_pullback, draw_transparent_sphere
    from graphviz import Source
    import CMGDB
    from CMGDB.morse_lattice import DerivedGraph
    import xml.etree.ElementTree as ET
    folder=ROOT/study['study']
    with capture_output(): evidence=show_sphere_pullback(fit,study)
    cert_path=ROOT/'build/sphere-certification/local-certificate.json'
    cert=json.loads(cert_path.read_text()) if cert_path.exists() else {}
    verified=cert.get('checkpoint_sha256')==evidence['checkpoint_sha256'] and cert.get('all_local_inequalities_verified',False)
    comparison=json.loads((folder/'comparison.json').read_text())
    row=next(r for r in comparison['resolutions'] if r['n']==128)
    agg=row['aggregate'];assert agg['matches_analytical']
    assignment={node:int(k) for k,vs in agg['groups'].items() for node in vs}
    exact=ROOT/'vendor/latent-equivariant/code/examples/data/sphere/exact-K128'
    learned=[dict(z=r['z'],q=r['q'],node=assignment[r['node']]) for r in evidence['cells']]
    render_replay_comparison(folder, learned, agg)
    expected_q={int(k):v for k,v in agg['annotations'].items()}
    reconstructed={0:['0','0','x-1'],1:['x-1','x-1','0'],2:['0','x-1','x-1'],3:['x-1','x-1','0'],4:['0','0','x-1']}
    (folder/'sphere-reconstruction.json').write_text(json.dumps(dict(checkpoint_sha256=evidence['checkpoint_sha256'],quotient=expected_q,reconstructed=reconstructed,status='Verified local indices of the specified sphere lift' if verified else 'Conditional local-model reconstruction',certificate=str(cert_path.relative_to(ROOT)) if verified else None),indent=2)+'\n')
    gallery([('',folder/'sphere-comparison.svg')],max_height=1150)
    display(Markdown('**128 interval cells.** Colored bands are encoder preimages of native Morse cells. The graph rows show recomputed aggregates and the local indices of the specified sphere lift.'))
    return evidence


def render_replay_comparison(folder, learned, agg):
    """Render the original comparison layout from the current replay's cells.

    Quotient indices and edges are checked against the saved analytical graph.
    Sphere indices below are supplied local-model tuples, not a new computation.
    """
    import matplotlib.pyplot as plt
    import xml.etree.ElementTree as ET
    import CMGDB
    from CMGDB.morse_lattice import DerivedGraph
    from sphere_pullback import draw_transparent_sphere

    folder = Path(folder)
    exact = ROOT/'vendor/latent-equivariant/code/examples/data/sphere/exact-K128'
    physical_to_role={1:0,0:1,3:2,2:3,4:4}
    explicit=[dict(z=[lo,hi],node=physical_to_role[int(k)]) for lo,hi,k in np.loadtxt(exact/'morse-sets.csv',delimiter=',',ndmin=2)]
    # The pole fibers collapse; interior circle fibers add a degree-one copy.
    # These are local normal-model reconstructions, not a new learned-map certificate.
    reconstructed={0:['0','0','x-1'],1:['x-1','x-1','0'],2:['0','x-1','x-1'],3:['x-1','x-1','0'],4:['0','0','x-1']}
    expected_q={0:['0','0'],1:['x-1','0'],2:['0','x-1'],3:['x-1','0'],4:['0','0']}
    assert {int(k):v for k,v in agg['annotations'].items()}==expected_q
    exact_dot=(exact/'morse-graph.dot').read_text()
    exact_edges=[[physical_to_role[int(a)],physical_to_role[int(b)]] for a,b in re.findall(r'(\d+) -> (\d+)',exact_dot)]
    assert set(map(tuple,exact_edges))==set(map(tuple,agg['edges']))
    elev=np.deg2rad(12)
    for case,rows in [('analytical',explicit),('learned',learned)]:
        with plt.rc_context({'font.family':'serif','mathtext.fontset':'cm','font.size':13}):
            fig,(ax,interval)=plt.subplots(2,1,figsize=(5,4.1),gridspec_kw={'height_ratios':[4.2,1]})
            draw_transparent_sphere(ax,rows,COLORS)
            for k in NAMES:
                bands=[r['z'] for r in rows if r['node']==k]
                z=(min(b[0] for b in bands)+max(b[1] for b in bands))/2
                if k in (0,4):z=-1 if k==0 else 1
                # Transparent view: both poles are visible; callouts target the front.
                longitude={1:1.02,2:1.13,3:1.00}.get(k,0.)
                depth=np.sqrt(max(0,1-z*z))*np.cos(longitude)*np.cos(elev)+z*np.sin(elev)
                point=(np.sqrt(max(0,1-z*z))*np.sin(longitude),z*np.cos(elev)-np.sqrt(max(0,1-z*z))*np.cos(longitude)*np.sin(elev))
                label={1:r'$A_-$',3:r'$A_+$'}.get(k,'$'+NAMES[k]+'$')
                target={1:(1.23,point[1]-.22),2:(1.26,point[1]+.16),3:(1.20,point[1]+.28)}.get(k,(0,1.13 if k==4 else -1.13))
                tip=point
                if k in (1,2,3):
                    # End in the uncolored gap, just beyond the approached band edge.
                    edge=min(b[0] for b in bands)-.025 if k==1 else max(b[1] for b in bands)+.025
                    tip=(np.sqrt(max(0,1-edge*edge))*np.sin(longitude),edge*np.cos(elev)-np.sqrt(max(0,1-edge*edge))*np.cos(longitude)*np.sin(elev))
                ax.annotate(label,xy=tip,xytext=target,zorder=10,color=COLORS[k],fontsize=14,ha='center',va='center',arrowprops={'arrowstyle':'->' if k in (1,2,3) else '-', 'connectionstyle':f'arc3,rad={ {1:-.38,2:.36,3:.42}.get(k,0)}','color':COLORS[k],'lw':.8,'mutation_scale':9,'shrinkA':4,'shrinkB':1})
                if k in (0,4):
                    ax.plot(*point,marker='o',linestyle='none',ms=3,
                            markerfacecolor='none' if depth<0 else COLORS[k],
                            markeredgecolor=COLORS[k],markeredgewidth=.7,
                            zorder=1 if depth<0 else 5,alpha=.45 if depth<0 else 1)
            ax.set(aspect='equal',xlim=(-1.2,1.45),ylim=(-1.25,1.25),title='Analytical' if case=='analytical' else 'Learned');ax.axis('off')
            interval.axhline(0,color='.65',lw=1,zorder=0)
            for r in rows:
                lo,hi=r['z'] if case=='analytical' else r['q']
                interval.fill_between([lo,hi],-.13,.13,color=COLORS[r['node']],alpha=.8)
            for k in NAMES:
                bands=[(r['z'] if case=='analytical' else r['q']) for r in rows if r['node']==k]
                center=(min(b[0] for b in bands)+max(b[1] for b in bands))/2
                label={1:r'$A_-$',3:r'$A_+$'}.get(k,'$'+NAMES[k]+'$')
                interval.text(center,.2,label,color=COLORS[k],ha='center',fontsize=11)
            interval.set(xlim=(-1.06,1.06),ylim=(-.35,.55),yticks=[],xticks=[-1,0,1],xlabel='$z$' if case=='analytical' else r'$q=E_\theta(p)$')
            for spine in ('left','right','top'):interval.spines[spine].set_visible(False)
            interval.spines['bottom'].set_position(('data',-.16))
            interval.spines['bottom'].set_visible(False)
            interval.tick_params(axis='x',length=0,pad=2,labelsize=11)
            # Explicit panel positions avoid the large sphere/interval gap.
            ax.set_position([.04,.23,.92,.70])
            interval.set_position([.08,.095,.84,.14])
            size_panel_title(fig,ax)
            ax.title.set_fontsize(ax.title.get_fontsize() * .485/.415)
            for ext in ('svg','png','pdf'):fig.savefig(folder/f'{case}-sphere-sets.{ext}',dpi=180)
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            for panel,axes in [('sphere',ax),('interval',interval)]:
                bounds=axes.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
                other=interval if axes is ax else ax
                other.set_visible(False)
                for ext in ('pdf','svg'):
                    fig.savefig(folder/f'{case}-{panel}-panel.{ext}',bbox_inches=bounds,pad_inches=.02)
                other.set_visible(True)

            plt.close(fig)
        for suffix,title,indices in [('quotient','Quotient',expected_q if case=='analytical' else {int(k):v for k,v in agg['annotations'].items()}),('reconstructed','Sphere lift (local-model indices)',reconstructed)]:
            edges=exact_edges if case=='analytical' else agg['edges']
            graph=DerivedGraph(nodes=list(NAMES),edges={k:[b for a,b in edges if a==k] for k in NAMES},labels={k:'('+', '.join(indices[k])+')' for k in NAMES})
            source=CMGDB.plot_derived_graph(graph).source.replace('digraph {',f'digraph {{\ngraph [label="{title}",labelloc=t,fontname="Times-Roman",fontsize=18];')
            for k,name in NAMES.items():
                source=source.replace(f'label="{k} : ',f'label="{name} : ')
                source=re.sub(rf'({k} \[.*?fillcolor=")[^"]+',lambda m:m[1]+style.tint(SPHERE_IDS[k]),source)
                source=source.replace(f'{k} [label=',f'{k} [color="{COLORS[k]}", fontcolor="{style.INK}", penwidth=1.2, label=')
            save_graph(source,folder/f'{case}-sphere-{suffix}',crop=False)
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    # Size each row from its vector panels, with only a short inter-row gap.
    panel_width=580
    row_specs=[]; y=0
    for suffix in ('sphere-panel','interval-panel','sphere-quotient','sphere-reconstructed'):
        row_width=panel_width*(.415/.485) if suffix=='sphere-panel' else panel_width
        heights=[]
        for case in ('analytical','learned'):
            box=list(map(float,ET.parse(folder/f'{case}-{suffix}.svg').getroot().attrib['viewBox'].split()))
            heights.append(row_width*box[3]/box[2])
        h=max(heights)
        row_specs.append((suffix,y,h,row_width));y+=h+8
    total_height=y-8
    canvas=ET.Element('{'+ns+'}svg',{'width':'1200','height':str(total_height),
        'viewBox':f'0 0 1200 {total_height}'})
    for col,case in enumerate(('analytical','learned')):
        for suffix,y,h,row_width in row_specs:
            path=folder/f'{case}-{suffix}'
            text=path.with_suffix('.svg').read_text();prefix=f'{case}-{suffix}-'
            for ident in sorted(set(re.findall(r'id="([^"]+)"',text)),key=len,reverse=True):
                text=text.replace(f'id="{ident}"',f'id="{prefix}{ident}"').replace(f'#{ident}"',f'#{prefix}{ident}"').replace(f'#{ident})',f'#{prefix}{ident})')
            child=ET.fromstring(text)
            for key,value in dict(x=col*600+10+(panel_width-row_width)/2,y=y,width=row_width,height=h,preserveAspectRatio='xMidYMin meet').items():child.set(key,str(value))
            canvas.append(child)
    svg=folder/'sphere-comparison.svg'
    ET.ElementTree(canvas).write(svg,encoding='unicode')
    converter=shutil.which('rsvg-convert')
    if converter is None:raise RuntimeError('Install librsvg (rsvg-convert) for vector PDF export.')
    subprocess.run([converter,'--format=pdf','--output',str(folder/'sphere-comparison.pdf'),str(svg)],check=True)
    subprocess.run([converter,'--format=png','--width=1800','--background-color=white',
                    '--output',str(folder/'sphere-comparison.png'),str(svg)],check=True)
