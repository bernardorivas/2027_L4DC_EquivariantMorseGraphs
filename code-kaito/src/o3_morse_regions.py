"""Native CMGDB Morse cells and finite-graph basin labels in 2D quotient coordinates."""
from pathlib import Path
import json,itertools,subprocess,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/o3-certification'
from paper_figure_style import style, O3_IDS, O3_COLORS as COLORS, save_graph, size_panel_title

def run():
    import CMGDB
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    study=ROOT/'build/o3-repair/cmgdb-001/study';n=64;level=6
    roots=json.loads((study/'point-checks.json').read_text())['roots']
    roles={0:'A-',1:'O',2:'C',3:'A+'}
    group=json.loads((ROOT/'build/o3-aggregation/result-64.json').read_text())['groups']
    node_roles={v:roles[int(k)] for k,vs in group.items() for v in vs}
    records=[]
    for case in ('analytical','learned'):
        if case=='learned':
            with np.load(study/f'grid-{n}.npz') as d:full,active,lower,upper=[d[k] for k in ('cut','active','lower','upper')]
        else:
            lower=np.array([-1.5,0.]);upper=np.array([1.5,1.5]);active=np.ones((n,n),bool)
            a,r=np.meshgrid(np.linspace(-1.5,1.5,4*n+1),np.linspace(0,1.5,4*n+1),indexing='ij');den=1+8*a*a
            val=np.stack((3*a/np.sqrt(den),3*r/np.sqrt(den*(den+8*r*r))),axis=-1)
            lo=np.full((n,n,2),np.inf);hi=-lo.copy()
            for i,j in itertools.product(range(5),repeat=2):
                v=val[i:4*n+i:4,j:4*n+j:4];np.minimum(lo,v,out=lo);np.maximum(hi,v,out=hi)
            full=np.concatenate((lo-1e-9,hi+1e-9),axis=-1);full[...,1]=np.maximum(0,full[...,1])
        side=(upper-lower)/n
        def ix(b):return np.floor(((np.asarray(b)[:2]+np.asarray(b)[2:])/2-lower)/side).astype(int)
        # Same native Atlas computation for both panels; no index transfer.
        model=CMGDB.AtlasModel(0,0,0,10000);model.add_chart(0,lower.tolist(),upper.tolist())
        model.set_active_subgrid([CMGDB.TaggedDyadicCell(0,level,list(map(int,ij))) for ij in np.argwhere(active)])
        model.set_map(lambda chart,b:[(0,full[tuple(ix(b))].tolist())])
        m,p=CMGDB.ComputeMorseGraph(model);dot=CMGDB.PlotMorseGraph(m);dotpath=OUT/f'{case}-region-graph.dot';dotpath.write_text(dot.source)
        singleton=CMGDB.MorseSingletonReachability(p,m,np.arange(p.num_vertices(),dtype=np.uint64))
        _,targets=p.csr_view()
        image_cells=np.unique(targets)
        included=np.zeros(p.num_vertices(),bool);included[image_cells]=True
        boxes=np.array([m.phase_space_box(i) for i in range(p.num_vertices())]);indices=np.array([ix(b) for b in boxes])
        cells=[];named={}
        for k in m.vertices():
            bs=np.array([m.phase_space_box(i) for i in m.morse_set(k)])
            if case=='learned':role=node_roles[int(k)]
            else:
                center=(bs[:,:2].min(axis=0)+bs[:,2:].max(axis=0))/2
                anchors=np.array([[-1,0],[1,0],[0,1],[0,0]])
                role=['A-','A+','C','O'][int(np.argmin(np.linalg.norm(anchors-center,axis=1)))]
            named[int(k)]=role;cells.extend([*b,int(k)] for b in bs)
        path_cells=CMGDB.MorseDirectedPathCells(p,m,[k for k,v in named.items() if v=='O'],[k for k,v in named.items() if v=='C'])
        color_list=[COLORS[named[k]] for k in range(len(named))]
        with plt.rc_context({'font.family':'serif','mathtext.fontset':'cm','font.size':12}):
            fig,ax=CMGDB.PlotMorseSets(cells,proj_dims=[0,1],clist=color_list,fig_w=5.2,fig_h=3.3,
                xlim=[lower[0],upper[0]],ylim=[-.035,upper[1]],xlabel='$a$' if case=='analytical' else '$q_1$',
                ylabel='$r$' if case=='analytical' else '$q_2$',fontsize=12,linewidth=.8,edge_clr='.3',show=False)
            raster=np.zeros((n,n),int)
            for i,(u,v) in enumerate(indices):
                role=named.get(int(singleton[i]))
                if included[i]:raster[u,v]=1 if role=='A-' else (2 if role=='A+' else 0)
            for i in path_cells:
                u,v=indices[int(i)];raster[u,v]=3
            cmap=ListedColormap(['white',style.tint(O3_IDS['A-']),style.tint(O3_IDS['A+']),style.tint(O3_IDS['C'])])
            # Bernardo's CMGDB helper removes shared edges without rasterizing.
            from CMGDB.PlotMorseSets import _merged_outline
            from matplotlib.patches import PathPatch
            for value in (1,2,3):
                ij=np.argwhere(raster==value)
                if not len(ij):continue
                lows=lower+ij*side
                outline=_merged_outline(lows,lows+side)
                if outline is None:raise ValueError('Basin cells must lie on one grid.')
                ax.add_patch(PathPatch(outline,facecolor=cmap(value),edgecolor='none',
                                       antialiased=False,zorder=-10))
            cell_array=np.asarray(cells)
            for role,label in [('O',r'$O$'),('C',r'$C$'),('A-',r'$A_-$'),('A+',r'$A_+$')]:
                selected=cell_array[np.isin(cell_array[:,-1],[k for k,v in named.items() if v==role])]
                center=(selected[:,:2].min(axis=0)+selected[:,2:4].max(axis=0))/2
                offset={'O':(16,22),'C':(15,22),'A-':(-22,22),'A+':(18,22)}[role]
                ax.annotate(label,xy=center,xytext=offset,textcoords='offset points',color=COLORS[role],
                            fontsize=15,ha='center',arrowprops={'arrowstyle':'-','color':COLORS[role],'lw':.7})
            ax.set_title('Analytical'  if case=='analytical' else 'Learned')
            fig.tight_layout()
            size_panel_title(fig,ax)
            for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{case}-morse-regions.{ext}',dpi=180)
            plt.close(fig)
        np.savez_compressed(OUT/f'{case}-basin-labels.npz',boxes=boxes,singleton=singleton,image_cells=image_cells,path_cells=path_cells)
        np.savetxt(OUT/f'{case}-morse-cells.csv',cells,delimiter=',')
        records.append(dict(checkpoint_sha256=json.loads((OUT/'network.json').read_text())['checkpoint_sha256'],case=case,n=n,active_cells=p.num_vertices(),morse_components=len(named),roles=named,
          image_cells=len(image_cells),path_cells=len(path_cells),labels='Bernardo restriction: singleton sink reachability on CSR target union; O-to-C path overlay. Sampled finite graph, not certified continuous-map RoA'))
        print(records[-1],flush=True)
    make_comparison(study,n)
    (OUT/'regions.json').write_text(json.dumps(records,indent=2)+'\n')

def make_comparison(study,n):
    """Reuse computed quotient graphs; change only names, colors and layout."""
    import CMGDB
    from CMGDB.morse_lattice import DerivedGraph
    from graphviz import Source
    from matplotlib import pyplot as plt
    import re
    import xml.etree.ElementTree as ET
    reference=next(r for r in json.loads((study/'analytical-matched-time/comparison.json').read_text())['resolutions'] if r['n']==n)
    learned=json.loads((ROOT/f'build/o3-aggregation/result-{n}.json').read_text())
    reconstruction=json.loads((ROOT/'build/o3-reconstruction/local-models.json').read_text())
    assert reconstruction['checkpoint_sha256']==json.loads((OUT/'network.json').read_text())['checkpoint_sha256']
    for case in ('analytical','learned'):
        if case=='analytical':
            names={0:'A-',1:'A+',2:'C',3:'O'}
            labels={v['id']:'('+', '.join(v['annotations'])+')' for v in reference['nodes']}
            edges=reference['edges']
        else:
            names={0:'A-',1:'O',2:'C',3:'A+'}
            labels={int(k):'('+', '.join(v)+')' for k,v in learned['indices'].items()}
            edges=learned['edges']
        for suffix,title in [('named-aggregate','Quotient'),('named-original','Original space')]:
            shown=labels if suffix=='named-aggregate' else {
                k:'('+', '.join(reconstruction['local_models'][role]['reconstructed']['shift_class']
                    if case=='learned' else reconstruction['expected_analytical'][role])+')'
                for k,role in names.items()}
            graph=DerivedGraph(nodes=list(names),edges={k:[b for a,b in edges if a==k] for k in names},labels=shown)
            native=CMGDB.plot_derived_graph(graph).source
            native=native.replace('digraph {',f'digraph {{\ngraph [label="{title}", labelloc=t, fontsize=18, fontname="Times-Roman", ranksep=0.25];')
            for k,role in names.items():
                name={'A-':'A−','A+':'A+'}.get(role,role)
                native=native.replace(f'label="{k} : ',f'label="{name} : ')
                native=re.sub(rf'({k} \[.*?fillcolor=")[^"]+',lambda m:m[1]+style.tint(O3_IDS[role]),native)
                native=native.replace(f'{k} [label=',f'{k} [color="{COLORS[role]}", fontcolor="{style.INK}", penwidth=1.2, label=')
            save_graph(native,OUT/f'{case}-{suffix}')
    # Editable vector composition; original plot SVG and native Graphviz SVG remain separate.
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    suffixes=('morse-regions','named-aggregate','named-original')
    sizes={}
    for case in ('analytical','learned'):
        for suffix in suffixes:
            box=list(map(float,ET.parse(OUT/f'{case}-{suffix}.svg').getroot().attrib['viewBox'].split()))
            sizes[case,suffix]=box[2:]
    # Keep the graph text at one scale across quotient and reconstruction rows.
    graph_scale=530/max(w for (case,suffix),(w,h) in sizes.items() if suffix!='morse-regions')
    row_specs=[];y=0
    for suffix in suffixes:
        scale=530/sizes['analytical',suffix][0] if suffix=='morse-regions' else graph_scale
        h=max(sizes[case,suffix][1]*scale for case in ('analytical','learned'))
        row_specs.append((suffix,y,h,scale));y+=h+8
    total_height=y-8
    canvas=ET.Element('{'+ns+'}svg',{'width':'1100','height':str(total_height),
        'viewBox':f'0 0 1100 {total_height}'})
    for col,case in enumerate(('analytical','learned')):
        for suffix,y,h,scale in row_specs:
            text=(OUT/f'{case}-{suffix}.svg').read_text()
            prefix=f'{case}-{suffix}-'
            for ident in sorted(set(re.findall(r'id="([^"]+)"',text)),key=len,reverse=True):
                text=text.replace(f'id="{ident}"',f'id="{prefix}{ident}"').replace(f'#{ident}"',f'#{prefix}{ident}"').replace(f'#{ident})',f'#{prefix}{ident})')
            child=ET.fromstring(text)
            w,ph=sizes[case,suffix];w*=scale;ph*=scale
            for key,value in dict(x=col*550+(550-w)/2,y=y,width=w,height=ph).items():child.set(key,str(value))
            canvas.append(child)
    svg=OUT/'quotient-comparison.svg'
    ET.ElementTree(canvas).write(svg,encoding='unicode')
    converter=shutil.which('rsvg-convert')
    if converter is None:raise RuntimeError('Install librsvg (rsvg-convert) for vector PDF export.')
    subprocess.run([converter,'--format=pdf','--output',str(OUT/'quotient-comparison.pdf'),str(svg)],check=True)
    subprocess.run([converter,'--format=png','--width=1800','--background-color=white',
                    '--output',str(OUT/'quotient-comparison.png'),str(svg)],check=True)

if __name__=='__main__':run()



def show_regions(fit):
    import hashlib
    from notebook_support import gallery
    from IPython.display import Markdown,display
    records=json.loads((OUT/'regions.json').read_text())
    assert all(r['checkpoint_sha256']==hashlib.sha256((fit/'models.npz').read_bytes()).hexdigest() for r in records)
    gallery([('',OUT/'quotient-comparison.svg')],max_height=1120)
    display(Markdown('Labels and colors identify the four aggregates. The middle row shows the corresponding computed quotient indices on the rectangular grids; these indices are not transferred to the restricted Atlas domain. The bottom row reconstructs the original-space indices using degree +1 sphere fibers (the specified learned lift in the right column). Solid cells are native Morse sets, colored by their aggregate. At **64 × 64**, pale basin colors are restricted to the one-step image cells; pale green marks O-to-C paths. Each panel uses its own quotient coordinates. White is uncolored, not necessarily outside the basin. These are finite-graph labels, not certified RoA. Index computations use the finer grids reported below.'))
