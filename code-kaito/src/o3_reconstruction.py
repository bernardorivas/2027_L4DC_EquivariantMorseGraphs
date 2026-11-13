"""O(3) reconstruction from local quotient pairs and direction-preserving S² fibers.

The local normal models supply identity chain maps. This is separate from the
sampled CMGDB aggregation: neither a certified neural-map carrier nor a global
encoder conjugacy is asserted here.
"""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/o3-reconstruction'


def square_pair(exit_edges):
    # Counterclockwise square vertices: bottom-left, bottom-right, top-right,
    # top-left. Edges: bottom, right, top (left-to-right), left (bottom-to-top).
    cells=[list(range(4)),list(range(4)),[0]]
    boundary=[{}, {0:{0:-1,1:1},1:{1:-1,2:1},2:{3:-1,2:1},3:{0:-1,3:1}},
              {0:{0:1,1:1,2:-1,3:-1}}]
    removed=[{v for e in exit_edges for v in boundary[1][e]},set(exit_edges),set()]
    remaining=[[v for v in cells[d] if v not in removed[d]] for d in range(3)]
    lookup=[{v:i for i,v in enumerate(c)} for c in remaining]
    entries=[[]]
    for d in (1,2):
        entries.append([(lookup[d-1][a],lookup[d][b],coef) for b in remaining[d]
                        for a,coef in boundary[d][b].items() if a in lookup[d-1]])
    counts=list(map(len,remaining));identity=[[(i,i,1) for i in range(n)] for n in counts]
    return counts,entries,identity


def shifted_sum(base,extra,shift=2,fiber_degree=1):
    counts=[];boundary=[];maps=[]
    for d in range(5):
        n=base[0][d] if d<3 else 0;m=extra[0][d-shift] if shift<=d<shift+3 else 0
        counts.append(n+m)
        bs=list(base[1][d]) if d<3 else []
        fs=list(base[2][d]) if d<3 else []
        if shift<=d<shift+3:
            prev=base[0][d-1] if 0<=d-1<3 else 0
            bs += [(r+prev,c+n,v) for r,c,v in extra[1][d-shift]]
            fs += [(r+n,c+n,fiber_degree*v) for r,c,v in extra[2][d-shift]]
        boundary.append(bs);maps.append(fs)
    return counts,boundary,maps


def compute(fit,verbose=False):
    import CMGDB
    from CMGDB.morse_lattice import DerivedGraph
    OUT.mkdir(parents=True,exist_ok=True)
    agg=json.loads((ROOT/'build/o3-aggregation/comparison.json').read_text())
    assert hashlib.sha256((fit/'models.npz').read_bytes()).hexdigest()==agg['checkpoint_sha256']
    roots=json.loads((ROOT/'build/o3-repair/cmgdb-001/study/point-checks.json').read_text())['roots']
    # Roles follow orbit stratum and numerical normal multipliers, not node IDs.
    roles={}
    for r in roots:
        if r['q'][1]>1e-8:role='C';assert r['unstable_directions']==1
        elif abs(r['q'][0])<1e-8:role='O';assert r['unstable_directions']==2
        else:role='A-' if r['q'][0]<0 else 'A+';assert r['unstable_directions']==0
        assert all(im==0 and re>0 for re,im in r['multipliers'])
        roles[str(r['id'])]=role
    # Local half-plane pairs: O exits at sides/top, C at the two sides,
    # sinks have no exit. P is the bottom edge except at the interior orbit C.
    pairs={'O':({1,2,3},{0}), 'C':({1,3},set()), 'A-':(set(),{0}), 'A+':(set(),{0})}
    expected={'O':['0','0','0','0','x-1'],'C':['0','x-1','0','x-1','0'],
              'A-':['x-1','0','0','0','0'],'A+':['x-1','0','0','0','0']}
    results={}
    for name,(exit_edges,p_edges) in pairs.items():
        base=square_pair(exit_edges);extra=square_pair(exit_edges|p_edges)
        b=CMGDB.ComputeRelativeHomologyShiftClass(*base)
        e=CMGDB.ComputeRelativeHomologyShiftClass(*extra)
        full=CMGDB.ComputeRelativeHomologyShiftClass(*shifted_sum(base,extra))
        results[name]=dict(quotient=b,relative_to_axis=e,reconstructed=full,
                          local_pair={'exit_edges':sorted(exit_edges),'collapsed_fiber_edges':sorted(p_edges)})
    for name,r in results.items():
        assert r['reconstructed']['shift_class']==expected[name]
    comparisons=[]
    for row in agg['original_grid']:
        assert all(row['indices'][key]==results[role]['quotient']['shift_class'] for key,role in roles.items())
        named_edges={(roles[str(a)],roles[str(b)]) for a,b in row['edges']}
        assert named_edges=={('O','C'),('C','A-'),('C','A+')}
        comparisons.append(dict(n=row['n'],quotient_indices_match_local_pairs=True,original_space_indices_match=True))
    # The quotient alone cannot select the fiber degree: antipodal changes the
    # recovered summand while leaving the quotient summand unchanged.
    antipodal={}
    for name in ('O','C'):
        le,pe=pairs[name]
        alt=CMGDB.ComputeRelativeHomologyShiftClass(*shifted_sum(square_pair(le),square_pair(le|pe),fiber_degree=-1))
        assert alt['shift_class']!=expected[name]
        antipodal[name]=alt['shift_class']
    labels={key:'('+', '.join(results[role]['reconstructed']['shift_class'])+')' for key,role in roles.items()}
    row=agg['original_grid'][-1];keys=sorted(map(int,roles));edges={k:[] for k in keys}
    for a,b in row['edges']:edges[a].append(b)
    for tag,lab in [('learned',{int(k):v for k,v in labels.items()}),
                    ('analytical',{int(k):'('+', '.join(expected[role])+')' for k,role in roles.items()})]:
        graph=DerivedGraph(nodes=keys,edges=edges,labels=lab)
        dot=CMGDB.plot_derived_graph(graph)
        from graphviz import Source
        source=dot.source
        for key,role in roles.items():source=source.replace(f'label="{key} :',f'label="{role} :')
        dot=Source(source)
        for ext in ('svg','png','pdf'):(OUT/f'{tag}-original.{ext}').write_bytes(dot.pipe(format=ext))
        (OUT/f'{tag}-original.dot').write_text(dot.source)
    if verbose:print(json.dumps({'comparisons':comparisons,'indices':expected,'antipodal_check':antipodal}),flush=True)
    # Store first, including full native return values for transparent auditing.
    record=dict(checkpoint_sha256=agg['checkpoint_sha256'],roles=roles,local_models=results,
                fiber_dimension=2,fiber_degree=1,field='F5',expected_analytical=expected,comparisons=comparisons,antipodal_sanity_check=antipodal,
                status='Local-model chain algebra with Arb-verified isolating continuation for the specified learned quotient lift; no global encoder conjugacy or global graph certificate.',
                local_certificate='build/o3-certification/check-001/local-certificate.json',
                source='paper/main.tex: thm:sphere-reconstruction, app:o3, prop:indices-differ(b)',
                manuscript_sha256=(hashlib.sha256((ROOT.parent/'paper/main.tex').read_bytes()).hexdigest()
                                   if (ROOT.parent/'paper/main.tex').exists() else None))
    (OUT/'local-models.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

def show_reconstruction(fit,show_graphs=False):
    from notebook_support import gallery,table
    from IPython.display import Markdown,display
    record=compute(fit)
    cert=json.loads((ROOT/'build/o3-certification/check-001/local-certificate.json').read_text())
    assert cert['checkpoint_sha256']==record['checkpoint_sha256'] and cert['all_local_inequalities_verified']
    display(Markdown(r"""### Original-space reconstruction

The nonzero-radius fibers are $S^2$ and collapse on $P=\{r=0\}$. For the specified direction-preserving lift, let $(B,L)$ be a quotient index pair, with isolating neighborhood $B$ and exit set $L$. Its lifted pair is $(N_X,L_X)$. With coefficients in $\mathbb F_5$,

$$H_k(N_X,L_X) \cong H_k(B,L)\oplus H_{k-2}(B,L\cup(B\cap P)).$$

Arb interval bounds verify local expansion/contraction and unique fixed-point brackets for this saved map. These give a local isolating continuation to the positive normal forms, whose index maps are identities. CMGDB computes the relative-chain indices over $\mathbb F_5$; entries below run from degree 0 to 4.
"""))
    if show_graphs:
        gallery([('Analytical — original space',OUT/'analytical-original.svg'),
                 ('Learned — original space',OUT/'learned-original.svg')],max_height=300)
    table(['Set','Quotient classes','Additional original-space classes','Original-space classes'],
          [('Origin','None','H₄','H₄'),('Saddle sphere','H₁','H₃','H₁, H₃'),('Each sink','H₀','None','H₀')])
    display(Markdown('The quotient graph alone does not determine the fiber degree. The specified lift has degree +1. Local bounds are restored to `build/o3-certification/local-index-argument.md`.'))
    from conjugacy_summary import show_conjugacy_check
    show_conjugacy_check(fit)
    return record

if __name__=='__main__':compute(ROOT/'build/o3-repair/gain-001/fit',verbose=True)
