"""Fixed-checkpoint O(3) grid diagnostics and path-closed CMGDB aggregation."""
from pathlib import Path
import json,itertools,os,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'build/o3-repair/cmgdb-001/study'
OUT=ROOT/'build/o3-aggregation'

def aggregate(n,full,lower,upper,roots,out):
    import CMGDB
    from CMGDB.morse_lattice import DerivedGraph,transitive_reduction
    side=(upper-lower)/n;level=int(np.log2(n))
    def batch(boxes):
        b=np.asarray(boxes);ix=np.floor(((b[:,:2]+b[:,2:])/2-lower)/side).astype(int)
        return full[ix[:,0],ix[:,1]].tolist()
    model=CMGDB.Model(2*level,2*level,2*level,10000,lower.tolist(),upper.tolist(),lambda b:batch([b])[0]);model.set_batch_map(batch)
    m,p=CMGDB.ComputeConleyMorseGraph(model)
    dot=CMGDB.PlotMorseGraph(m)
    for ext in ('svg','png','pdf'):(out/f'native-{n}.{ext}').write_bytes(dot.pipe(format=ext))
    groups={r['id']:[] for r in roots};rp=np.array([r['q'] for r in roots]);distances=[]
    for v in m.vertices():
        boxes=np.array([m.phase_space_box(i) for i in m.morse_set(v)])
        centers=(boxes[:,:2]+boxes[:,2:])/2
        ds=np.linalg.norm((centers[:,None,:]-rp[None,:,:])/side,axis=2)
        k=int(np.argmin(ds.mean(axis=0)));assert ds[:,k].max()<4
        groups[roots[k]['id']].append(int(v));distances.append(float(ds[:,k].max()))
    labels={};occupied=set();members={};touched_check={}
    for k,g in groups.items():
        assert g
        cells=set(map(int,CMGDB.MorseDirectedPathCells(p,m,g,g)))
        touched={int(v) for v in m.vertices() if cells.intersection(m.morse_set(v))}
        assert touched==set(g) and not occupied.intersection(cells)
        occupied.update(cells);members[k]=sorted(cells);touched_check[k]=sorted(touched)
        # This native API checks the index-pair condition and raises on failure.
        labels[k]=list(CMGDB.ComputeConleyIndexForCells(model,m,sorted(cells)))
    assignment={v:k for k,g in groups.items() for v in g}
    edges={k:set() for k in groups}
    for a,b in m.edges():
        if assignment[a]!=assignment[b]:edges[assignment[a]].add(assignment[b])
    # Reject cycles in the contracted order before transitive reduction.
    pending=set(groups)
    while pending:
        sinks={k for k in pending if not (edges[k]&pending)}
        assert sinks, 'Grouping introduces a cycle'
        pending-=sinks
    edges=transitive_reduction(list(groups),edges)
    flat=sorted([a,b] for a,bs in edges.items() for b in bs)
    derived=DerivedGraph(nodes=list(groups),edges={k:sorted(v) for k,v in edges.items()},labels={k:'('+', '.join(v)+')' for k,v in labels.items()})
    dot=CMGDB.plot_derived_graph(derived)
    for ext in ('svg','png','pdf'):(out/f'aggregate-{n}.{ext}').write_bytes(dot.pipe(format=ext))
    (out/f'aggregate-{n}.dot').write_text(dot.source)
    # Compare annotated directed graphs, allowing arbitrary CMGDB node numbering.
    analytical=json.loads((STUDY/'analytical-matched-time/comparison.json').read_text())['resolutions'][-1]
    refnodes=analytical['nodes'];keys=list(groups);match=False
    for permutation in itertools.permutations([v['id'] for v in refnodes]):
        mapping=dict(zip(keys,permutation));refindices={v['id']:v['annotations'] for v in refnodes}
        if all(labels[k]==refindices[mapping[k]] for k in keys) and {(mapping[a],mapping[b]) for a,b in flat}=={tuple(e) for e in analytical['edges']}:
            match=True;break
    result=dict(n=n,native_components=len(list(m.vertices())),groups=groups,indices=labels,edges=flat,
      matches_analytical_annotated_graph=match,max_distance_cells=max(distances),
      group_cell_counts={k:len(v) for k,v in members.items()},path_closed_disjoint=True,index_pairs_validated=True,
      index_scope='CMGDB rectangular quotient domain; not transferred to restricted Atlas domain',certified_continuous_map=False)
    (out/f'result-{n}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result

def shifted_images():
    from quotient_runtime import load_quotient
    cfg,E,D,F=load_quotient(ROOT/'build/o3-repair/gain-001/fit')
    with np.load(STUDY/'grid-256.npz') as d:oldlo,oldhi=d['lower'],d['upper']
    # Equal width/resolution for both registrations; extra margin retains the image.
    width=2*max(abs(oldlo[0]),abs(oldhi[0]))*1.03
    for n in (128,256):
      for shift in (0.,.5):
        lower=np.array([-width/2+shift*width/n,0.]);upper=np.array([width/2+shift*width/n,oldhi[1]])
        axes=[np.linspace(lower[i],upper[i],4*n+1) for i in range(2)]
        mesh=np.meshgrid(*axes,indexing='ij');q=np.stack(mesh,axis=-1).reshape(-1,2)
        values=np.concatenate([np.asarray(F(q[i:i+4096])) for i in range(0,len(q),4096)]).reshape(4*n+1,4*n+1,2)
        lo=np.full((n,n,2),np.inf);hi=-lo.copy()
        for i,j in itertools.product(range(5),repeat=2):
          val=values[i:4*n+i:4,j:4*n+j:4];np.minimum(lo,val,out=lo);np.maximum(hi,val,out=hi)
        full=np.concatenate((lo-1e-9,hi+1e-9),axis=-1);full[...,1]=np.maximum(0,full[...,1])
        np.savez_compressed(OUT/f'shift-{shift}-{n}.npz',full=full,lower=lower,upper=upper)
        print('evaluated',n,shift,flush=True)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    record=json.loads((STUDY/'comparison.json').read_text());roots=record['point_checks']['roots'];results=[]
    for n in (64,128,256):
        with np.load(STUDY/f'grid-{n}.npz') as d:results.append(aggregate(n,d['full'],d['lower'],d['upper'],roots,OUT))
    env=dict(os.environ,MPLCONFIGDIR=str(ROOT/'build/matplotlib'))
    subprocess.run([str(ROOT/'.venv-emlp/bin/python'),__file__,'--images'],check=True,env=env)
    shifts=[]
    for n in (128,256):
      for shift in (0.,.5):
        dest=OUT/f'shift-{shift}';dest.mkdir(exist_ok=True)
        with np.load(OUT/f'shift-{shift}-{n}.npz') as d:
            r=aggregate(n,d['full'],d['lower'],d['upper'],roots,dest);r['grid_shift_cells']=shift;shifts.append(r)
    (OUT/'comparison.json').write_text(json.dumps(dict(original_grid=results,shifted_grids=shifts,checkpoint_sha256=record['identity']['inputs']['models.npz']),indent=2)+'\n')

if __name__=='__main__':
    import sys
    shifted_images() if '--images' in sys.argv else run()


def show_aggregation(fit,show_graphs=True,show_placement=True):
    import hashlib
    from notebook_support import gallery,table
    from IPython.display import HTML,display
    result=json.loads((OUT/'comparison.json').read_text())
    assert hashlib.sha256((fit/'models.npz').read_bytes()).hexdigest()==result['checkpoint_sha256'], 'Recompute aggregation for this checkpoint.'
    n=256
    if show_graphs:
        gallery([('Analytical',STUDY/'analytical-matched-time/analytical-256.svg'),
                 ('Learned — native',OUT/'native-256.svg'),
                 ('Learned — aggregated',OUT/'aggregate-256.svg')],max_height=350)
    table(['Grid','Native learned','Aggregated','Analytical graph + indices'],
          [(r['n'],r['native_components'],len(r['groups']),'Match' if r['matches_analytical_annotated_graph'] else 'Mismatch') for r in result['original_grid']])
    display(HTML('<p>Groups include all directed connecting cells; they are disjoint, and CMGDB validates each index pair and recomputes its index. The extra nodes group with the origin and saddle. These are quotient indices on the rectangular domain, not reconstructed original-space indices.</p>'))
    if not show_placement:return
    table(['Grid','Aligned grid: native nodes','Half-cell shift: native nodes','All aggregated indices match'],
          [(n,*[next(r['native_components'] for r in result['shifted_grids'] if r['n']==n and r['grid_shift_cells']==s) for s in (0.,.5)],'Yes') for n in (128,256)])
    display(HTML('<p>The grid-placement test uses identical weights, time, cell widths and sampling; only placement changes between its two columns. Together with sampled strict drift in both extra components, this supports a grid-artifact explanation. It is not a rigorous exclusion of all recurrent dynamics of the continuous neural map: cell-image bounds remain sampled.</p>'))
