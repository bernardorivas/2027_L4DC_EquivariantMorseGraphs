"""Fixed-checkpoint domain restriction and refinement using native CMGDB.

The learned encoder image is approximated from its parameterized boundary.
Images are sampled hulls, not certified enclosures. Exits are never projected
back into the domain. AtlasModel computes the restricted Morse graph natively.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def identity(fit, levels):
    return dict(inputs={name:digest(fit/name) for name in
                       ('models.npz','config.json','evaluation.npz')},
                implementation=digest(__file__), loader=digest(ROOT/'src/quotient_runtime.py'),
                levels=list(levels), boundary_points_per_side=4097, samples_per_axis=5)


def worker(fit, out, levels):
    from quotient_runtime import load_quotient
    from scipy.optimize import root, brentq, least_squares
    from scipy.spatial import cKDTree
    from matplotlib.path import Path as Polygon
    from importlib.metadata import version
    cfg, E, D, F = load_quotient(fit)
    sphere = cfg['system']=='sphere'
    dim = 1 if sphere else 2
    with np.load(fit/'evaluation.npz') as old:
        lower, upper = old['domain_lower'], old['domain_upper']
        baseline_boxes, baseline_images = old['boxes'], old['image_boxes']
    def encode(p):
        p=np.atleast_2d(p)
        x=(np.column_stack((np.sqrt(np.maximum(1-p[:,0]**2,0)),np.zeros(len(p)),p[:,0]))
           if sphere else np.column_stack((p,np.zeros((len(p),2)))))
        return np.asarray(E(x))
    def one(q):return np.asarray(F(np.atleast_2d(q)))
    def advance(q):
        for _ in range(cfg.get('cmgdb_iterations',10 if sphere else 1)):q=one(q)
        return q
    def batches(fn, q):
        return np.concatenate([fn(q[i:i+4096]) for i in range(0,len(q),4096)])
    if sphere:
        boundary=np.array([[-1.],[1.]])
        physical=np.linspace(-1,1,4097)[:,None]
        contains=lambda q:np.all((q>=-1-1e-10)&(q<=1+1e-10),axis=1)
        contains_coarse=contains
        anchor_physical=np.array([[-1.],[-.4],[0.],[.4],[1.]])
    else:
        t=np.linspace(0,1,4097)
        pb=np.concatenate((np.column_stack((-1.5+3*t,0*t)),
            np.column_stack((1.5+0*t,1.5*t)),np.column_stack((1.5-3*t,1.5+0*t)),
            np.column_stack((-1.5+0*t,1.5-1.5*t))))
        boundary=batches(encode,pb)
        # The dense closed boundary is sampled in physical quotient coordinates;
        # these are evaluation parameters, never encoder features or targets.
        # A positive Path radius expands only a counterclockwise boundary.
        # Learned invariant coordinates may reverse orientation.
        area=np.sum(boundary[:,0]*np.roll(boundary[:,1],-1)-boundary[:,1]*np.roll(boundary[:,0],-1))
        oriented=boundary if area>=0 else boundary[::-1]
        polygon=Polygon(oriented,closed=True)
        coarse=Polygon(np.vstack((oriented[::2],oriented[0])),closed=True)
        sides=boundary.reshape(4,4097,2)
        chord_error=float(np.linalg.norm(sides[:,1:-1:2]-(sides[:,:-2:2]+sides[:,2::2])/2,axis=2).max())
        boundary_tolerance=4*chord_error+1e-10
        contains=lambda q:polygon.contains_points(np.atleast_2d(q),radius=boundary_tolerance)
        contains_coarse=lambda q:coarse.contains_points(np.atleast_2d(q),radius=boundary_tolerance)
        physical=np.array([(a,r) for a in np.linspace(-1.5,1.5,101)
                           for r in np.linspace(0,1.5,81)])
        anchor_physical=np.array([[-1,0],[1,0],[0,1],[0,0]])
    encoded=batches(encode,physical)
    encoded_next=batches(one,encoded)
    initial_inside=contains(encoded)
    if not np.all(initial_inside):
        raise ValueError("Sampled encoder image excludes encoded inputs; inspect its boundary before CMGDB.")
    step_inside=contains(encoded_next)
    coarse_disagreement=int(np.sum(step_inside!=contains_coarse(encoded_next)))
    # Invert E by bounded least squares only to independently check membership;
    # D is never used as a surrogate for the inverse.
    tree=cKDTree(encoded)
    def inverse_check(q):
        start=physical[tree.query(q)[1]]
        lo,hi=([-1.],[1.]) if sphere else ([-1.5,0.],[1.5,1.5])
        answer=least_squares(lambda p:encode(p)[0]-q,start,bounds=(lo,hi),
            xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=250)
        return dict(physical=answer.x.tolist(),residual=float(np.linalg.norm(encode(answer.x)[0]-q)))
    roots=[]
    if sphere:
        q=np.linspace(-1,1,16385)
        delta=one(q[:,None])[:,0]-q
        roots=[np.array([-1.]),np.array([1.])]
        for a,b,da,db in zip(q[:-1],q[1:],delta[:-1],delta[1:]):
            if da*db<0:roots.append(np.array([brentq(lambda x:one([[x]])[0,0]-x,a,b,xtol=1e-13)]))
        min_slope=float(np.diff(one(q[:,None])[:,0]).min()/(q[1]-q[0]))
        starts_count=16384
    else:
        starts=encode(np.array([(a,r) for a in np.linspace(-1.5,1.5,15)
                               for r in np.linspace(0,1.5,13)]))
        starts_count=len(starts)
        for start in starts:
            answer=root(lambda q:one(q)[0]-q,start)
            # The inverse-softplus floor at q2=0 is a numerical extension, not
            # an actual zero fixed point. Exclude those near-zero candidates.
            candidate=answer.x.copy()
            gain=cfg.get('parameterization')=='gain'
            if gain:
                candidate[np.abs(candidate)<1e-10]=0.
            admissible=candidate[1]>=0 if gain else candidate[1]>1e-6
            if (np.linalg.norm(one(candidate)[0]-candidate)<1e-9 and admissible
                and all(np.linalg.norm(candidate-r)>1e-5 for r in roots)):
                roots.append(candidate)
        min_slope=None
    roots=sorted(roots,key=lambda q:tuple(q));root_rows=[]
    for i,q in enumerate(roots):
        h=1e-6;eye=np.eye(dim)*h
        jac=np.column_stack([(one(q+e)[0]-one(q-e)[0])/(2*h) for e in eye])
        eig=np.linalg.eigvals(jac)
        root_rows.append(dict(id=i,q=q.tolist(),inside=bool(contains(q[None])[0]),
            residual=float(np.linalg.norm(one(q)[0]-q)),
            multipliers=[[float(e.real),float(e.imag)] for e in eig],
            unstable_directions=int(np.sum(np.abs(eig)>1)),inverse_check=inverse_check(q)))
    witnesses=[]
    ids=np.flatnonzero(~step_inside)
    # Quantify distance for evenly spaced exit witnesses, not just polygon labels.
    for i in ids[np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)] if len(ids) else []:
        witnesses.append(dict(physical=physical[i].tolist(),q=encoded[i].tolist(),
                              image=encoded_next[i].tolist(),inverse_check=inverse_check(encoded_next[i])))
    # Exact checkpoint point iterates; first-exit counts use the same domain test.
    trajectory=encoded.copy();surviving=np.ones(len(encoded),dtype=bool);survival=[]
    for step in range(1,21):
        trajectory=batches(one,trajectory)
        surviving &= contains(trajectory)
        if step in (1,4,10,20):survival.append(dict(step=step,remaining=int(surviving.sum())))
    np.savez_compressed(out/'domain.npz',boundary=boundary,physical=physical,
        encoded=encoded,encoded_next=encoded_next,initial_inside=initial_inside,
        step_inside=step_inside,anchors=encode(anchor_physical))
    point_checks=dict(roots=root_rows,starts=starts_count,samples=len(encoded),
        initial_outside=int((~initial_inside).sum()),one_step_outside=int((~step_inside).sum()),
        boundary_membership_disagreements=coarse_disagreement,
        boundary_tolerance=0. if sphere else boundary_tolerance,
        min_sampled_one_step_slope=min_slope,exit_witnesses=witnesses,survival=survival,
        versions={p:version(p) for p in ('emlp','objax','jax','numpy','scipy')})
    save_json(out/'point-checks.json',point_checks)
    print(f"{cfg['system']}: {len(roots)} positive-domain fixed points found; "
          f"{point_checks['one_step_outside']}/{len(encoded)} one-step exits",flush=True)
    for level in levels:
        started=time.perf_counter();n=2**level;side=4*n+1
        shape=(side,)*dim
        ids=np.arange(side**dim)
        indices=np.stack(np.unravel_index(ids,shape),axis=1)
        q=lower+(upper-lower)*indices/(side-1)
        values=batches(advance,q).reshape(shape+(dim,))
        inside=contains(q).reshape(shape)
        full_lo=np.full((n,)*dim+(dim,),np.inf);full_hi=-full_lo.copy()
        cut_lo=full_lo.copy();cut_hi=-cut_lo.copy();counts=np.zeros((n,)*dim,dtype=np.int32)
        for offset in itertools.product(range(5),repeat=dim):
            slices=tuple(slice(k,4*n+k,4) for k in offset)
            sample=values[slices];valid=inside[slices]
            np.minimum(full_lo,sample,out=full_lo);np.maximum(full_hi,sample,out=full_hi)
            np.minimum(cut_lo,np.where(valid[...,None],sample,np.inf),out=cut_lo)
            np.maximum(cut_hi,np.where(valid[...,None],sample,-np.inf),out=cut_hi)
            counts+=valid
        # Include sampled E(boundary) in every touching source cell. This retains
        # physical boundary points even when a cell has no interior lattice node.
        coord=(boundary-lower)/(upper-lower)*n
        bv=batches(advance,boundary)
        for shifts in itertools.product((-1e-10,1e-10),repeat=dim):
            bi=np.floor(coord+shifts).astype(int)
            valid=np.all((bi>=0)&(bi<n),axis=1);bi=bi[valid]
            key=tuple(bi[:,j] for j in range(dim))
            for j in range(dim):
                np.minimum.at(cut_lo[...,j],key,bv[valid,j]);np.maximum.at(cut_hi[...,j],key,bv[valid,j])
            np.add.at(counts,key,1)
        active=counts>0
        full=np.concatenate((full_lo-1e-9,full_hi+1e-9),axis=-1)
        cut=np.concatenate((cut_lo-1e-9,cut_hi+1e-9),axis=-1)
        if sphere:
            full=np.clip(full,-1,1);cut=np.clip(cut,-1,1)
        else:full[...,1]=np.maximum(full[...,1],0)
        parity=None
        if level==7:
            center=(baseline_boxes[:,:dim]+baseline_boxes[:,dim:])/2
            bi=np.floor((center-lower)/(upper-lower)*n).astype(int)
            parity=float(np.max(np.abs(full[tuple(bi[:,j] for j in range(dim))]-baseline_images)))
            assert parity<1e-11,parity
        np.savez_compressed(out/f'grid-{n}.npz',full=full,cut=cut,active=active,
                            counts=counts,lower=lower,upper=upper)
        save_json(out/f'grid-{n}-evaluation.json',dict(n=n,active_cells=int(active.sum()),
            full_cells=n**dim,samples_per_axis=5,baseline_max_difference=parity,
            seconds=time.perf_counter()-started))
        print(f"Images {n}^{dim}: {active.sum()}/{n**dim} active cells",flush=True)


def native_run(out, n, full, cut, active, lower, upper, roots, sphere):
    import CMGDB
    dim=len(lower);level=int(np.log2(n));depth=dim*level
    side=(upper-lower)/n
    def index(box):
        b=np.asarray(box);return np.floor(((b[:dim]+b[dim:])/2-lower)/side).astype(int)
    def batch(boxes):
        b=np.asarray(boxes);ij=np.floor(((b[:,:dim]+b[:,dim:])/2-lower)/side).astype(int)
        return full[tuple(ij[:,j] for j in range(dim))].tolist()
    model=CMGDB.Model(depth,depth,depth,10000,lower.tolist(),upper.tolist(),lambda b:batch([b])[0])
    model.set_batch_map(batch)
    full_morse,full_phase=CMGDB.ComputeConleyMorseGraph(model)
    def summarize(morse,phase,name):
        nodes=[];saved=[]
        for k in morse.vertices():
            boxes=np.asarray([morse.phase_space_box(i) for i in morse.morse_set(k)])
            found=[r['id'] for r in roots if np.any(np.all(np.asarray(r['q'])>=boxes[:,:dim]-1e-10,axis=1)&
                        np.all(np.asarray(r['q'])<=boxes[:,dim:]+1e-10,axis=1))]
            nodes.append(dict(id=int(k),annotations=list(morse.annotations(k)),cell_count=len(boxes),
                lower=boxes[:,:dim].min(axis=0).tolist(),upper=boxes[:,dim:].max(axis=0).tolist(),roots=found))
            saved.extend([*b,int(k)] for b in boxes)
        np.savetxt(out/f'{name}-{n}-cells.csv',np.asarray(saved).reshape(-1,2*dim+1),delimiter=',')
        graph=CMGDB.PlotMorseGraph(morse)
        (out/f'{name}-{n}.dot').write_text(graph.source)
        for fmt in ('png','pdf','svg'):(out/f'{name}-{n}.{fmt}').write_bytes(graph.pipe(format=fmt))
        return dict(components=len(nodes),cells=phase.num_vertices(),nodes=nodes,
                    edges=[list(e) for e in morse.edges()])
    original=summarize(full_morse,full_phase,'rectangle')
    aggregate=None
    if sphere:
        from CMGDB.morse_lattice import DerivedGraph,transitive_reduction
        rp=np.array([r['q'] for r in roots]);groups={r['id']:[] for r in roots}
        for node in original['nodes']:
            center=(np.array(node['lower'])+node['upper'])/2
            distances=np.linalg.norm((rp-center)/side,axis=1)
            target=int(np.argmin(distances))
            if distances[target]>4:raise ValueError('Grouping exceeds four cells from a numerical fixed point.')
            groups[target].append(node['id'])
        labels={};members={};occupied=set()
        for k,group in groups.items():
            assert group
            cells=set(map(int,CMGDB.MorseDirectedPathCells(full_phase,full_morse,group,group)))
            touched={v for v in full_morse.vertices() if cells.intersection(full_morse.morse_set(v))}
            assert touched==set(group) and not cells.intersection(occupied)
            occupied.update(cells);members[k]=sorted(cells)
            labels[k]=list(CMGDB.ComputeConleyIndexForCells(model,full_morse,sorted(cells)))
        assignment={v:k for k,g in groups.items() for v in g}
        edges={k:set() for k in groups}
        for a,b in full_morse.edges():
            if assignment[a]!=assignment[b]:edges[assignment[a]].add(assignment[b])
        edges=transitive_reduction(list(groups),edges)
        expected={(0,1),(2,1),(2,3),(4,3)}
        flat={(a,b) for a,bs in edges.items() for b in bs}
        expected_indices={0:['0','0'],1:['x-1','0'],2:['0','x-1'],3:['x-1','0'],4:['0','0']}
        aggregate=dict(groups=groups,annotations=labels,edges=sorted([a,b] for a,b in flat),
            matches_analytical=flat==expected and labels==expected_indices,
            rule='Nearest found fixed point within four cells; include all directed connecting cells; check disjointness and recompute each index pair')
        graph=DerivedGraph(nodes=list(groups),edges={a:sorted(b) for a,b in edges.items()},
                           labels={k:'('+', '.join(v)+')' for k,v in labels.items()})
        dot=CMGDB.plot_derived_graph(graph)
        (out/f'aggregate-{n}.dot').write_text(dot.source)
        for fmt in ('png','pdf','svg'):(out/f'aggregate-{n}.{fmt}').write_bytes(dot.pipe(format=fmt))
    if sphere:
        restricted=original.copy()
        restricted['same_as_rectangle']=True
    else:
        atlas=CMGDB.AtlasModel(0,0,0,10000)
        atlas.add_chart(0,lower.tolist(),upper.tolist())
        atlas.set_active_subgrid([CMGDB.TaggedDyadicCell(0,level,list(map(int,ij)))
                                 for ij in np.argwhere(active)])
        atlas.set_map(lambda chart,b:[(0,cut[tuple(index(b))].tolist())])
        morse,phase=CMGDB.ComputeMorseGraph(atlas)
        assert phase.num_vertices()==int(active.sum())
        restricted=summarize(morse,phase,'restricted')
        restricted['same_as_rectangle']=False
        restricted['index_annotations']='Unavailable for native AtlasModel; none transferred from the rectangular graph.'
    return dict(n=n,rectangle=original,restricted=restricted,aggregate=aggregate)


def compute(fit,out,levels):
    before=identity(fit,levels)
    cfg=json.loads((fit/'config.json').read_text());sphere=cfg['system']=='sphere'
    env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'build/matplotlib'),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([str(ROOT/'.venv-emlp/bin/python'),'-B',__file__,'--worker','--fit',str(fit),
                    '--out',str(out),'--levels',*[str(x) for x in levels]],env=env,check=True)
    checks=json.loads((out/'point-checks.json').read_text());rows=[]
    for level in levels:
        n=2**level
        with np.load(out/f'grid-{n}.npz') as data:
            row=native_run(out,n,*[data[k] for k in ('full','cut','active','lower','upper')],checks['roots'],sphere)
        if level==7:
            old=json.loads((fit/'cmgdb.json').read_text())
            assert row['rectangle']['edges']==old['edges']
            assert [x['annotations'] for x in row['rectangle']['nodes']]==[x['annotations'] for x in old['nodes']]
        rows.append(row)
        print(f"CMGDB {n}: rectangle {row['rectangle']['components']}, restricted {row['restricted']['components']}",flush=True)
    assert identity(fit,levels)==before
    result=dict(system=cfg['system'],fit=str(fit.relative_to(ROOT)),identity=before,
                certified=False,domain='Sampled E(K), boundary from 4097 evaluations per physical quotient edge',
                exits='Targets outside active cells are exits, never projected or clipped to E(K)',
                point_checks=checks,resolutions=rows)
    save_json(out/'comparison.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--levels',type=int,nargs='+',default=[7,8,9])
    p.add_argument('--worker',action='store_true');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    (worker if args.worker else compute)(args.fit.resolve(),args.out.resolve(),args.levels)
