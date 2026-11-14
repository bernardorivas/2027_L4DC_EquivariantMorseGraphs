"""Independent point checks and native Atlas/rectangle graph parity."""
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def points(study,out):
    from quotient_runtime import load_quotient
    from matplotlib.path import Path as Polygon
    from scipy.optimize import linprog
    record=json.loads((study/'comparison.json').read_text())
    cfg,E,D,F=load_quotient(ROOT/record['fit']);sphere=cfg['system']=='sphere';dim=1 if sphere else 2
    with np.load(study/'domain.npz') as data:boundary=data['boundary']
    if not sphere:
        area=np.sum(boundary[:,0]*np.roll(boundary[:,1],-1)-boundary[:,1]*np.roll(boundary[:,0],-1))
        boundary=boundary if area>=0 else boundary[::-1]
    polygon=None if sphere else Polygon(boundary,closed=True)
    def one(q):return np.asarray(F(np.atleast_2d(q)))
    def image(q):
        for _ in range(cfg.get('cmgdb_iterations',10 if sphere else 1)):q=one(q)
        return q
    rows=[]
    for row in record['resolutions']:
        n=row['n'];prefix='rectangle' if sphere else 'restricted'
        cells=np.loadtxt(study/f'{prefix}-{n}-cells.csv',delimiter=',',ndmin=2)
        for node in row['restricted']['nodes']:
            boxes=cells[cells[:,-1]==node['id'],:2*dim]
            samples=[]
            for box in boxes:
                axes=[np.linspace(box[d],box[d+dim],257 if sphere else 33) for d in range(dim)]
                mesh=np.meshgrid(*axes,indexing='ij');samples.append(np.stack([a.ravel() for a in mesh],axis=1))
            q=np.concatenate(samples)
            if not sphere:q=q[polygon.contains_points(q,radius=record['point_checks']['boundary_tolerance'])]
            if not len(q):continue
            delta=image(q)-q
            center=q.mean(axis=0);direction=(image(center[None])[0]-center)
            norm=np.linalg.norm(direction)
            projection=delta@(direction/norm) if norm>1e-14 else np.zeros(len(q))
            direction_method='center displacement'
            if not node['roots'] and projection.min()<=1e-12:
                # Search a separating linear functional for the sampled drift
                # vectors. This remains a sample check, not a uniform bound.
                answer=linprog(np.r_[np.zeros(dim),-1.],
                    A_ub=np.column_stack((-delta,np.ones(len(delta)))),b_ub=np.zeros(len(delta)),
                    bounds=[(-1,1)]*dim+[(None,None)],method='highs')
                if answer.success and answer.x[-1]>1e-12:
                    direction=answer.x[:dim];norm=np.linalg.norm(direction)
                    projection=delta@(direction/norm);direction_method='linear separation of sampled displacements'
            rows.append(dict(n=n,node=node['id'],roots=node['roots'],samples=len(q),
                displacement_min=delta.min(axis=0).tolist(),displacement_max=delta.max(axis=0).tolist(),
                min_displacement_norm=float(np.linalg.norm(delta,axis=1).min()),
                sampled_strict_direction=bool(projection.min()>1e-12),
                direction_method=direction_method,
                direction=(direction/norm).tolist() if norm>1e-14 else None,
                min_directional_displacement=float(projection.min())))
    (out/'local-displacement.json').write_text(json.dumps(rows,indent=2)+'\n')


def verify(study,out):
    import CMGDB
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from CMGDB.morse_lattice import DerivedGraph,transitive_reduction
    record=json.loads((study/'comparison.json').read_text());sphere=record['system']=='sphere'
    def construct(n,restricted=False):
        with np.load(study/f'grid-{n}.npz') as data:
            full,cut,active,lower,upper=[data[k] for k in ('full','cut','active','lower','upper')]
        dim=len(lower);side=(upper-lower)/n;level=int(np.log2(n))
        def ix(box):return np.floor(((np.asarray(box)[:dim]+np.asarray(box)[dim:])/2-lower)/side).astype(int)
        def batch(boxes):
            b=np.asarray(boxes);ij=np.floor(((b[:,:dim]+b[:,dim:])/2-lower)/side).astype(int)
            return full[tuple(ij[:,i] for i in range(dim))].tolist()
        if restricted:
            model=CMGDB.AtlasModel(0,0,0,10000);model.add_chart(0,lower.tolist(),upper.tolist())
            model.set_active_subgrid([CMGDB.TaggedDyadicCell(0,level,list(map(int,ij))) for ij in np.argwhere(active)])
            model.set_map(lambda chart,b:[(0,cut[tuple(ix(b))].tolist())])
            morse,phase=CMGDB.ComputeMorseGraph(model)
        else:
            depth=dim*level
            model=CMGDB.Model(depth,depth,depth,10000,lower.tolist(),upper.tolist(),lambda b:batch([b])[0]);model.set_batch_map(batch)
            morse,phase=CMGDB.ComputeConleyMorseGraph(model)
        return model,morse,phase,full,cut,active,lower,upper,ix
    evidence={}
    if not sphere:
        # Same real checkpoint images, all cells active: Atlas must reproduce
        # the rectangle's SCC sets and order before domain restriction is trusted.
        model,morse,phase,full,cut,active,lower,upper,ix=construct(128)
        atlas=CMGDB.AtlasModel(0,0,0,10000);atlas.add_chart(0,lower.tolist(),upper.tolist())
        atlas.set_active_subgrid([CMGDB.TaggedDyadicCell(0,7,[i,j]) for i in range(128) for j in range(128)])
        atlas.set_map(lambda chart,b:[(0,full[tuple(ix(b))].tolist())]);am,ap=CMGDB.ComputeMorseGraph(atlas)
        def signatures(m):
            return {frozenset(tuple(ix(m.phase_space_box(i))) for i in m.morse_set(v)):int(v) for v in m.vertices()}
        sm,sa=signatures(morse),signatures(am);assert set(sm)==set(sa)
        mapping={sa[k]:sm[k] for k in sa}
        assert {(mapping[a],mapping[b]) for a,b in am.edges()}==set(morse.edges())
        evidence['atlas_full_rectangle_parity']=True
        # Independent sparse graph SCC calculation on the restricted 512 grid.
        n=max(r['n'] for r in record['resolutions'])
        _,am,ap,full,cut,active,lower,upper,ix=construct(n,True)
        offsets,targets=ap.csr_view();offsets=np.asarray(offsets);targets=np.asarray(targets)
        matrix=csr_matrix((np.ones(len(targets),dtype=np.uint8),targets,offsets),shape=(ap.num_vertices(),)*2)
        count,labels=connected_components(matrix,directed=True,connection='strong')
        sizes=np.bincount(labels,minlength=count);self_loops=matrix.diagonal().astype(bool)
        recurrent_ids=set(np.flatnonzero(sizes>1))|set(labels[self_loops])
        native=[set(map(int,am.morse_set(v))) for v in am.vertices()]
        independent=[set(np.flatnonzero(labels==i)) for i in recurrent_ids]
        assert {frozenset(s) for s in native}=={frozenset(s) for s in independent}
        evidence['restricted_scc_parity']=True
        evidence['restricted_components']=len(native)
        # A local coarsening is permitted only when every recurrent component
        # lies within four cells of the same interior numerical fixed point.
        inside=[r for r in record['point_checks']['roots'] if r['inside']]
        boxes=np.asarray([am.phase_space_box(i) for v in am.vertices() for i in am.morse_set(v)])
        side=(upper-lower)/n;centers=(boxes[:,:2]+boxes[:,2:])/2
        distances=np.linalg.norm((centers-np.asarray(inside[0]['q']))/side,axis=1) if len(inside)==1 else np.array([np.inf])
        if len(inside)==1 and distances.max()<=4:
            selected=set(map(int,CMGDB.MorseDirectedPathCells(ap,am,am.vertices(),am.vertices())))
            full_model,fm,fp,full,cut,active,lower,upper,ix=construct(n)
            lookup={tuple(ix(fm.phase_space_box(i))):i for i in range(fp.num_vertices())}
            keys=[tuple(ix(am.phase_space_box(i))) for i in selected]
            assert all(np.allclose(full[k],cut[k],atol=1e-12,rtol=0) for k in keys)
            cells=[lookup[k] for k in keys]
            ann=list(CMGDB.ComputeConleyIndexForCells(full_model,fm,cells))
            graph=DerivedGraph(nodes=[0],edges={0:[]},labels={0:'('+', '.join(ann)+')'})
            dot=CMGDB.plot_derived_graph(graph)
            for fmt in ('png','pdf','svg'):(out/f'aggregate-{n}.{fmt}').write_bytes(dot.pipe(format=fmt))
            (out/f'aggregate-{n}.dot').write_text(dot.source)
            evidence['local_aggregate']=dict(n=n,nodes=1,annotations=ann,components_grouped=len(native),
                max_distance_cells=float(distances.max()),index_scope='Ambient local index pair recomputed on identical interior cell images',
                matches_analytical=False)
        else:evidence['local_aggregate']=None
    else:
        reference=json.loads((ROOT/'vendor/latent-equivariant/code/examples/data/sphere/exact-K128/native-result.json').read_text())
        names={0:'S',1:'T-',2:'E',3:'T+',4:'N'}
        expected_indices={n['geometric_name']:n['native_annotations'] for n in reference['nodes']}
        expected_edges={tuple(e) for e in reference['named_edges']}
        matches=[]
        for row in record['resolutions']:
            aggregate=row['aggregate']
            matches.append(len(aggregate['groups'])==5 and
                {(names[a],names[b]) for a,b in aggregate['edges']}==expected_edges and
                {names[int(k)]:v for k,v in aggregate['annotations'].items()}==expected_indices)
        evidence['analytical_annotated_graph_parity_all_grids']=all(matches)
        n=max(r['n'] for r in record['resolutions'])
        _,m,p,*_=construct(n)
        offsets,targets=p.csr_view();offsets=np.asarray(offsets);targets=np.asarray(targets)
        matrix=csr_matrix((np.ones(len(targets),dtype=np.uint8),targets,offsets),shape=(p.num_vertices(),)*2)
        count,labels=connected_components(matrix,directed=True,connection='strong')
        recurrent=set(np.flatnonzero(np.bincount(labels,minlength=count)>1))|set(labels[matrix.diagonal().astype(bool)])
        assert {frozenset(map(int,m.morse_set(v))) for v in m.vertices()}=={frozenset(np.flatnonzero(labels==i)) for i in recurrent}
        evidence['independent_scc_parity']=True
    env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'build/matplotlib'))
    subprocess.run([str(ROOT/'.venv-emlp/bin/python'),'-B',__file__,'--worker','--study',str(study),'--out',str(out)],env=env,check=True)
    local=json.loads((out/'local-displacement.json').read_text())
    evidence['local_checks_count']=len(local)
    evidence['nonroot_components_with_sampled_strict_drift']=sum(bool(x['sampled_strict_direction']) for x in local if not x['roots'])
    evidence['certified']=False
    (out/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    (points if args.worker else verify)(args.study.resolve(),args.out.resolve())
