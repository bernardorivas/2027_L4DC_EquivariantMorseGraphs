"""Native singleton sink basins and O-to-C path cells for the saved experiment."""

from .training import digest, read, write
from .analysis import bin_indices
import numpy as np
import CMGDB


def component_at(point, boxes, membership):
    ids=np.flatnonzero(np.all((boxes[:,:2]<=point)&(point<=boxes[:,2:]),axis=1))
    nodes=set(map(int,membership[ids]))-{-1}
    if len(nodes)!=1:raise ValueError(f"Expected one component at {point}; found {nodes}")
    return nodes.pop()


def counts(selected, basin, sink_ids, unknown, active, path):
    return dict(cells=int(selected.sum()),
          minus=int(np.sum(selected&(basin==sink_ids[0]))),
          plus=int(np.sum(selected&(basin==sink_ids[1]))),
          unresolved=int(np.sum(selected&(basin<0)&active)),
          unknown=int(np.sum(selected&unknown)),
          unsupported=int(np.sum(selected&~active)),
          path_cells=int(np.sum(selected&path)))


def compute(folder, radial_root, replay=False, baseline=None):
    record=read(folder/"native-results.json")
    with np.load(folder/"native-phase.npz",allow_pickle=False) as data:
        boxes,images,membership,active,unknown=[data[key] for key in
               ("boxes","image_boxes","morse_node","active","can_reach_unknown")]
        saved_offsets,saved_targets=data["csr_offsets"],data["csr_targets"]
    lower,upper=np.array(record["domain_lower"]),np.array(record["domain_upper"])
    n=record["grid_shape"][0]
    lookup=dict(zip(map(tuple,boxes),images.tolist(),strict=True))
    depth=2*(n.bit_length()-1)
    model=CMGDB.Model(depth,depth,depth,10000,lower.tolist(),upper.tolist(),lambda b:lookup[tuple(b)])
    model.set_batch_map(lambda batch:[lookup[tuple(b)] for b in batch])
    morse,phase=CMGDB.ComputeMorseGraph(model,cache_transition_graph=True,cache_map_graph=True,
                                      batch_chunk_size=512,reserve_min_edges=0)
    assert np.array_equal(boxes,np.array([morse.phase_space_box(i) for i in range(len(boxes))]))
    assert sorted(map(list,morse.edges()))==record["edges"]
    for row in record["nodes"]:
        assert sorted(map(int,morse.morse_set(row["node"])))==row["cell_ids"]
    offsets,targets=[np.asarray(x) for x in phase.csr_view()]
    assert np.array_equal(offsets,saved_offsets) and np.array_equal(targets,saved_targets)
    nodes={r["node"] for r in record["nodes"]}
    outgoing={source for source,_ in record["edges"]}
    sinks=sorted(nodes-outgoing,key=lambda node:float(boxes[membership==node,0].mean()))
    assert len(sinks)==2
    assert boxes[membership==sinks[0],2].max()<0 and boxes[membership==sinks[1],0].min()>0
    source=component_at(np.array([0.,0.]),boxes,membership)
    saddle=component_at(np.array([0.,radial_root]),boxes,membership)
    cells=np.arange(len(boxes),dtype=np.uint64)
    singleton=np.asarray(CMGDB.MorseSingletonReachability(phase,morse,cells))
    reach=np.asarray(CMGDB.MorseReachabilityMasks(phase,morse,cells))
    nonempty=reach!=0
    one=nonempty&((reach&(reach-np.uint64(1)))==0)
    assert np.array_equal(singleton>=0,one)
    assert np.array_equal(singleton==-1,~nonempty)
    assert np.array_equal(singleton==-2,nonempty&~one)
    assert np.array_equal(reach[one],np.left_shift(np.uint64(1),singleton[one].astype(np.uint64)))
    path_cells=np.asarray(CMGDB.MorseDirectedPathCells(phase,morse,[source],[saddle]),dtype=int)
    path=np.zeros(len(boxes),bool);path[path_cells]=True
    image=np.zeros(len(boxes),bool);image[np.unique(targets)]=True
    strict=np.isin(singleton,sinks)
    assert not np.any(strict&path)
    assert image[membership>=0].all() and image[path].all()
    # A singleton Morse label alone does not exclude paths that terminate at missing data.
    basin=np.full(len(boxes),-1,dtype=int)
    safe=strict&active&~unknown
    basin[safe]=singleton[safe]
    assert np.all(boxes[basin==sinks[0],2]<0)
    assert np.all(boxes[basin==sinks[1],0]>0)
    rows=np.repeat(np.arange(len(boxes)),np.diff(offsets).astype(int))
    safe_edges=safe[rows]
    assert np.all(basin[targets[safe_edges]]==basin[rows[safe_edges]])
    values=dict(boxes=boxes,singleton=singleton,reach_mask=reach,safe_sink_node=basin,
                strict_sink=strict,active=active,unknown=unknown,path_cells=path_cells,
                path_mask=path,image_mask=image,image_cells=np.unique(targets),sink_nodes=np.array(sinks))
    if baseline is not None and baseline.exists():
        with np.load(baseline,allow_pickle=False) as old:
            for key in ("boxes","singleton"):
                assert np.array_equal(old[key],values[key]),key
            for key in ("path_cells","image_cells"):
                assert np.array_equal(np.sort(old[key]),np.sort(values[key])),key
        write(folder/"basin-paper-verification.json",dict(passed=True,
              source_sha256=digest(baseline),singleton_path_and_image_cells_match=True))
    data_path=folder/"basins.npz"
    if replay:
        with np.load(data_path,allow_pickle=False) as previous:
            for key,value in values.items():assert np.array_equal(previous[key],value),key
        write(folder/"basin-replay-verification.json",dict(passed=True,all_native_labels_and_paths_match=True))
    else:np.savez_compressed(data_path,**values)
    summaries={name:counts(selected,basin,sinks,unknown,active,path) for name,selected in
               (("full_grid",np.ones(len(boxes),bool)),("active",active),("image_cover",image&active))}
    result=dict(method="CMGDB.MorseSingletonReachability, MorseReachabilityMasks, and MorseDirectedPathCells",
          sink_nodes={"A-":sinks[0],"A+":sinks[1]},origin_node=source,saddle_node=saddle,
          grid=n,counts=summaries,
          criterion="the only reachable Morse node is that sink, and no unsupported cell is reachable",
          image_cover="cells with an incoming native cell-graph edge, restricted to active cells for learned panels",
          phase_sha256=digest(folder/"native-phase.npz"),
          forward_invariance_of_safe_basin_cells_checked=True,
          evidence_status="exact finite-graph reachability; learned box images and domain coverage remain numerical")
    write(folder/"basins.json",result)
    physical_source=folder/"physical-preimages.npz"
    if physical_source.exists():
        with np.load(physical_source,allow_pickle=False) as saved:
            physical_a,physical_r=saved["physical_a"],saved["physical_r"]
            physical_bins,physical_inside=bin_indices(saved["latent_codes"].reshape(-1,2),n,lower,upper)
            assert physical_inside.all()
        order,inside=bin_indices((boxes[:,:2]+boxes[:,2:])/2,n,lower,upper)
        assert inside.all()
        by_bin={}
        for key,value in (("basin_node",basin),("active",active),("unknown",unknown),
                          ("path",path),("image",image),("morse_node",membership)):
            arranged=np.empty_like(value);arranged[order]=value
            by_bin[key]=arranged[physical_bins].reshape(len(physical_a),len(physical_r))
        np.savez_compressed(folder/"physical-basins.npz",physical_a=physical_a,physical_r=physical_r,**by_bin)
    c=summaries["active"]
    print(f"{folder}: A- {c['minus']}, A+ {c['plus']}, unresolved {c['unresolved']}, unknown {c['unknown']}",flush=True)
    return result
