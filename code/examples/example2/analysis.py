"""Native graphs and physical preimages of sampled O(3) data maps."""

from collections import deque

from .training import LOWER, UPPER, read, write
import numpy as np
from scipy.sparse import csr_matrix
import CMGDB

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#A6761D"]


def grid(n, lower, upper):
    a = np.linspace(lower[0], upper[0], n+1)
    r = np.linspace(lower[1], upper[1], n+1)
    i,j = np.indices((n,n))
    return np.column_stack((a[i.ravel()],r[j.ravel()],a[i.ravel()+1],r[j.ravel()+1]))


def bin_indices(points, n, lower, upper):
    scaled = (points-lower)/(upper-lower)*n
    inside = np.all((scaled >= 0)&(scaled <= n), axis=1)
    indices = np.clip(np.floor(scaled).astype(int), 0, n-1)
    return indices[:,0]*n+indices[:,1], inside


def graph_files(morse, folder, count):
    graph = CMGDB.PlotMorseGraph(morse,clist=[COLORS[i%len(COLORS)] for i in range(max(1,count))])
    (folder/"native-morse-graph.dot").write_text(graph.source)
    for kind in ("pdf","png"):
        (folder/f"native-morse-graph.{kind}").write_bytes(graph.pipe(format=kind))
    CMGDB.SaveMorseSets(morse,str(folder/"native-morse-sets.csv"))


def native(folder, boxes, images, lower, upper, n, status, support_counts, box_map, construction, replay=False):
    replay = replay or (folder/"native-results.json").exists()
    folder.mkdir(parents=True,exist_ok=True)
    original_images = images.copy()
    support = support_counts>0
    assert isinstance(box_map, CMGDB.BoxMapData)
    assert box_map.map_empty == "outside" and not box_map.domain_padding and not box_map.padding
    assert np.all(images[~support,:2] > upper), "Empty data cells must map outside the rectangle"
    depth = 2*(n.bit_length()-1)
    model = CMGDB.Model(depth,depth,depth,10000,lower.tolist(),upper.tolist(),box_map)
    model.set_batch_map(box_map.batch)
    morse, phase = CMGDB.ComputeConleyMorseGraph(model,cache_transition_graph=True,cache_map_graph=True,
                                               batch_chunk_size=512,reserve_min_edges=0)
    phase_boxes = np.array([morse.phase_space_box(i) for i in range(phase.num_vertices())])
    assert len(phase_boxes)==n*n
    order, inside = bin_indices((phase_boxes[:,:2]+phase_boxes[:,2:])/2,n,lower,upper)
    assert inside.all() and len(np.unique(order))==n*n
    active = support[order]
    phase_images = images[order]
    physical_images = original_images[order]
    assert np.array_equal(np.asarray(box_map.batch(phase_boxes)),phase_images)
    offsets, targets = [np.asarray(x) for x in phase.csr_view()]
    rows = np.repeat(np.arange(n*n,dtype=np.int64),np.diff(offsets).astype(int))
    unknown_edges = active[rows]&~active[targets]
    exit_sources = np.unique(rows[unknown_edges])
    reverse = csr_matrix((np.ones(len(targets),dtype=np.int8),(targets.astype(int),rows)),shape=(n*n,n*n))
    can_reach_unknown = np.zeros(n*n,dtype=bool)
    can_reach_unknown[exit_sources]=True
    queue=deque(map(int,exit_sources))
    while queue:
        dst=queue.popleft()
        for src in reverse.indices[reverse.indptr[dst]:reverse.indptr[dst+1]]:
            if active[src] and not can_reach_unknown[src]:
                can_reach_unknown[src]=True; queue.append(int(src))
    membership = np.full(n*n,-1,dtype=int)
    nodes=[]
    for node in morse.vertices():
        ids=sorted(map(int,morse.morse_set(node)))
        membership[ids]=node
        nodes.append(dict(node=int(node),cell_ids=ids,cells=len(ids),annotations=list(morse.annotations(node)),
              active_cells=int(np.sum(active[ids])),can_reach_unsampled_cell=bool(can_reach_unknown[ids].any()),
              hull=np.r_[phase_boxes[ids,:2].min(axis=0),phase_boxes[ids,2:].max(axis=0)].tolist()))
        assert active[ids].all(), "An unsupported source became a recurrent component"
    record=dict(grid_shape=[n,n],domain_lower=lower.tolist(),domain_upper=upper.tolist(),
                nodes=nodes,edges=sorted([list(map(int,e)) for e in morse.edges()]),
                raw_native_node_count=len(nodes),image_status=status,occupancy_restricted=True,
                coefficient_field=5,annotation_space="quotient coordinates only",
                active_cells=int(active.sum()),inactive_cells=int((~active).sum()),
                known_to_unsampled_edges=int(unknown_edges.sum()),
                known_sources_with_unsampled_targets=len(exit_sources),
                occupancy_rule="at least one encoded physical-section source sample in the closed cell",
                unknown_boundary="BoxMapData map_empty=outside; no interpolation or domain padding",
                evidence_status="numerical approximation",construction=construction)
    if replay:
        saved=read(folder/"native-results.json")
        assert record==saved,folder
        with np.load(folder/"native-phase.npz",allow_pickle=False) as old:
            assert np.array_equal(old["boxes"],phase_boxes)
            difference=float(np.max(np.abs(old["image_boxes"]-phase_images)))
            assert difference<2e-13
            assert np.array_equal(old["morse_node"],membership)
        write(folder/"replay-verification.json",dict(passed=True,maximum_image_endpoint_difference=difference,
                                                    cells_edges_annotations_and_occupancy_match=True))
        if not (folder/"native-morse-graph.png").exists():
            graph_files(morse,folder,len(nodes))
    else:
        write(folder/"native-results.json",record)
        np.savez_compressed(folder/"native-phase.npz",boxes=phase_boxes,image_boxes=phase_images,
               unmasked_image_boxes=physical_images,morse_node=membership,active=active,
               occupancy_counts=support_counts[order],
               can_reach_unknown=can_reach_unknown,csr_offsets=offsets,csr_targets=targets,
               unknown_source_ids=rows[unknown_edges],unknown_target_ids=targets[unknown_edges])
        graph_files(morse,folder,len(nodes))
    print(f"{folder}: {len(nodes)} native nodes, {int(active.sum())} active cells, {int(unknown_edges.sum())} unknown edges",flush=True)
    return record,model,morse,phase,phase_boxes


def physical_preimages(model, folder, n, geometry):
    with np.load(geometry,allow_pickle=False) as saved:
        a,r,codes=saved["physical_a"],saved["physical_r"],saved["latent_codes"].reshape(-1,2)
    bins,inside=bin_indices(codes,n,LOWER,UPPER)
    with np.load(folder/"native-phase.npz",allow_pickle=False) as data:
        pboxes,membership,active=data["boxes"],data["morse_node"],data["active"]
    order,_=bin_indices((pboxes[:,:2]+pboxes[:,2:])/2,n,LOWER,UPPER)
    node_by_bin=np.full(n*n,-1,int); active_by_bin=np.zeros(n*n,bool)
    node_by_bin[order]=membership; active_by_bin[order]=active
    labels=node_by_bin[bins]; labels[~inside]=-1
    supported=active_by_bin[bins]&inside
    np.savez_compressed(folder/"physical-preimages.npz",physical_a=a,physical_r=r,
          latent_codes=codes.reshape(len(a),len(r),2),node_labels=labels.reshape(len(a),len(r)),
          active=supported.reshape(len(a),len(r)),latent_cell_ids=bins.reshape(len(a),len(r)))
