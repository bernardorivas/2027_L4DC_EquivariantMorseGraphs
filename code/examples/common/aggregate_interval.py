"""Aggregate an interval Morse graph using the prescribed pole and sink structure."""

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix)/".cache/matplotlib"))

import CMGDB
import numpy as np
from CMGDB.morse_lattice import DerivedGraph, transitive_reduction

from . import figure_style as style
from .workflow import read, write


def require_order_convex(nodes, edges, group):
    adjacency = {node: set() for node in nodes}
    for a, b in edges:
        adjacency[a].add(b)
    reachable = {}
    for node in nodes:
        seen, pending = set(), [node]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(adjacency[current])
        reachable[node] = seen
    for source in group:
        for outside in set(nodes)-set(group):
            if outside in reachable[source] and set(group) & reachable[outside]:
                raise ValueError("The prescribed aggregate is not order convex.")


def aggregate(model, morse, phase, record, arrays, merge_interior):
    nodes = {row["node"]: row for row in record["nodes"]}
    group = {node for node, row in nodes.items() if row["interval"][0] > -1. and row["interval"][1] < 1.}
    merged_cells = set()
    groups = {node: [node] for node in nodes}
    representative = None
    if merge_interior:
        sinks = group & set(record["sink_nodes"])
        if len(sinks) != 1:
            raise ValueError("The coning structure requires one interior attracting component.")
        representative, = sinks
        if nodes[representative]["annotations"] != ["x-1", "0"]:
            raise ValueError("The coning component has an unexpected interval index.")
        for node in group-{representative}:
            if nodes[node]["annotations"] != ["0", "0"]:
                raise ValueError("An additional nontrivial interior component cannot be absorbed by this structure.")
        require_order_convex(set(nodes), record["edges"], group)
        if any(a in group and b not in group for a, b in record["edges"]):
            raise ValueError("The attracting-latitude aggregate has an outgoing Morse edge.")
        extras = group-{representative}
        if extras:
            merged_cells = set(map(int, CMGDB.MorseDirectedPathCells(phase, morse, sorted(extras), [representative])))
        else:
            merged_cells = set(nodes[representative]["cell_ids"])
        assert set().union(*(set(nodes[node]["cell_ids"]) for node in group)) <= merged_cells
        assert not any(merged_cells & set(row["cell_ids"]) for node, row in nodes.items() if node not in group)
        groups = {node: ([*sorted(group)] if node == representative else [node])
                  for node in sorted(nodes) if node not in group-{representative}}
    identifiers = {old: new for new, members in enumerate(groups.values()) for old in members}
    kept = list(groups)
    boxes = arrays["boxes"]
    aggregate_nodes, membership = [], np.full(len(boxes), -1, dtype=int)
    for new, old in enumerate(kept):
        members = groups[old]
        cells = sorted(merged_cells) if old == representative else nodes[old]["cell_ids"]
        annotation = list(CMGDB.ComputeConleyIndexForCells(model, morse, cells))
        expected = ["x-1", "0"] if old == representative else nodes[old]["annotations"]
        if annotation != expected:
            raise ValueError("The recomputed aggregate Conley index disagrees with the prescribed structure.")
        membership[cells] = new
        aggregate_nodes.append(dict(node=new, original_nodes=members, cell_ids=cells,
            interval=[float(boxes[cells, 0].min()), float(boxes[cells, 1].max())], annotations=annotation))
    edges = {node: set() for node in range(len(kept))}
    for a, b in record["edges"]:
        if identifiers[a] != identifiers[b]:
            edges[identifiers[a]].add(identifiers[b])
    edges = {node: sorted(values) for node, values in transitive_reduction(list(edges), edges).items()}
    edge_pairs = sorted([a, b] for a, values in edges.items() for b in values)
    sinks = sorted(node for node, values in edges.items() if not values)
    masks = np.asarray(CMGDB.MorseReachabilityMasks(phase, morse, np.arange(len(boxes), dtype=np.uint64)))
    labels = []
    for mask in masks:
        targets = {identifiers[node] for node in nodes if int(mask) & (1 << node)}
        labels.append(next(iter(targets)) if len(targets) == 1 else -1)
    basins = np.where(np.isin(labels, sinks), labels, -1)
    offsets, targets = arrays["csr_offsets"], arrays["csr_targets"]
    sources = np.repeat(np.arange(len(boxes)), np.diff(offsets).astype(int))
    known = basins[sources] >= 0
    assert np.all(basins[targets[known]] == basins[sources[known]])
    result = dict(record, nodes=aggregate_nodes, edges=edge_pairs, sink_nodes=sinks,
        basin_cells={str(node): int(np.sum(basins == node)) for node in sinks},
        unresolved_cells=int(np.sum(basins < 0)),
        aggregation=dict(structure="repelling poles and one attracting interior latitude" if merge_interior else "separate endpoint components",
            order_convex=True, includes_connecting_cells=True, index_pairs_checked=True,
            native_components=len(nodes), aggregate_components=len(kept),
            index_tool="CMGDB.ComputeConleyIndexForCells", path_tool="CMGDB.MorseDirectedPathCells"))
    return result, dict(arrays, membership=membership, basin_node=basins)


def save(result, arrays, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write(folder/"results.json", result)
    np.savez_compressed(folder/"phase.npz", **arrays)
    rows = [[*arrays["boxes"][cell], node["node"]] for node in result["nodes"] for cell in node["cell_ids"]]
    np.savetxt(folder/"morse-sets.csv", rows, delimiter=",", fmt=["%.17g", "%.17g", "%d"])
    graph = DerivedGraph(nodes=[node["node"] for node in result["nodes"]],
                         edges={row["node"]: [b for a, b in result["edges"] if a == row["node"]] for row in result["nodes"]},
                         labels={row["node"]: "("+", ".join(row["annotations"])+")" for row in result["nodes"]})
    picture = CMGDB.plot_derived_graph(graph, clist=[style.TINT[node] for node in graph.nodes])
    (folder/"morse-graph.dot").write_text(picture.source)
    for extension in ("pdf", "svg"):
        (folder/f"morse-graph.{extension}").write_bytes(picture.pipe(format=extension))


def from_saved(record, arrays, merge_interior):
    lookup = {tuple(box): image.tolist() for box, image in zip(arrays["boxes"], arrays["images"], strict=True)}
    depth = len(arrays["boxes"]).bit_length()-1
    model = CMGDB.Model(depth, depth, depth, 10000, [-1.], [1.], lambda box: lookup[tuple(box)])
    morse, phase = CMGDB.ComputeConleyMorseGraph(model, cache_transition_graph=True, cache_map_graph=True)
    assert sorted(map(list, morse.edges())) == record["edges"]
    for row in record["nodes"]:
        assert sorted(map(int, morse.morse_set(row["node"]))) == row["cell_ids"]
        assert list(morse.annotations(row["node"])) == row["annotations"]
    return aggregate(model, morse, phase, record, arrays, merge_interior)
