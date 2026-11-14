"""Uniform grids, images of cells, and Morse graphs computed with CMGDB.

A grid is a uniform product grid of a box in R^k with 2^l cells per axis. Its cells
are listed in product order, the last coordinate varying fastest, as rows
(lower_1, ..., lower_k, upper_1, ..., upper_k). CMGDB is used as it is: it computes
the outer approximation from the supplied images, its Morse sets, their order, and
the Conley index of each Morse set over the field with five elements.
"""
from dataclasses import dataclass, field
import itertools

import numpy as np
import CMGDB

NATIVE_OPTIONS = dict(cache_transition_graph=True, cache_map_graph=True,
                      batch_chunk_size=512, reserve_min_edges=0)


def cells(per_axis, lower, upper):
    """The cells of the uniform grid of [lower, upper] with `per_axis` cells per axis."""
    if per_axis < 2 or per_axis & (per_axis - 1):
        raise ValueError("Use a power of two cells per axis.")
    lower, upper = np.atleast_1d(np.asarray(lower, float)), np.atleast_1d(np.asarray(upper, float))
    edges = [np.linspace(a, b, per_axis + 1) for a, b in zip(lower, upper, strict=True)]
    index = np.array(list(itertools.product(range(per_axis), repeat=len(lower))))
    low = np.column_stack([e[index[:, i]] for i, e in enumerate(edges)])
    high = np.column_stack([e[index[:, i] + 1] for i, e in enumerate(edges)])
    return np.column_stack((low, high))


def sampled_hulls(function, boxes, samples, padding, lower=None, upper=None):
    """Boxes containing the images of `samples` points per axis of each cell.

    The points include the corners of each cell. Each hull is padded by `padding`
    and rounded outward. Where `lower` or `upper` is given, the hull is cut at that
    face, which must be a face of the space itself, such as a pole. These hulls are
    sampled, not validated enclosures.
    """
    boxes = np.asarray(boxes, float)
    k = boxes.shape[1] // 2
    t = np.array(list(itertools.product(np.linspace(0., 1., samples), repeat=k)))
    points = boxes[:, None, :k] + (boxes[:, None, k:] - boxes[:, None, :k]) * t[None]
    values = np.asarray(function(points.reshape(-1, k)), float).reshape(len(boxes), len(t), k)
    low = np.nextafter(values.min(axis=1) - padding, -np.inf)
    high = np.nextafter(values.max(axis=1) + padding, np.inf)
    if lower is not None:
        low = np.maximum(low, np.asarray(lower, float))
    if upper is not None:
        high = np.minimum(high, np.asarray(upper, float))
    return np.column_stack((low, high))


def data_map(sources, images, lower, upper):
    """The map defined by sample pairs (x, f(x)): a cell maps to the hull of the images
    of the sources it contains, and a cell with no source maps outside the box."""
    return CMGDB.BoxMapData(np.asarray(sources, float), np.asarray(images, float), map_empty="outside",
                            lower_bounds=list(map(float, lower)), upper_bounds=list(map(float, upper)),
                            domain_padding=False, padding=False)


class CellIndex:
    """Lookup of the cells of a uniform product grid listed in any order."""

    def __init__(self, lower, upper):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if lower.ndim != 2 or lower.shape != upper.shape or not np.all(upper > lower):
            raise ValueError("cells need lower and upper corners of shape (n, k)")
        self.lower, self.upper = lower, upper
        self.domain_lower, self.domain_upper = lower.min(axis=0), upper.max(axis=0)
        self.side = (upper - lower)[0]
        if not np.allclose(upper - lower, self.side, rtol=1e-12, atol=0):
            raise ValueError("the cells do not have one size")
        position = (lower - self.domain_lower) / self.side
        coordinates = np.rint(position).astype(np.int64)
        if not np.allclose(position, coordinates, rtol=0, atol=1e-9):
            raise ValueError("the cells are not aligned")
        self.shape = tuple(int(c) for c in coordinates.max(axis=0) + 1)
        if int(np.prod(self.shape)) != len(lower):
            raise ValueError("the cells do not form a product grid")
        self.ids = np.full(self.shape, -1, np.int64)
        self.ids[tuple(coordinates.T)] = np.arange(len(lower))
        if np.any(self.ids < 0):
            raise ValueError("the cells do not form a product grid")

    def _range(self, low, high, slack):
        """Integer ranges of the closed cells that meet the boxes [low, high]."""
        if not 0 <= slack < .5:
            raise ValueError("the slack is less than half a cell")
        if np.any(np.isnan(low)) or np.any(np.isnan(high)):
            raise ValueError("a coordinate is not a number")
        limit = np.array(self.shape, float) + 1
        first = np.ceil(np.clip((low - self.domain_lower) / self.side - slack, -1, limit))
        last = np.floor(np.clip((high - self.domain_lower) / self.side + slack, -1, limit))
        return first.astype(np.int64) - 1, last.astype(np.int64)

    def containing(self, points, slack=1e-9):
        """Ids of the closed cells that contain each point, -1 where there is none.

        The result has shape (m, 2^k). A point on a face shared by several cells
        gets all of them; the slack, in units of the cell side, also assigns a point
        to a neighbor of its cell when it is that close to the shared face.
        """
        points = np.asarray(points, float)
        first, last = self._range(points, points, slack)
        out = np.full((len(points), 2 ** points.shape[1]), -1, np.int64)
        shape = np.array(self.shape)
        for column, choice in enumerate(itertools.product((0, 1), repeat=points.shape[1])):
            index = np.where(np.array(choice, bool), last, first)
            valid = np.all((index >= 0) & (index < shape), axis=1)
            out[valid, column] = self.ids[tuple(index[valid].T)]
        return out

    def points_in(self, points, cells, slack=1e-9):
        """Whether each point lies in the union of the closed cells of the set."""
        mask = np.append(as_mask(cells, len(self.lower)), False)
        return mask[self.containing(points, slack)].any(axis=1)


def as_mask(cells, count):
    """A set of cells, given by a boolean mask or by distinct ids."""
    cells = np.asarray(cells)
    if cells.dtype == bool:
        if cells.shape != (count,):
            raise ValueError("a mask needs one entry for each cell")
        return cells.copy()
    mask = np.zeros(count, bool)
    if cells.size:
        if not np.issubdtype(cells.dtype, np.integer):
            raise ValueError("cells are given by a boolean mask or by their ids")
        if cells.min() < 0 or cells.max() >= count or len(np.unique(cells)) != cells.size:
            raise ValueError("ids of cells are distinct and lie in the grid")
        mask[cells] = True
    return mask


@dataclass
class MorseGraph:
    """A Morse graph of the outer approximation on a grid, in the grid's own cell order.

    `nodes` lists each Morse set with its cells and its Conley index annotation,
    `edges` the order between Morse sets, `membership` the Morse set of each cell
    (-1 outside every Morse set), and `offsets`, `targets` the outer approximation
    as a compressed relation on cells. `model`, `morse` and `phase` are the CMGDB
    objects, kept for index computations on other sets of cells.
    """
    boxes: np.ndarray
    images: np.ndarray
    nodes: list
    edges: list
    membership: np.ndarray
    offsets: np.ndarray
    targets: np.ndarray
    order: np.ndarray
    model: object = field(repr=False)
    morse: object = field(repr=False)
    phase: object = field(repr=False)
    merged: dict = field(default_factory=dict)

    @property
    def minimal(self):
        """The Morse sets with no outgoing edge."""
        return sorted({row["node"] for row in self.nodes} - {a for a, _ in self.edges})

    def record(self):
        return dict(cells=len(self.boxes), nodes=[{k: v for k, v in row.items()} for row in self.nodes],
                    edges=self.edges, minimal=self.minimal, coefficient_field=5)


def morse_graph(boxes, images=None, *, data_map=None):
    """Compute the Morse graph of the outer approximation defined by cell images.

    Give either `images`, one image box per cell in the order of `boxes`, or a
    native `data_map`. The grid must be a uniform product grid with a power of two
    cells in total, as `cells` builds it.
    """
    boxes = np.asarray(boxes, float)
    k = boxes.shape[1] // 2
    count = len(boxes)
    if count < 2 or count & (count - 1):
        raise ValueError("Use a product grid with a power-of-two cell count.")
    depth = count.bit_length() - 1
    lower, upper = boxes[:, :k].min(axis=0).tolist(), boxes[:, k:].max(axis=0).tolist()
    if data_map is None:
        images = np.asarray(images, float)
        if images.shape != boxes.shape or not np.isfinite(images).all():
            raise ValueError("Each cell needs a finite image box.")
        lookup = {tuple(box): image.tolist() for box, image in zip(boxes, images, strict=True)}
        model = CMGDB.Model(depth, depth, depth, 10000, lower, upper, lambda box: lookup[tuple(box)])
        model.set_batch_map(lambda batch: [lookup[tuple(box)] for box in batch])
    else:
        model = CMGDB.Model(depth, depth, depth, 10000, lower, upper, data_map)
        model.set_batch_map(data_map.batch)
        images = np.asarray(data_map.batch(boxes), float)
    morse, phase = CMGDB.ComputeConleyMorseGraph(model, **NATIVE_OPTIONS)
    phase_boxes = np.asarray([morse.phase_space_box(i) for i in range(count)])
    index = CellIndex(boxes[:, :k], boxes[:, k:])
    # position i of the grid order is CMGDB cell order[i]
    centers = (phase_boxes[:, :k] + phase_boxes[:, k:]) / 2
    grid_of_native = index.containing(centers, slack=0)[:, 0]
    if np.any(grid_of_native < 0) or len(np.unique(grid_of_native)) != count:
        raise RuntimeError("The native grid does not match the supplied cells.")
    if not np.array_equal(phase_boxes, boxes[grid_of_native]):
        raise RuntimeError("The native cell boxes differ from the supplied grid.")
    order = np.argsort(grid_of_native)
    native_of_grid = order
    offsets, targets = [np.asarray(v, dtype=np.int64) for v in phase.csr_view()]
    # Relabel the relation from native cell ids to grid positions.
    sources = np.repeat(np.arange(count), np.diff(offsets))
    grid_sources, grid_targets = grid_of_native[sources], grid_of_native[targets]
    sort = np.lexsort((grid_targets, grid_sources))
    grid_sources, grid_targets = grid_sources[sort], grid_targets[sort]
    grid_offsets = np.concatenate(([0], np.cumsum(np.bincount(grid_sources, minlength=count))))
    membership = np.full(count, -1, dtype=int)
    nodes = []
    for node in morse.vertices():
        ids = sorted(int(grid_of_native[i]) for i in morse.morse_set(node))
        membership[ids] = node
        nodes.append(dict(node=int(node), cell_ids=ids, annotations=list(morse.annotations(node)),
                          hull=np.r_[boxes[ids, :k].min(axis=0), boxes[ids, k:].max(axis=0)].tolist()))
    edges = sorted([int(a), int(b)] for a, b in morse.edges())
    return MorseGraph(boxes=boxes, images=images, nodes=nodes, edges=edges, membership=membership,
                      offsets=grid_offsets, targets=grid_targets, order=native_of_grid,
                      model=model, morse=morse, phase=phase)


def order_convex(nodes, edges, group):
    """Whether no path of the Morse graph leaves the group and comes back to it."""
    adjacency = {node: set() for node in nodes}
    for a, b in edges:
        adjacency[a].add(b)
    def reach(start):
        seen, pending = set(), [start]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(adjacency[current])
        return seen
    reachable = {node: reach(node) for node in nodes}
    return not any(outside in reachable[source] and set(group) & reachable[outside]
                   for source in group for outside in set(nodes) - set(group))


def merge(graph, groups):
    """Merge prescribed groups of Morse sets, with all cells on paths between them.

    Each group is a list of Morse sets that is order convex in the Morse graph. Its
    cells are those on some directed path of the outer approximation from a cell of
    a member to a cell of a member, and its Conley index is recomputed by CMGDB on
    that set of cells, which raises ValueError if the set does not give an index
    pair. Returns a new MorseGraph whose nodes are the groups and the remaining
    Morse sets, with `merged` mapping each new node to its original Morse sets.
    """
    native = {row["node"]: row for row in graph.nodes}
    grouped = [sorted(group) for group in groups]
    used = [node for group in grouped for node in group]
    if len(used) != len(set(used)) or not set(used) <= set(native):
        raise ValueError("Groups are disjoint sets of Morse sets of the graph.")
    blocks = grouped + [[node] for node in sorted(native) if node not in used]
    blocks.sort(key=lambda block: block[0])
    new_of = {old: new for new, block in enumerate(blocks) for old in block}
    nodes, membership = [], np.full(len(graph.boxes), -1, dtype=int)
    k = graph.boxes.shape[1] // 2
    for new, block in enumerate(blocks):
        if len(block) == 1:
            ids = native[block[0]]["cell_ids"]
            annotations = native[block[0]]["annotations"]
        else:
            if not order_convex(list(native), graph.edges, block):
                raise ValueError(f"The group {block} is not order convex.")
            cells_native = CMGDB.MorseDirectedPathCells(graph.phase, graph.morse, block, block)
            ids = sorted(int(graph.order.argsort()[i]) for i in cells_native)
            annotations = list(CMGDB.ComputeConleyIndexForCells(graph.model, graph.morse,
                                                                [int(i) for i in cells_native]))
        membership[ids] = new
        nodes.append(dict(node=new, original_nodes=block, cell_ids=ids, annotations=annotations,
                          hull=np.r_[graph.boxes[ids, :k].min(axis=0), graph.boxes[ids, k:].max(axis=0)].tolist()))
    pairs = {(new_of[a], new_of[b]) for a, b in graph.edges if new_of[a] != new_of[b]}
    edges = sorted(list(pair) for pair in _transitive_reduction(len(blocks), pairs))
    return MorseGraph(boxes=graph.boxes, images=graph.images, nodes=nodes, edges=edges,
                      membership=membership, offsets=graph.offsets, targets=graph.targets,
                      order=graph.order, model=graph.model, morse=graph.morse, phase=graph.phase,
                      merged={new: block for new, block in enumerate(blocks)})


def _transitive_reduction(count, pairs):
    adjacency = {node: {b for a, b in pairs if a == node} for node in range(count)}
    def reach(start):
        seen, pending = set(), list(adjacency[start])
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(adjacency[current])
        return seen
    reachable = {node: reach(node) for node in range(count)}
    return {(a, b) for a, b in pairs
            if not any(b in reachable[c] for c in adjacency[a] if c != b)}
