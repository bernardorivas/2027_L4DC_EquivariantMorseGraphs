"""Geometric block families and sampled residual checks on nonperiodic 2D grids.

Blocks retain every cubical face. Distances are Euclidean distances for the supplied float64
boxes, not validated enclosures. Residual maxima describe the supplied finite sample only.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.ndimage import distance_transform_cdt

from .equivariant_graph import forward_closure, morse_decomposition, quotient_graph


def _geometry(grid):
    if grid.dim != 2 or any(grid.grid.periodic):
        raise ValueError("block certificates currently require a nonperiodic two-dimensional grid")
    return dict(shape=grid.shape.tolist(), lower=grid.lower.tolist(), upper=grid.upper.tolist(),
                action=dict(kind=grid.kind, elements=[[list(part) for part in g] for g in grid.group.elements],
                            identity=grid.identity, multiplication=grid.multiplication.tolist()))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _subgroup(grid, elements):
    h = tuple(sorted(set(int(x) for x in elements)))
    if not h or min(h) < 0 or max(h) >= len(grid.group) or grid.identity not in h:
        raise ValueError("invalid subgroup elements")
    if any(int(grid.multiplication[a, b]) not in h for a in h for b in h):
        raise ValueError("elements are not a subgroup")
    return h


class CubicalBlock:
    """A closed cubical subcomplex, including lower-dimensional intersections.

Cell coordinates use homology.Grid's convention. The mask has shape (2*nx+1,2*ny+1).
The open cell containing a point determines membership, with exact float comparisons at
physical grid vertices and no tolerance or clipping of points outside the grid.
"""

    def __init__(self, grid, cells, *, check=True):
        _geometry(grid)
        shape = tuple(2 * grid.shape + 1)
        a = np.asarray(cells, dtype=bool)
        if a.shape != shape:
            raise ValueError("cubical cell mask has the wrong shape")
        if check:
            for axis in range(2):
                odd = [slice(None)] * 2
                odd[axis] = slice(1, -1, 2)
                low, high = odd.copy(), odd.copy()
                low[axis], high[axis] = slice(0, -2, 2), slice(2, None, 2)
                if np.any(a[tuple(odd)] & ~(a[tuple(low)] & a[tuple(high)])):
                    raise ValueError("cell mask is not closed under faces")
        self.grid = grid
        self._cells = np.array(a, copy=True)
        self._cells.flags.writeable = False

    @classmethod
    def from_top_ids(cls, grid, ids):
        _geometry(grid)
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        indices = grid.indices(ids)
        a = np.zeros(tuple(2 * grid.shape + 1), dtype=bool)
        for dx in range(3):
            for dy in range(3):
                a[2 * indices[:, 0] + dx, 2 * indices[:, 1] + dy] = True
        return cls(grid, a, check=False)

    @classmethod
    def empty(cls, grid):
        return cls.from_top_ids(grid, [])

    @property
    def cells(self):
        out = self._cells.view()
        out.flags.writeable = False
        return out

    @property
    def top_ids(self):
        return np.flatnonzero(self._cells[1::2, 1::2].ravel())

    @property
    def empty_set(self):
        return not self._cells.any()

    @property
    def pure(self):
        return np.array_equal(self._cells, CubicalBlock.from_top_ids(self.grid, self.top_ids)._cells)

    @property
    def fingerprint(self):
        h = hashlib.sha256(json.dumps(_geometry(self.grid), sort_keys=True).encode())
        h.update(np.packbits(self._cells, bitorder="little").tobytes())
        return h.hexdigest()

    def _compatible(self, other):
        if not isinstance(other, CubicalBlock) or _geometry(self.grid) != _geometry(other.grid):
            raise ValueError("blocks use different physical grids or group actions")

    def union(self, other):
        self._compatible(other)
        return CubicalBlock(self.grid, self._cells | other._cells, check=False)

    def intersection(self, other):
        self._compatible(other)
        return CubicalBlock(self.grid, self._cells & other._cells, check=False)

    def is_subset(self, other):
        self._compatible(other)
        return not np.any(self._cells & ~other._cells)

    def transform(self, element):
        g = int(element)
        if not 0 <= g < len(self.grid.group):
            raise ValueError("invalid group element")
        axes = tuple(i for i, sign in enumerate(self.grid.group.elements[g][1]) if sign == -1)
        return CubicalBlock(self.grid, np.flip(self._cells, axis=axes), check=False)

    @property
    def stabilizer(self):
        return tuple(g for g in range(len(self.grid.group))
                     if np.array_equal(self._cells, self.transform(g)._cells))

    def contains(self, points):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("points must have shape (n,2)")
        valid = np.isfinite(p).all(axis=1)
        codes = []
        for axis in range(2):
            vertices = self.grid.lower[axis] + np.arange(self.grid.shape[axis] + 1) * self.grid.width[axis]
            insertion = np.searchsorted(vertices, p[:, axis], side="left")
            safe = np.clip(insertion, 0, len(vertices) - 1)
            on_vertex = p[:, axis] == vertices[safe]
            code = np.where(on_vertex, 2 * insertion, 2 * insertion - 1)
            valid &= (p[:, axis] >= vertices[0]) & (p[:, axis] <= vertices[-1])
            codes.append(np.clip(code, 0, 2 * self.grid.shape[axis]))
        out = np.zeros(len(p), dtype=bool)
        out[valid] = self._cells[codes[0][valid], codes[1][valid]]
        return out

    def summary(self):
        return dict(hash=self.fingerprint, cells=int(self._cells.sum()),
                    top_cells=len(self.top_ids), pure=self.pure, stabilizer=list(self.stabilizer))


@dataclass(frozen=True)
class BlockFamily:
    blocks: tuple
    generators: tuple
    join_irreducibles: tuple
    predecessors: dict
    candidates: tuple
    source_hash: str | None
    fingerprint: str

    def manifest(self):
        data = dict(geometry=_geometry(self.blocks[0].grid),
                    blocks=[block.summary() for block in self.blocks],
                    generators=list(self.generators), join_irreducibles=list(self.join_irreducibles),
                    predecessors={str(k): v for k, v in self.predecessors.items()},
                    candidates=list(self.candidates), source_hash=self.source_hash)
        if _hash(data) != self.fingerprint:
            raise ValueError("the frozen block family changed")
        return dict(data, hash=self.fingerprint)


def generated_block_family(generators, *, candidates=(), source_hash=None, max_elements=256):
    """Generate by geometric binary union and intersection, without adjoining a top.

The limit is a resource stop, not permission to omit a block. The empty set is retained.
The predecessor of a nonempty join-irreducible is its unique maximal proper subcomplex in
this finite lattice. Hash this result before supplying independently registered samples.
"""
    generators = tuple(generators)
    if not generators:
        raise ValueError("at least one candidate is needed to identify the grid")
    if max_elements < 1:
        raise ValueError("max_elements must be positive")
    first = generators[0]
    for block in generators:
        first._compatible(block)
    known = {}
    queue = []

    def add(block):
        key = block.fingerprint
        if key not in known:
            if len(known) >= max_elements:
                raise MemoryError("generated geometric block family exceeds its element budget")
            known[key] = block
            queue.append(block)

    add(CubicalBlock.empty(first.grid))
    for block in generators:
        add(block)
    cursor = 0
    while cursor < len(queue):
        block = queue[cursor]
        cursor += 1
        for other in list(known.values()):
            add(block.union(other))
            add(block.intersection(other))
    blocks = tuple(sorted(known.values(), key=lambda b: (int(b._cells.sum()), b.fingerprint)))
    ids = {b.fingerprint: i for i, b in enumerate(blocks)}
    less = {(i, j) for i, a in enumerate(blocks) for j, b in enumerate(blocks)
            if i != j and a.is_subset(b)}
    predecessors = {}
    for j, block in enumerate(blocks):
        if block.empty_set:
            continue
        below = [i for i in range(len(blocks)) if (i, j) in less]
        covers = [i for i in below if not any((i, k) in less for k in below)]
        if len(covers) == 1:
            predecessors[j] = covers[0]
    data = dict(geometry=_geometry(first.grid), blocks=[block.summary() for block in blocks],
                generators=[ids[g.fingerprint] for g in generators],
                join_irreducibles=list(predecessors),
                predecessors={str(k): v for k, v in predecessors.items()},
                candidates=list(candidates), source_hash=source_hash)
    return BlockFamily(blocks, tuple(data["generators"]), tuple(predecessors), predecessors,
                       tuple(candidates), source_hash, _hash(data))


def principal_block_family(outer, *, max_elements=256):
    """The fixed quotient principal-downset closures and their geometric lattice.

An escaped candidate remains in candidates with escaped=True. Its geometric part is retained
when generating the family, but it is not a certified block. Quotient indices and active
upstairs positions are converted to physical top-cell IDs explicitly. The quotient is rebuilt
from this outer approximation so a stale relation cannot select the candidates.
"""
    grid = outer.equivariant_grid
    _geometry(grid)
    q = quotient_graph(outer)
    decomposition = q.morse_decomposition()
    if not decomposition.components:
        return generated_block_family([CubicalBlock.empty(grid)], source_hash=_outer_hash(outer),
                                      max_elements=max_elements)
    blocks, candidates = [], []
    for node in range(len(decomposition.components)):
        downset = [node] + [target for source, target in decomposition.order if source == node]
        seeds = np.concatenate([decomposition.components[i] for i in sorted(downset)])
        reached = forward_closure(q.adjacency, seeds)
        escaped = q.outside is not None and bool(np.any(reached == q.outside))
        positions = np.flatnonzero(np.isin(q.orbit_index[:outer.outside], reached))
        block = CubicalBlock.from_top_ids(grid, outer.cell_ids[positions])
        blocks.append(block)
        candidates.append(dict(node=node, downset=sorted(downset), escaped=escaped,
                               block_hash=block.fingerprint,
                               quotient_cells=[int(x) for x in reached if x != q.outside]))
    return generated_block_family(blocks, candidates=candidates, source_hash=_outer_hash(outer),
                                  max_elements=max_elements)


def separate_attractor_blocks(outer):
    """Fixed forward closures of the minimal upstairs recurrent components.

Every candidate is retained, including one reaching the outside node. Stabilizers are those
of the realized closed cubical blocks, not inferred equilibrium isotropy.
"""
    grid = outer.equivariant_grid
    _geometry(grid)
    adjacency = outer.adjacency()
    decomposition = morse_decomposition(adjacency, outer.outside)
    nonminimal = {source for source, _ in decomposition.order}
    candidates = []
    for node, component in enumerate(decomposition.components):
        if node in nonminimal:
            continue
        reached = forward_closure(adjacency, component)
        escaped = bool(np.any(reached == outer.outside))
        positions = reached[reached != outer.outside]
        block = CubicalBlock.from_top_ids(grid, outer.cell_ids[positions])
        candidates.append(dict(component=node, block=block, block_hash=block.fingerprint,
                               escaped=escaped, stabilizer=block.stabilizer))
    return tuple(candidates)


def _outer_hash(outer):
    h = hashlib.sha256(json.dumps(_geometry(outer.equivariant_grid), sort_keys=True).encode())
    for array in [outer.cell_ids, outer.image_lower, outer.image_upper, outer.index_lower, outer.index_upper]:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def boundary_segments(block):
    """Closed boundary segments of the top-cell union, in ambient Euclidean coordinates."""
    grid = block.grid
    mask = block._cells[1::2, 1::2]
    padded = np.pad(mask, 1)
    segments = []
    for axis, boundary in enumerate([padded[:-1, 1:-1] != padded[1:, 1:-1],
                                     padded[1:-1, :-1] != padded[1:-1, 1:]]):
        indices = np.column_stack(np.nonzero(boundary))
        lo = grid.lower + indices * grid.width
        hi = lo.copy()
        hi[:, 1-axis] += grid.width[1-axis]
        segments.append(np.stack([lo, hi], axis=1))
    return np.concatenate(segments)


def _box_boundary_min(lower, upper, segments, pair_budget=1_000_000):
    result = np.inf
    if not len(segments):
        return 0.0
    batch = max(1, pair_budget // len(segments))
    for start in range(0, len(lower), batch):
        lo, hi = lower[start:start+batch], upper[start:start+batch]
        delta = np.maximum(0, np.maximum(segments[None, :, 0] - hi[:, None],
                                         lo[:, None] - segments[None, :, 1]))
        result = min(result, float(np.sqrt(np.sum(delta*delta, axis=2)).min()))
    return result


def box_complement_distance(block, lower, upper):
    """Minimum distance of a finite union of boxes to cl(R2 minus the block).

Connected boxes are inside if a center is inside and no boundary is met. The chessboard
transform supplies only a lower bound for lossless pruning. Final distances use Euclidean
box-to-segment distances. Corners, holes, anisotropic cells, and the exterior are retained.
"""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != hi.shape or lo.ndim != 2 or lo.shape[1] != 2:
        raise ValueError("box arrays must have shape (n,2)")
    if not np.isfinite([lo, hi]).all() or np.any(lo > hi):
        raise ValueError("box bounds must be finite and ordered")
    if not len(lo):
        return float("inf")
    center = (lo+hi)/2
    if not CubicalBlock.from_top_ids(block.grid, block.top_ids).contains(center).all() or not len(block.top_ids):
        return 0.0
    segments = boundary_segments(block)
    mask = block._cells[1::2, 1::2]
    cdt = distance_transform_cdt(np.pad(mask, 1), metric="chessboard")[1:-1, 1:-1]
    indices = np.floor((center-block.grid.lower)/block.grid.width).astype(np.int64)
    indices = np.clip(indices, 0, block.grid.shape-1)
    distance_lower = np.maximum(0, (cdt[indices[:, 0], indices[:, 1]]-1)*block.grid.width.min()
                                - np.linalg.norm((hi-lo)/2, axis=1))
    order = np.argsort(distance_lower, kind="stable")
    initial = order[:min(128, len(order))]
    result = _box_boundary_min(lo[initial], hi[initial], segments)
    rest = order[len(initial):]
    rest = rest[distance_lower[rest] < result]
    if len(rest):
        result = min(result, _box_boundary_min(lo[rest], hi[rest], segments))
    return result


def tolerance_2d(outer, block):
    """A lower tolerance estimate from stored padded image boxes, with explicit failures."""
    if _geometry(outer.equivariant_grid) != _geometry(block.grid):
        raise ValueError("outer approximation and block use different grids")
    result = dict(block_hash=block.fingerprint, label="float64 enclosure, not validated",
                  ambient_metric="Euclidean R2", top_cells=len(block.top_ids), positive=False)
    if block.empty_set:
        return dict(result, lower=float("inf"), empty=True, positive=True, reason="empty block")
    if not block.pure:
        return dict(result, lower=0.0, reason="lower-dimensional cells outside the top-cell closure are unenclosed")
    try:
        positions = outer.positions(block.top_ids)
    except ValueError:
        return dict(result, lower=0.0, reason="block contains cells outside the active enclosure domain")
    targets = np.unique(outer.adjacency()[positions].indices)
    if outer.outside in targets:
        return dict(result, lower=0.0, escaped=True, reason="image reaches the outside node")
    target_ids = outer.cell_ids[targets]
    allowed = block._cells[1::2, 1::2].ravel()
    if not allowed[target_ids].all():
        return dict(result, lower=0.0, reason="block is not forward invariant under the stored relation")
    lower = box_complement_distance(block, outer.image_lower[positions], outer.image_upper[positions])
    return dict(result, lower=lower, positive=bool(lower > 0), escaped=False,
                reason=None if lower > 0 else "an image box meets the geometric complement closure")


def sampled_tolerance_upper(block, source, images):
    """Sampled upper estimate of the tolerance, requiring every sampled source in the block."""
    source, images = np.asarray(source, float), np.asarray(images, float)
    if source.shape != images.shape or not block.contains(source).all():
        raise ValueError("sample sources must all belong to the block")
    if not len(source):
        return dict(upper=None, count=0, label="sampled", checked=False)
    return dict(upper=box_complement_distance(block, images, images), count=len(source),
                label="sampled", checked=True)


def residual_values(grid, predicted, encoded_true, subgroup=None):
    """Plain, full-group, and subgroup residuals for supplied finite samples."""
    _geometry(grid)
    a, b = np.asarray(predicted, float), np.asarray(encoded_true, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite([a, b]).all():
        raise ValueError("residual arrays must be finite with shape (n,2)")
    subgroup = _subgroup(grid, [grid.identity] if subgroup is None else subgroup)
    distances = np.stack([np.linalg.norm(a-grid.act_boxes(g, b, b)[0], axis=1)
                          for g in range(len(grid.group))])
    return dict(plain=np.linalg.norm(a-b, axis=1), modulo_group=distances.min(axis=0),
                stabilizer=distances[list(subgroup)].min(axis=0))


def sample_residuals(block, encoded, predicted, encoded_true, *, domain_mask, subgroup=None,
                     tolerance_lower=None):
    """Residual statistics only on samples in the declared X and in E^-1(block).

No saturation or clipping changes that domain. A full-group residual is a block certificate
only for a full-group invariant block. By default H is its actual stabilizer. Empty nonempty-
block samples cannot pass. This function evaluates the entire supplied finite sample.
"""
    z, a, b = [np.asarray(x, float) for x in (encoded, predicted, encoded_true)]
    domain = np.asarray(domain_mask)
    if z.shape != a.shape or z.shape != b.shape or domain.dtype != bool or domain.shape != (len(z),):
        raise ValueError("sample arrays or explicit domain mask do not agree")
    if not np.isfinite(z[domain]).all():
        raise ValueError("encoded samples in X must be finite")
    if tolerance_lower is not None and not block.empty_set and not np.isfinite(tolerance_lower):
        raise ValueError("a nonempty block needs a finite tolerance estimate")
    actual = block.stabilizer
    h = _subgroup(block.grid, actual if subgroup is None else subgroup)
    if not set(h) <= set(actual):
        raise ValueError("the proposed subgroup does not preserve the block")
    selected = np.flatnonzero(domain & block.contains(z))
    eligible = dict(plain=True, modulo_group=len(actual) == len(block.grid.group), stabilizer=True)
    report = dict(block_hash=block.fingerprint, count=len(selected), selected_indices=selected,
                  subgroup=h, block_stabilizer=actual, eligible=eligible, label="sampled",
                  total_samples=len(z), domain_samples=int(domain.sum()))
    if not len(selected):
        return dict(report, maxima={k: None for k in eligible}, margins={k: None for k in eligible},
                    passed={k: None if block.empty_set else False for k in eligible},
                    reason="empty block" if block.empty_set else "no samples in the required domain")
    values = residual_values(block.grid, a[selected], b[selected], h)
    maxima = {k: float(v.max()) for k, v in values.items()}
    margins = {k: None if tolerance_lower is None else float(tolerance_lower-maxima[k]) for k in values}
    passed = {k: bool(eligible[k] and margins[k] is not None and margins[k] > 0) for k in values}
    return dict(report, maxima=maxima, margins=margins, passed=passed,
                worst_indices={k: int(selected[np.argmax(v)]) for k, v in values.items()}, reason=None)


def representative_indices(points, action, elements):
    """One supplied point per exact orbit under the specified subgroup.

Call on full-state samples already restricted to X intersect E^-1(N), using H <= G_N.
The returned indices select actual supplied points. Latent coordinates cannot identify
full-state orbits. Equality is exact on the supplied floating-point arrays.
"""
    points = np.asarray(points, float)
    if points.ndim != 2 or not np.isfinite(points).all() or not len(elements):
        raise ValueError("finite point array and nonempty subgroup are required")
    canonical = points.copy()
    for g in elements:
        transformed = np.asarray(action(g, points), float)
        if transformed.shape != points.shape or not np.isfinite(transformed).all():
            raise ValueError("action returned an invalid point array")
        equal = np.ones(len(points), dtype=bool)
        smaller = np.zeros(len(points), dtype=bool)
        for axis in range(points.shape[1]):
            smaller |= equal & (transformed[:, axis] < canonical[:, axis])
            equal &= transformed[:, axis] == canonical[:, axis]
        canonical[smaller] = transformed[smaller]
    _, indices = np.unique(canonical, axis=0, return_index=True)
    return np.sort(indices)


def block_representative_indices(block, full_states, encoded, action, *, domain_mask, subgroup=None):
    """Restrict full-state samples to the actual block domain before H-orbit reduction."""
    states, z = np.asarray(full_states, float), np.asarray(encoded, float)
    domain = np.asarray(domain_mask)
    if states.ndim != 2 or len(states) != len(z) or domain.dtype != bool or domain.shape != (len(z),):
        raise ValueError("full-state samples and explicit domain mask do not agree")
    if not np.isfinite(z[domain]).all():
        raise ValueError("encoded samples in X must be finite")
    actual = block.stabilizer
    h = _subgroup(block.grid, actual if subgroup is None else subgroup)
    if not set(h) <= set(actual):
        raise ValueError("sample reduction subgroup does not preserve the block")
    indices = np.flatnonzero(domain & block.contains(z))
    chosen = representative_indices(states[indices], action, h)
    return indices[chosen]
