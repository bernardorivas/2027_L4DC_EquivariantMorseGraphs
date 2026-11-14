"""Attracting regions of minimal Morse sets, their tolerances, and residuals modulo G.

For a minimal Morse set M of a learned map, or a prescribed union of Morse sets, the
region is N = |A|, where A is the set of cells from which no path of the outer
approximation leaves the box or reaches a Morse set outside M. The tolerance of N is
estimated by the distance from the image boxes of the cells of A to the complement of
N, and the residual modulo G by its maximum over a sample of E^{-1}(N), both for the
map whose Morse graph is computed. All numbers are binary64 and not validated: image
boxes and residuals are sampled.
"""
import numpy as np

from .grids import CellIndex, as_mask

LABEL = "binary64, not validated"


class Euclidean:
    """The quotient metric is the Euclidean metric of the coordinates."""

    name = "euclidean"

    @staticmethod
    def point(u, w):
        return np.linalg.norm(np.asarray(u, float) - np.asarray(w, float), axis=-1)

    @staticmethod
    def lower(a_lo, a_hi, b_lo, b_hi):
        """The distance between two boxes."""
        gap = np.maximum(0.0, np.maximum(b_lo - a_hi, a_lo - b_hi))
        return np.linalg.norm(gap, axis=-1)


class SphereHeight:
    """Heights in [-1, 1] of the unit sphere modulo rotations about its axis, with the
    chordal metric: the distance between two points of one meridian."""

    name = "chordal"

    @staticmethod
    def _heights(*values):
        out = []
        for value in values:
            value = np.asarray(value, float)
            if value.shape[-1] != 1:
                raise ValueError("heights have one coordinate")
            value = value[..., 0]
            if np.any(np.isfinite(value) & (np.abs(value) > 1 + 1e-9)):
                raise ValueError("a height lies outside [-1, 1]")
            out.append(np.clip(value, -1.0, 1.0))
        return out

    @staticmethod
    def _chord(z, w):
        return np.hypot(z - w, np.sqrt((1 - z) * (1 + z)) - np.sqrt((1 - w) * (1 + w)))

    def point(self, u, w):
        return self._chord(*self._heights(u, w))

    def lower(self, a_lo, a_hi, b_lo, b_hi):
        a_lo, a_hi, b_lo, b_hi = self._heights(a_lo, a_hi, b_lo, b_hi)
        return np.where(b_lo > a_hi, self._chord(a_hi, b_lo),
                        np.where(a_lo > b_hi, self._chord(a_lo, b_hi), 0.0))


def reverse(offsets, targets):
    """The relation with every edge reversed, in the same compressed form."""
    offsets, targets = np.asarray(offsets, np.int64), np.asarray(targets, np.int64)
    count = len(offsets) - 1
    sources = np.repeat(np.arange(count), np.diff(offsets))
    order = np.argsort(targets, kind="stable")
    counts = np.bincount(targets, minlength=count)
    return np.concatenate(([0], np.cumsum(counts))), sources[order]


def forward_closure(offsets, targets, seeds):
    """The cells reachable from the seeds, the seeds included."""
    offsets, targets = np.asarray(offsets, np.int64), np.asarray(targets, np.int64)
    reached = as_mask(seeds, len(offsets) - 1)
    frontier = np.flatnonzero(reached)
    while len(frontier):
        pieces = [targets[offsets[i]:offsets[i + 1]] for i in frontier]
        new = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, np.int64)
        frontier = new[~reached[new]]
        reached[frontier] = True
    return reached


def is_forward_invariant(offsets, targets, cells):
    offsets, targets = np.asarray(offsets, np.int64), np.asarray(targets, np.int64)
    mask = as_mask(cells, len(offsets) - 1)
    sources = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    return bool(mask[targets[mask[sources]]].all())


def attracting_regions(graph, faces, available=None):
    """The region A of each minimal Morse set of a MorseGraph, as a label per cell.

    A cell belongs to the region of a minimal Morse set when that is the only Morse
    set it reaches. Cells without an image (`available` False), cells that can reach
    them, and cells whose image leaves the box through a face that is not a face of
    the space (`faces[i, side]` False) belong to no region. Returns the labels, -1
    for no region, after checking that each region is forward invariant.
    """
    boxes, images = graph.boxes, graph.images
    k = boxes.shape[1] // 2
    count = len(boxes)
    available = np.ones(count, bool) if available is None else np.asarray(available, bool)
    faces = np.asarray(faces, bool)
    reached = {}
    for row in graph.nodes:
        reached[row["node"]] = forward_closure(*reverse(graph.offsets, graph.targets), row["cell_ids"])
    minimal = graph.minimal
    labels = np.full(count, -1, dtype=int)
    for node in minimal:
        others = np.zeros(count, bool)
        for other, mask in reached.items():
            if other != node:
                others |= mask
        labels[reached[node] & ~others] = node
    outside = np.any(((images[:, :k] < boxes[:, :k].min(axis=0)) & ~faces[:, 0]) |
                     ((images[:, k:] > boxes[:, k:].max(axis=0)) & ~faces[:, 1]), axis=1)
    unsafe = forward_closure(*reverse(graph.offsets, graph.targets), ~available | outside)
    labels[unsafe] = -1
    sources = np.repeat(np.arange(count), np.diff(graph.offsets))
    known = labels[sources] >= 0
    if not np.all(labels[graph.targets[known]] == labels[sources[known]]):
        raise RuntimeError("An attracting region is not forward invariant.")
    return labels


def tolerance(boxes, images, cells, metric, faces, geometry=None):
    """The distance from the image boxes of the cells of the set to its complement.

    `faces[i, side]` is True where that face of the box is a face of the space
    itself; image boxes are cut there, and beyond the other faces the complement
    continues. `geometry` gives the cells in the coordinates of the metric when
    these differ from the grid coordinates. Returns the value and a witness: the
    cell whose image comes closest and the exposed face it comes closest to.
    """
    boxes = np.asarray(boxes, float)
    geometry = boxes if geometry is None else np.asarray(geometry, float)
    images = np.asarray(images, float)
    k = boxes.shape[1] // 2
    mask = as_mask(cells, len(boxes))
    if not mask.any():
        return dict(value=float("inf"), witness=None)
    index = CellIndex(boxes[:, :k], boxes[:, k:])
    positions = np.rint((boxes[:, :k] - index.domain_lower) / index.side).astype(int)
    exposed, owners = [], []
    for cell in np.flatnonzero(mask):
        for axis in range(k):
            for side in (0, 1):
                neighbor = positions[cell].copy()
                neighbor[axis] += 2 * side - 1
                inside = bool(np.all((neighbor >= 0) & (neighbor < np.array(index.shape))))
                if inside and mask[index.ids[tuple(neighbor)]]:
                    continue
                if not inside and faces[axis][side]:
                    continue
                low, high = geometry[cell, :k].copy(), geometry[cell, k:].copy()
                low[axis] = high[axis] = low[axis] if side == 0 else high[axis]
                exposed.append(np.r_[low, high])
                owners.append(int(cell))
    if not exposed:
        return dict(value=float("inf"), witness=None)
    exposed = np.asarray(exposed)
    lower_face = np.asarray(faces, bool)[:, 0]
    upper_face = np.asarray(faces, bool)[:, 1]
    best, witness = np.inf, None
    ids = np.flatnonzero(mask)
    for start in range(0, len(ids), 128):
        part = ids[start:start + 128]
        low = np.where(lower_face, np.maximum(images[part, :k], geometry[:, :k].min(axis=0)), images[part, :k])
        high = np.where(upper_face, np.minimum(images[part, k:], geometry[:, k:].max(axis=0)), images[part, k:])
        distance = metric.lower(low[:, None], high[:, None], exposed[None, :, :k], exposed[None, :, k:])
        row, column = np.unravel_index(distance.argmin(), distance.shape)
        if distance[row, column] < best:
            best = float(distance[row, column])
            witness = dict(cell=int(part[row]), exposed_face=exposed[column].tolist(),
                           face_of_cell=owners[column], image=images[part[row]].tolist())
    return dict(value=best, witness=witness)


def compare(graph, labels, codes, predicted, encoded_image, metric, faces, geometry=None):
    """Compare, for each minimal Morse set, the tolerance of its region with the residual.

    `codes` are the encoded samples E(x) in the grid coordinates, `predicted` the
    learned map applied to them and `encoded_image` the encoding E(F(x)) of the
    true image, both in the coordinates of the metric. The residual is maximized
    over the samples whose code lies in the region.
    """
    k = graph.boxes.shape[1] // 2
    index = CellIndex(graph.boxes[:, :k], graph.boxes[:, k:])
    codes = np.asarray(codes, float).reshape(-1, k)
    residual = metric.point(np.asarray(predicted, float).reshape(len(codes), -1),
                            np.asarray(encoded_image, float).reshape(len(codes), -1))
    rows = []
    images = graph.images if geometry is None else geometry["images"]
    boxes = graph.boxes if geometry is None else geometry["boxes"]
    for node in graph.minimal:
        selected = labels == node
        if not selected.any():
            rows.append(dict(node=int(node), cells=0, available=False, reason="no cells in the region"))
            continue
        member = index.points_in(codes, selected)
        margin = tolerance(boxes, images, selected, metric, faces)
        largest = float(residual[member].max()) if member.any() else None
        rows.append(dict(node=int(node), cells=int(selected.sum()), available=True,
                         hull=np.r_[graph.boxes[selected, :k].min(axis=0), graph.boxes[selected, k:].max(axis=0)].tolist(),
                         tolerance=margin["value"], witness=margin["witness"], samples=int(member.sum()),
                         max_residual=largest,
                         worst_sample=int(np.flatnonzero(member)[np.argmax(residual[member])]) if member.any() else None,
                         passed=bool(largest is not None and largest < margin["value"]),
                         metric=metric.name, label=f"sampled residual; {LABEL}"))
    return rows, residual
