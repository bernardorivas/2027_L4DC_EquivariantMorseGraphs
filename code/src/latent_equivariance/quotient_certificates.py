"""Attracting block certificates in coordinates of an orbit space.

The cells are boxes of R^k that tile a box. The box is the image of the orbit space
Z/G of a latent space, or of a window of it, under coordinates in which the quotient
metric is known. Given a box that contains the image of each cell under the quotient
map, this module bounds the tolerance of a set of cells from below, evaluates
residuals modulo the group in the same coordinates, and compares the two.

A positive tolerance bound shows directly that the union of the cells is an
attracting block of the quotient map: no image box comes closer than the bound to
a cell outside the set. The relation on cells computed elsewhere is used only to
propose sets of cells.

All numbers are binary64 and are not validated.
"""
import itertools

import numpy as np

LABEL = "binary64, not validated"


class Euclidean:
    """The quotient metric is the Euclidean metric of the coordinates.

    This is the case for (a, |v|) when an orthogonal group acts on v and the
    metric of the latent space is Euclidean.
    """

    name = "Euclidean"

    @staticmethod
    def point(u, w):
        return np.linalg.norm(np.asarray(u, float) - np.asarray(w, float), axis=-1)

    @staticmethod
    def lower(a_lo, a_hi, b_lo, b_hi):
        """The distance between two boxes."""
        gap = np.maximum(0.0, np.maximum(b_lo - a_hi, a_lo - b_hi))
        return np.linalg.norm(gap, axis=-1)

    @staticmethod
    def upper(a_lo, a_hi, b_lo, b_hi):
        """The largest distance between a point of one box and a point of the other."""
        far = np.maximum(np.abs(a_hi - b_lo), np.abs(b_hi - a_lo))
        return np.linalg.norm(far, axis=-1)


class SphereHeight:
    """Heights z in [-1, 1] of the unit sphere modulo the rotations about its axis.

    The quotient of the chordal metric is the distance between the two points of
    one meridian with the given heights. It increases with the angle between them,
    so for two intervals of heights it is extremal at endpoints.
    """

    name = "chordal distance between heights of the unit sphere"

    @staticmethod
    def _heights(*values):
        """The single coordinate of each argument. Finite heights lie in [-1, 1]."""
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

    def upper(self, a_lo, a_hi, b_lo, b_hi):
        a_lo, a_hi, b_lo, b_hi = self._heights(a_lo, a_hi, b_lo, b_hi)
        return np.maximum(self._chord(a_lo, b_hi), self._chord(a_hi, b_lo))


def _mask(cells, count):
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


def _graph(offsets, targets):
    offsets, targets = np.asarray(offsets, np.int64), np.asarray(targets, np.int64)
    count = len(offsets) - 1
    if offsets[0] != 0 or offsets[-1] != len(targets) or np.any(np.diff(offsets) < 0):
        raise ValueError("offsets do not describe the targets")
    if len(targets) and (targets.min() < 0 or targets.max() >= count):
        raise ValueError("a target is not a cell")
    return offsets, targets, count


def reverse(offsets, targets):
    """The relation with every edge reversed, in the same compressed form."""
    offsets, targets, count = _graph(offsets, targets)
    sources = np.repeat(np.arange(count), np.diff(offsets))
    order = np.argsort(targets, kind="stable")
    counts = np.bincount(targets, minlength=count)
    return np.concatenate(([0], np.cumsum(counts))), sources[order]


def forward_closure(offsets, targets, seeds):
    """The cells that can be reached from the seeds, the seeds included."""
    offsets, targets, count = _graph(offsets, targets)
    reached = _mask(seeds, count)
    frontier = np.flatnonzero(reached)
    while len(frontier):
        pieces = [targets[offsets[i]:offsets[i + 1]] for i in frontier]
        new = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, np.int64)
        frontier = new[~reached[new]]
        reached[frontier] = True
    return reached


def is_forward_invariant(offsets, targets, cells):
    offsets, targets, count = _graph(offsets, targets)
    mask = _mask(cells, count)
    sources = np.repeat(np.arange(count), np.diff(offsets))
    return bool(mask[targets[mask[sources]]].all())


def region_of_attraction(offsets, targets, attractor, morse_sets):
    """The cells from which no Morse set outside the attractor can be reached.

    `morse_sets` lists every Morse set of the relation. The attractor is a
    forward invariant set of cells, so each Morse set lies inside it or is
    disjoint from it. The result is forward invariant, contains the attractor,
    and contains the same Morse sets.
    """
    offsets, targets, count = _graph(offsets, targets)
    attractor = _mask(attractor, count)
    if not is_forward_invariant(offsets, targets, attractor):
        raise ValueError("the attractor is not forward invariant")
    outside = np.zeros(count, bool)
    for cells in morse_sets:
        inside = attractor[np.asarray(cells)]
        if inside.all():
            continue
        if inside.any():
            raise ValueError("a Morse set meets the attractor without lying in it")
        outside[np.asarray(cells)] = True
    region = ~forward_closure(*reverse(offsets, targets), outside)
    assert region[attractor].all() and is_forward_invariant(offsets, targets, region)
    return region


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
        gets all of them. The slack, in units of the cell side, also assigns a
        point to a neighbor of its cell when it is that close to the shared face.
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
        mask = np.append(_mask(cells, len(self.lower)), False)
        return mask[self.containing(points, slack)].any(axis=1)

    def boxes_meet(self, low, high, cells, slack=1e-9):
        """Whether each box [low, high] meets the union of the closed cells of the set."""
        mask = _mask(cells, len(self.lower))
        low, high = np.asarray(low, float), np.asarray(high, float)
        first, last = self._range(low, high, slack)
        first = np.maximum(first, 0)
        last = np.minimum(last, np.array(self.shape) - 1)
        out = np.zeros(len(low), bool)
        for row, (a, b) in enumerate(zip(first, last)):
            if np.all(a <= b):
                block = self.ids[tuple(slice(int(i), int(j) + 1) for i, j in zip(a, b))]
                out[row] = mask[block].any()
        return out


def tolerance_lower(lower, upper, cells, image_lower, image_upper, metric, *,
                    boundary_faces=None, chunk=256):
    """A lower bound for the tolerance of the union of a set of cells.

    `image_lower` and `image_upper` have one row for each cell of the grid: a box
    that contains the image of that cell under the quotient map. The tolerance of
    the union B is the distance, in the quotient metric, from the image of B to
    the complement of B in the orbit space. If the bound is positive, B is an
    attracting block of the quotient map.

    `boundary_faces` has shape (k, 2). An entry is True where that face of the
    box tiled by the cells is a face of the orbit space itself, as z = -1 and
    z = 1 for heights of a sphere or |v| = 0 for a norm. Image boxes are cut
    there, because the quotient map takes values in the orbit space. An entry is
    False where the orbit space continues beyond the grid. The outside then
    counts as part of the complement. The default is True everywhere.

    The empty set has infinite tolerance. `chunk` limits the memory of the
    search: chunk times the number of cells outside the set, times k, numbers.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    count, dimension = lower.shape
    mask = _mask(cells, count)
    faces = np.ones((dimension, 2), bool) if boundary_faces is None \
        else np.asarray(boundary_faces, bool)
    if faces.shape != (dimension, 2):
        raise ValueError("boundary_faces needs shape (k, 2)")
    hull = np.prod(upper.max(axis=0) - lower.min(axis=0))
    if not np.isclose(np.prod(upper - lower, axis=1).sum(), hull, rtol=1e-9, atol=0):
        raise ValueError("the cells do not tile a box")
    result = dict(cells=int(mask.sum()), complement_cells=int((~mask).sum()),
                  metric=metric.name, label=LABEL, worst_cell=None, reason=None)
    if not mask.any():
        return dict(result, lower=float("inf"), positive=True, reason="empty set of cells")
    domain_lower, domain_upper = lower.min(axis=0), upper.max(axis=0)
    ids = np.flatnonzero(mask)
    low = np.where(faces[:, 0], np.maximum(np.asarray(image_lower, float)[ids], domain_lower),
                   np.asarray(image_lower, float)[ids])
    high = np.where(faces[:, 1], np.minimum(np.asarray(image_upper, float)[ids], domain_upper),
                    np.asarray(image_upper, float)[ids])
    if np.any(low > high):
        raise ValueError("an image box lies beyond a face of the orbit space")

    best, worst = np.full(len(ids), np.inf), None
    for axis in range(dimension):
        for side in (0, 1):
            if faces[axis, side]:
                continue
            outside_low = np.full(dimension, -np.inf)
            outside_high = np.full(dimension, np.inf)
            if side == 0:
                outside_high[axis] = domain_lower[axis]
            else:
                outside_low[axis] = domain_upper[axis]
            best = np.minimum(best, metric.lower(low, high, outside_low[None], outside_high[None]))
    other_low, other_high = lower[~mask], upper[~mask]
    if len(other_low):
        for start in range(0, len(ids), chunk):
            part = slice(start, start + chunk)
            distance = metric.lower(low[part, None, :], high[part, None, :],
                                    other_low[None], other_high[None])
            best[part] = np.minimum(best[part], distance.min(axis=1))
    bound = float(best.min())
    if np.isfinite(bound):
        worst = int(ids[int(best.argmin())])
    if bound <= 0:
        result["reason"] = "an image box meets a cell outside the set or leaves the grid"
    return dict(result, lower=bound, positive=bool(bound > 0), worst_cell=worst)


def residual_modulo_group(predicted, encoded_image, metric):
    """The residual modulo the group at each sample, in quotient coordinates.

    `predicted` holds the quotient map applied to the coordinates of E(x), and
    `encoded_image` holds the coordinates of E(F(x)).
    """
    return metric.point(predicted, encoded_image)


def residual_upper(predicted_lower, predicted_upper, image_lower, image_upper, metric):
    """Upper bounds for the residual modulo the group on sets of points.

    For each set, one box contains the quotient map applied to the coordinates of
    E(x) and another contains the coordinates of E(F(x)), for every x of the set.
    """
    return metric.upper(np.asarray(predicted_lower, float), np.asarray(predicted_upper, float),
                        np.asarray(image_lower, float), np.asarray(image_upper, float))


def compare(tolerance, residuals, member, kind):
    """Compare a tolerance bound with the residuals of the samples or sets in the block.

    `kind` is "sampled" for residuals at sample points and "enclosed" for upper
    bounds on sets that cover every orbit of the preimage of the block.
    """
    if kind not in ("sampled", "enclosed"):
        raise ValueError("kind is 'sampled' or 'enclosed'")
    residuals, member = np.asarray(residuals, float), np.asarray(member)
    if member.dtype != bool or member.shape != residuals.shape:
        raise ValueError("member is a boolean mask with one entry for each residual")
    residuals = residuals[member]
    result = dict(tolerance_lower=float(tolerance), count=int(len(residuals)), kind=kind,
                  label=f"{kind} residual; {LABEL}")
    if not len(residuals):
        return dict(result, max_residual=None, margin=None, passed=False,
                    reason="nothing in the block")
    largest = float(residuals.max())
    return dict(result, max_residual=largest, rms_residual=float(np.sqrt(np.mean(residuals**2))),
                margin=float(tolerance - largest), passed=bool(largest < tolerance), reason=None)
