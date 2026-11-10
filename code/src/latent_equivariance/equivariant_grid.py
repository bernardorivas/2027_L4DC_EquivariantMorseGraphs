"""Equivariant uniform grids and float64 box images on cell representatives.

Box arithmetic is a float64 enclosure, not validated. Integer cell relations are exact once
these boxes are fixed. Supported actions are diagonal signs on boxes and the Klein involution.
"""
from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
from scipy.sparse import csr_matrix

from .homology import BoxMap, Grid, Group


@dataclass
class CellOrbits:
    """Active cell IDs, orbit IDs, and elements carrying representatives to active cells."""

    cell_ids: np.ndarray
    representatives: np.ndarray
    orbit_index: np.ndarray
    transporter: np.ndarray
    stabilizers: tuple


class EquivariantGrid:
    """A physical uniform grid together with its action on oriented grid cells."""

    def __init__(self, shape, lower, upper, group_generators, periodic=False, kind="signs"):
        self.grid = Grid(shape, periodic=periodic)
        self.shape = np.asarray(self.grid.shape, dtype=np.int64)
        self.dim = self.grid.dim
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        if self.lower.shape != (self.dim,) or self.upper.shape != (self.dim,):
            raise ValueError("bounds must have one entry per axis")
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("bounds must be finite and strictly ordered")
        self.width = (self.upper - self.lower) / self.shape
        self.group = Group(self.grid, group_generators)
        self.kind = kind
        self.size = prod(self.grid.shape)
        self._gids = {g: i for i, g in enumerate(self.group.elements)}
        self.multiplication = np.asarray([
            [self._gids[self.group.compose(g, h)] for h in self.group.elements]
            for g in self.group.elements], dtype=np.int64)
        self.identity = self._gids[self.group.identity]
        self.inverse = np.asarray([int(np.flatnonzero(row == self.identity)[0])
                                   for row in self.multiplication], dtype=np.int64)
        self.check_grid_condition()

    @classmethod
    def diagonal_signs(cls, shape, lower, upper, sign_generators):
        """Diagonal sign generators on a symmetric box, with zero a vertex when flipped."""
        shape = tuple(int(n) for n in shape)
        d = len(shape)
        signs = [tuple(int(s) for s in row) for row in sign_generators]
        if any(len(row) != d or any(s not in (-1, 1) for s in row) for row in signs):
            raise ValueError("each sign generator must contain d entries in {-1,1}")
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if lo.shape != (d,) or hi.shape != (d,):
            raise ValueError("bounds must have one entry per axis")
        for a in range(d):
            if any(s[a] == -1 for s in signs):
                if lo[a] != -hi[a] or shape[a] % 2:
                    raise ValueError("a reflected axis must be symmetric with zero a vertex")
        gens = [(tuple(range(d)), s, tuple(n if t == -1 else 0
                                         for n, t in zip(shape, s))) for s in signs]
        return cls(shape, lo, hi, gens, kind="signs")

    @classmethod
    def klein(cls, shape):
        """The unit torus with (x,y) mapped to (x+1/2,-y)."""
        if len(shape) != 2 or int(shape[0]) % 2:
            raise ValueError("the Klein grid must have two axes and an even first size")
        gens = [((0, 1), (1, -1), (int(shape[0]) // 2, 0))]
        return cls(shape, (0., 0.), (1., 1.), gens, periodic=True, kind="klein")

    def check_grid_condition(self):
        """Check the admitted actions structurally, without enumerating every face."""
        identity_perm = tuple(range(self.dim))
        if any(g[0] != identity_perm for g in self.group.elements):
            raise ValueError("permuted axes are outside the supported grid class")
        if self.kind == "signs":
            if any(self.grid.periodic):
                raise ValueError("diagonal signs currently require a nonperiodic box")
            for g in self.group.elements:
                for a, s in enumerate(g[1]):
                    if s == -1 and (self.shape[a] % 2 or self.lower[a] != -self.upper[a]):
                        raise ValueError("grid condition fails on a reflected axis")
        elif self.kind == "klein":
            expected = ((0, 1), (1, -1), (int(self.shape[0]) // 2, 0))
            if (self.dim != 2 or not all(self.grid.periodic) or self.shape[0] % 2
                    or len(self.group) != 2 or expected not in self.group.elements
                    or not np.array_equal(self.lower, [0., 0.])
                    or not np.array_equal(self.upper, [1., 1.])):
                raise ValueError("invalid Klein action")
        else:
            raise ValueError("unsupported action")
        return True

    @property
    def closed_cell_injective(self):
        """Injectivity on each realized closed-cell subset, not regularity of its attaching map."""
        return self.kind == "signs" or self.shape[0] >= 4

    def indices(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.size):
            raise ValueError("cell ID outside grid")
        return np.stack(np.unravel_index(ids, tuple(self.shape)), axis=-1)

    def ids(self, indices):
        a = np.asarray(indices, dtype=np.int64)
        if a.shape[-1] != self.dim:
            raise ValueError("cell indices have wrong dimension")
        return np.ravel_multi_index(a.T, tuple(self.shape))

    def act_indices(self, g, indices):
        p, s, b = self.group.elements[int(g)]
        x = np.asarray(indices, dtype=np.int64)[..., p]
        s, b = np.asarray(s), np.asarray(b)
        y = s * x + b - (s == -1)
        return np.where(self.grid.periodic, y % self.shape, y)

    def act_ids(self, g, ids):
        return self.ids(self.act_indices(g, self.indices(ids)))

    def act_ranges(self, g, lo, hi):
        p, s, b = self.group.elements[int(g)]
        lo, hi = np.asarray(lo)[..., p], np.asarray(hi)[..., p]
        s, b = np.asarray(s), np.asarray(b)
        return (np.where(s == 1, lo + b, -hi - 1 + b),
                np.where(s == 1, hi + b, -lo - 1 + b))

    def act_boxes(self, g, lo, hi):
        p, s, b = self.group.elements[int(g)]
        s, b = np.asarray(s), np.asarray(b)
        shift = self.lower - s * self.lower[list(p)] + self.width * b
        lo, hi = np.asarray(lo)[..., p], np.asarray(hi)[..., p]
        return np.where(s == 1, lo, -hi) + shift, np.where(s == 1, hi, -lo) + shift

    def cell_bounds(self, ids):
        indices = self.indices(ids)
        lo = self.lower + indices * self.width
        return lo, self.lower + (indices + 1) * self.width

    def orbits(self, active_ids=None):
        ids = (np.arange(self.size, dtype=np.int64) if active_ids is None
               else np.unique(np.asarray(active_ids, dtype=np.int64)))
        if not len(ids):
            raise ValueError("active grid is empty")
        images = np.stack([self.act_ids(g, ids) for g in range(len(self.group))])
        positions = np.searchsorted(ids, images)
        if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], images):
            raise ValueError("active cells are not invariant under the group")
        canonical = images.min(axis=0)
        reps, orbit_index = np.unique(canonical, return_inverse=True)
        transporter = np.full(len(ids), -1, dtype=np.int64)
        stabilizers = []
        rep_images = np.stack([self.act_ids(g, reps) for g in range(len(self.group))])
        for g in range(len(self.group)):
            matches = rep_images[g, orbit_index] == ids
            transporter[(transporter < 0) & matches] = g
        for j, rep in enumerate(reps):
            stabilizers.append(tuple(np.flatnonzero(rep_images[:, j] == rep).tolist()))
        return CellOrbits(ids, reps, orbit_index, transporter, tuple(stabilizers))


class OuterApproximation:
    """Box images and integer image ranges indexed by active top-cell IDs.

An outside vertex retains exits from the physical box or from the active cell set. It has no
outgoing edge. Its absence from recurrent SCCs never removes the escape diagnostics.
"""

    label = "float64 enclosure, not validated"

    def __init__(self, grid, orbits, image_lower, image_upper, index_lower, index_upper,
                 padding, evaluated):
        self.equivariant_grid, self.orbits = grid, orbits
        self.cell_ids = orbits.cell_ids
        self.image_lower, self.image_upper = image_lower, image_upper
        self.index_lower, self.index_upper = index_lower, index_upper
        self.padding, self.evaluated = float(padding), int(evaluated)
        self.outside = len(self.cell_ids)
        self._adjacency = None

    def positions(self, ids):
        a = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.cell_ids, a)
        if np.any(pos >= self.outside) or not np.array_equal(self.cell_ids[pos], a):
            raise ValueError("cell is not in the active set")
        return pos

    def image_ids(self, row, max_cells=20_000_000):
        """Active image IDs and an escape flag, using the stored closed-cell relation."""
        g = self.equivariant_grid
        axes, outside = [], False
        for a, (lo, hi, n, periodic) in enumerate(zip(
                self.index_lower[row], self.index_upper[row], g.shape, g.grid.periodic)):
            if periodic:
                length = min(int(hi - lo + 1), int(n))
                axes.append((np.arange(length, dtype=np.int64) + lo) % n)
            else:
                outside |= lo < 0 or hi >= n
                axes.append(np.arange(max(0, lo), min(n - 1, hi) + 1, dtype=np.int64))
        if any(len(axis) == 0 for axis in axes):
            return np.empty(0, dtype=np.int64), True
        if prod(len(axis) for axis in axes) > max_cells:
            raise MemoryError("one image exceeds the configured cell budget")
        multi = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, g.dim)
        ids = g.ids(multi)
        positions = np.searchsorted(self.cell_ids, ids)
        valid = positions < self.outside
        valid[valid] &= self.cell_ids[positions[valid]] == ids[valid]
        outside |= not valid.all()
        return ids[valid], bool(outside)

    def adjacency(self, max_edges=20_000_000):
        """CSR relation including the outside node, with an allocation limit."""
        if self._adjacency is not None:
            return self._adjacency
        rows, cols, total = [], [], 0
        for row in range(self.outside):
            ids, escaped = self.image_ids(row, max_cells=max_edges - total)
            positions = self.positions(ids)
            if escaped:
                positions = np.append(positions, self.outside)
            total += len(positions)
            if total > max_edges:
                raise MemoryError("outer relation exceeds the configured edge budget")
            rows.append(np.full(len(positions), row, dtype=np.int64))
            cols.append(positions)
        self._adjacency = csr_matrix((np.ones(total, dtype=np.int8),
            (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.outside + 1, self.outside + 1))
        return self._adjacency

    def local_box_map(self, ids=None, require_no_escape=True):
        """Return the existing homology BoxMap on explicitly requested active cells only."""
        g = self.equivariant_grid
        rows = np.arange(self.outside) if ids is None else self.positions(ids)
        if require_no_escape:
            if any(self.image_ids(int(row))[1] for row in rows):
                raise ValueError("requested local box map has an outside image")
        boxes = {tuple(i): (tuple(self.index_lower[row]), tuple(self.index_upper[row]))
                 for row, i in zip(rows, g.indices(self.cell_ids[rows]))}
        return BoxMap(g.grid, boxes)


def build_outer_approximation(grid, enclose, padding=1e-12, active_ids=None, batch_size=8192):
    """Evaluate enclose(lo, hi) on one cell per orbit and extend integer images by the action.

The callback accepts two float64 arrays of shape (batch, dimension) and returns image bounds
with that shape. It must enclose the intended equivariant map. This function adds padding once.
"""
    if not np.isfinite(padding) or padding <= 0 or batch_size < 1:
        raise ValueError("positive finite padding and batch size are required")
    orbits = grid.orbits(active_ids)
    n, d = len(orbits.cell_ids), grid.dim
    rep_lo, rep_hi = np.empty((len(orbits.representatives), d)), np.empty((len(orbits.representatives), d))
    for start in range(0, len(orbits.representatives), batch_size):
        ids = orbits.representatives[start:start + batch_size]
        lo, hi = grid.cell_bounds(ids)
        a, b = (np.asarray(v, dtype=np.float64) for v in enclose(lo, hi))
        if a.shape != lo.shape or b.shape != lo.shape or not np.isfinite([a, b]).all():
            raise ValueError("enclosure returned invalid bounds")
        if np.any(a > b):
            raise ValueError("enclosure has an empty interval")
        rep_lo[start:start + len(ids)], rep_hi[start:start + len(ids)] = a - padding, b + padding
    scaled_lo = (rep_lo - grid.lower) / grid.width
    scaled_hi = (rep_hi - grid.lower) / grid.width
    limit = np.iinfo(np.int64).max / 4
    if np.any(np.abs([scaled_lo, scaled_hi]) >= limit):
        raise ValueError("enclosure exceeds representable integer range")
    rilo = np.ceil(scaled_lo).astype(np.int64) - 1
    rihi = np.floor(scaled_hi).astype(np.int64)
    elo, ehi = np.empty((n, d)), np.empty((n, d))
    ilo, ihi = np.empty((n, d), dtype=np.int64), np.empty((n, d), dtype=np.int64)
    for j, stab in enumerate(orbits.stabilizers):
        for h in stab:
            a, b = grid.act_ranges(h, rilo[j], rihi[j])
            # Universal-cover periodic boxes may differ by one common period.
            delta = a - rilo[j]
            equiv = b - rihi[j] == delta
            for axis, periodic in enumerate(grid.grid.periodic):
                equiv[axis] &= delta[axis] % grid.shape[axis] == 0 if periodic else delta[axis] == 0
            if not equiv.all():
                raise ValueError("representative value is not invariant under its stabilizer")
    for h in range(len(grid.group)):
        target = np.flatnonzero(orbits.transporter == h)
        source = orbits.orbit_index[target]
        elo[target], ehi[target] = grid.act_boxes(h, rep_lo[source], rep_hi[source])
        ilo[target], ihi[target] = grid.act_ranges(h, rilo[source], rihi[source])
    return OuterApproximation(grid, orbits, elo, ehi, ilo, ihi, padding, len(rep_lo))
