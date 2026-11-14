"""Conley indices in X computed from the orbit space X/G.

This module evaluates Theorem thm:sphere-reconstruction of the paper. The orbit space X/G is
identified with a box carrying a uniform grid, and the fixed-point set P/G with a union of facets of
the box. For a Morse set S of an outer approximation F of the quotient map, the index pair is
N/G = |S u F(S)| and L/G = |F(S) minus S|, and N_P/G = N/G n P/G and K = L u N_P. The index map on
H_k(N, L) of the lifted pair is the direct sum of the index map on H_k(N/G, L/G) and deg(delta)
times the index map on H_{k-m}(N/G, K/G), where G/H is homeomorphic to S^m.

Homology is taken over a field: a prime p (5 by default, the field of CMGDB) or "Q".

Cells. A grid has shape[a] cells on axis a, of side 1 in grid units. A cell of its cubical complex
is a tuple c of integers with c[a] = 2 lo_a + d_a, the product of the intervals [lo_a, lo_a + d_a]
with d_a in {0, 1}. The top cell with multi-index i is (2 i_a + 1)_a, which is CMGDB's cell
[lower + i w, lower + (i + 1) w]. A cell is oriented by its nondegenerate axes a_1 < ... < a_k, so
d c = sum_j (-1)^(j-1) ((c + e_{a_j}) - (c - e_{a_j})). Chains are dicts {cell: coefficient}. The
top cells are also numbered by ids in product order with the last axis fastest (NumPy's C order).

Index maps. The index map of the pair (P1, P0) = (S u F(S), F(S) minus S) is inc_*^{-1} phi_*,
where phi is a chain selector of the carrier K(c), the intersection of |F(xi)| over the top cells
xi of P1 that contain c, from (|P1|, |P0|) to (|P1 u F(P0)|, |P0 u F(P0)|), and inc is the
inclusion, checked to induce an isomorphism. On a cell c of P/G, the selector takes values in K(c)
intersected with the facets of P/G that contain c, so it maps N_P/G into P/G. This needs F to map
each of these facets into itself, as the quotient maps of the paper's examples do.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import ceil, floor, prod

import numpy as np

__all__ = ["Grid", "Complex", "relative", "Homology", "BoxMap", "index_pair", "Selector",
           "ConleyIndex", "Reconstruction", "conley_index", "reconstructed_index", "cmgdb_shape",
           "cmgdb_cell", "cmgdb_index", "cmgdb_box", "cmgdb_cells", "cmgdb_morse_set", "cmgdb_map",
           "cdim"]


def cdim(c):
    """The dimension of the cell c, its number of nondegenerate axes."""
    return sum(x & 1 for x in c)


def _add(acc, chain, a=1):
    for c, v in chain.items():
        acc[c] = acc.get(c, 0) + a * v
    return acc


def _clean(chain):
    return {c: v for c, v in chain.items() if v}


# ---------------------------------------------------------------- grids and cubical complexes

class Grid:
    """The uniform grid of a box with shape[a] cells on axis a."""

    def __init__(self, shape):
        self.shape = tuple(int(n) for n in shape)
        self.dim = len(self.shape)
        if not self.dim or min(self.shape) < 1:
            raise ValueError("a grid has at least one axis and one cell on each axis")

    def top(self, i):
        """The top cell with multi-index i."""
        if len(i) != self.dim or any(not 0 <= x < n for x, n in zip(i, self.shape)):
            raise ValueError(f"top cell {i} outside the grid")
        return tuple(2 * x + 1 for x in i)

    def tops(self):
        """The multi-indices of all top cells, in id order."""
        return list(product(*(range(n) for n in self.shape)))

    def multi_index(self, cells):
        """The set of multi-indices of top cells given by ids or by multi-indices."""
        out = set()
        for c in cells:
            if isinstance(c, (int, np.integer)):
                out.add(tuple(int(x) for x in np.unravel_index(int(c), self.shape)))
            else:
                i = tuple(int(x) for x in c)
                self.top(i)
                out.add(i)
        return out

    def ids(self, tops):
        """The sorted ids of top cells given by multi-indices."""
        return sorted(int(np.ravel_multi_index(i, self.shape)) for i in tops)

    def faces(self, c):
        """The faces of codimension one of c with their incidence numbers, [(face, +-1)]."""
        out, k = [], 0
        for a, x in enumerate(c):
            if x & 1:
                s = -1 if k & 1 else 1
                out += [(c[:a] + (x + 1,) + c[a + 1:], s), (c[:a] + (x - 1,) + c[a + 1:], -s)]
                k += 1
        return out

    def boundary(self, chain):
        acc = {}
        for c, v in chain.items():
            for f, s in self.faces(c):
                acc[f] = acc.get(f, 0) + s * v
        return _clean(acc)

    def closure(self, cells):
        """The cells and all their faces."""
        todo = list(cells)
        out = set(todo)
        while todo:
            for f, _ in self.faces(todo.pop()):
                if f not in out:
                    out.add(f)
                    todo.append(f)
        return out

    def cells_of(self, tops):
        """|P|: the cells of the closed top cells with multi-indices in tops."""
        return self.closure([self.top(i) for i in tops])


class Complex:
    """The relative chain complex C(N, L) on the cells of N minus L, with bd[c] = {face: coefficient}.
    The cells of L are `zero`; restrict() sends them to 0."""

    def __init__(self, cells, bd, dim, zero=()):
        self.cells, self.bd, self.dim, self.zero = set(cells), bd, dim, set(zero)
        self.basis = {k: [] for k in range(dim + 1)}
        for c in sorted(self.cells):
            self.basis[cdim(c)].append(c)

    def restrict(self, chain):
        """The class in C(N, L) of a chain on the cells of N."""
        out = {}
        for c, v in chain.items():
            if c in self.cells:
                out[c] = out.get(c, 0) + v
            elif c not in self.zero:
                raise ValueError(f"cell {c} is not in the pair")
        return _clean(out)

    def d(self, chain):
        acc = {}
        for c, v in chain.items():
            _add(acc, self.bd[c], v)
        return _clean(acc)


def relative(grid, N, L=()):
    """C(N, L) for subcomplexes L of N, given as sets of cells closed under faces."""
    N, L = set(N), set(L)
    if not L <= N:
        raise ValueError("L is not contained in N")
    bd = {}
    for c in N:
        fs = grid.faces(c)
        if not {f for f, _ in fs} <= (L if c in L else N):
            raise ValueError(f"not a subcomplex at {c}")
        if c not in L:
            acc = {}
            for f, s in fs:
                if f not in L:
                    acc[f] = acc.get(f, 0) + s
            bd[c] = _clean(acc)
    return Complex(N - L, bd, grid.dim, L)


# ---------------------------------------------------------------- fields and homology

def _field(field):
    """0 for Q, p for the field with p elements."""
    if field == "Q":
        return 0
    p = int(field)
    if p < 2 or any(p % q == 0 for q in range(2, int(p ** 0.5) + 1)):
        raise ValueError(f"the field is a prime or 'Q', not {field!r}")
    return p


def _red(x, p):
    """x as an element of the field; over Q, integers stay Python ints."""
    if p == 0:
        return x if type(x) is int or x.denominator != 1 else x.numerator
    if isinstance(x, Fraction):
        return x.numerator * pow(x.denominator, -1, p) % p
    return int(x) % p


def _inv(x, p):
    if x in (1, -1):
        return x
    return Fraction(1) / x if p == 0 else pow(x, -1, p)


class Homology:
    """Homology of a Complex over a field, with generators and coordinates.

    The complex is reduced by eliminating nonzero incidences, coreductions first, which is a chain
    homotopy equivalence onto a complex with zero differential. gens[k] are cycles whose classes form
    a basis of H_k, betti[k] = dim H_k, and coords(z, k) are the coordinates of the class of a k-cycle
    z in that basis."""

    def __init__(self, cx, field=5):
        p = _field(field)
        self.cx, self.field, self.p = cx, field, p
        col = {c: {} for c in cx.cells}
        for c in cx.cells:
            for f, v in cx.bd[c].items():
                v = _red(v, p)
                if v:
                    col[c][f] = v
        row = {c: {} for c in cx.cells}
        for c, b in col.items():
            for f, v in b.items():
                row[f][c] = v
        steps, crit = [], set()

        def eliminate(a, b):
            ui = _inv(col[b][a], p)
            r = {c: x for c, x in row[a].items() if c != b}
            cb = {f: x for f, x in col[b].items() if f != a}
            for c, x in r.items():
                cc = col[c]
                del cc[a]
                for f, y in cb.items():
                    v = _red(cc.get(f, 0) - x * ui * y, p)
                    if v:
                        cc[f] = row[f][c] = v
                    else:
                        cc.pop(f, None)
                        row[f].pop(c, None)
            for f in col[b]:
                del row[f][b]
            for e in row[b]:
                del col[e][b]
            for f in col[a]:
                del row[f][a]
            for c in (a, b):
                del col[c], row[c]
            steps.append((a, b, cdim(b), ui, r, cb))
            return r

        # coreductions: a cell b with exactly one noncritical face a
        nf = {c: len(col[c]) for c in col}
        pool = {}
        for c in col:
            pool.setdefault(cdim(c), set()).add(c)
        queue = sorted(c for c in col if nf[c] == 1)

        def lose_face(cells):
            for e in cells:
                nf[e] -= 1
                if nf[e] == 1:
                    queue.append(e)

        while True:
            while queue:
                b = queue.pop()
                if b not in col or b in crit or nf[b] != 1:
                    continue
                a = next(f for f in col[b] if f not in crit)
                cob = list(row[b])
                r = eliminate(a, b)
                pool[cdim(a)].discard(a)
                pool[cdim(b)].discard(b)
                lose_face(r)
                lose_face(cob)
            k = next((k for k in sorted(pool) if pool[k]), None)
            if k is None:
                break
            c = min(pool[k])
            pool[k].discard(c)
            crit.add(c)
            lose_face(list(row[c]))
        # eliminate the remaining nonzero incidences between critical cells
        done = False
        while not done:
            done = True
            for b in sorted(col, key=lambda c: (cdim(c), c)):
                faces = list(col.get(b, {}))
                if faces:
                    eliminate(min(faces, key=lambda f: (len(row[f]), f)), b)
                    done = False
        self._steps = steps
        self._rem = {k: sorted(c for c in col if cdim(c) == k) for k in range(cx.dim + 1)}
        self.gens = {k: [self._lift({c: 1}, k) for c in self._rem[k]] for k in range(cx.dim + 1)}
        self.betti = [len(self._rem[k]) for k in range(cx.dim + 1)]
        for k in self.gens:
            for g in self.gens[k]:
                if self._chain(cx.d(g)):
                    raise AssertionError("a generator is not a cycle")

    def _chain(self, z):
        return {c: v for c, v in ((c, _red(x, self.p)) for c, x in z.items()) if v}

    def _project(self, z):
        p, z = self.p, self._chain(z)
        for a, b, _, ui, _, cb in self._steps:
            z.pop(b, None)
            x = z.pop(a, 0)
            if x:
                for f, y in cb.items():
                    v = _red(z.get(f, 0) - x * ui * y, p)
                    if v:
                        z[f] = v
                    else:
                        z.pop(f, None)
        return z

    def _lift(self, x, k):
        p, x = self.p, dict(x)
        for a, b, kb, ui, r, _ in reversed(self._steps):
            if kb == k and r:
                t = _red(sum(y * x[c] for c, y in r.items() if c in x), p)
                if t:
                    x[b] = _red(x.get(b, 0) - ui * t, p)
        return self._chain(x)

    def coords(self, z, k):
        """The coordinates of the class of the k-cycle z, a chain on the cells of the complex."""
        if self._chain(self.cx.d(z)):
            raise ValueError("not a cycle")
        y = self._project(z)
        return [y.get(c, 0) for c in self._rem[k]]


def _induced(Hs, Ht, chain_map, k):
    """The matrix, with rows indexed by the generators of Ht, of the map on H_k of chain_map:
    cell of Hs.cx -> chain on the cells of Ht.cx."""
    cols = []
    for g in Hs.gens[k]:
        img = {}
        for c, v in g.items():
            _add(img, chain_map(c), v)
        cols.append(Ht.coords(_clean(img), k))
    return [list(r) for r in zip(*cols)] if cols else [[] for _ in Ht.gens[k]]


def _solve(M, Y, p):
    """X with M X = Y for a square matrix M over the field, or None if M is singular."""
    n = len(M)
    A = [[_red(x, p) for x in M[i]] + [_red(x, p) for x in Y[i]] for i in range(n)]
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c]), None)
        if piv is None:
            return None
        A[c], A[piv] = A[piv], A[c]
        iv = _inv(A[c][c], p)
        A[c] = [_red(x * iv, p) for x in A[c]]
        for i in range(n):
            if i != c and A[i][c]:
                f = A[i][c]
                A[i] = [_red(x - f * y, p) for x, y in zip(A[i], A[c])]
    return [r[n:] for r in A]


# ---------------------------------------------------------------- outer approximations

class BoxMap:
    """An outer approximation on the top cells of a grid whose images are boxes: F(i) is the set of
    top cells j with lo <= j <= hi for boxes[i] = (lo, hi), and empty for None. Boxes that reach
    outside the grid are clipped to it, as CMGDB clips images."""

    def __init__(self, grid, boxes):
        self.grid, self.boxes, self._img = grid, dict(boxes), {}

    def __call__(self, i):
        if i not in self._img:
            b = self.boxes.get(i)
            rng = [range(max(lo, 0), min(hi, n - 1) + 1) for lo, hi, n in zip(*b, self.grid.shape)] \
                if b else [[]]
            self._img[i] = frozenset(product(*rng))
        return self._img[i]

    def image(self, tops):
        out = set()
        for i in tops:
            out |= self(i)
        return out

    @classmethod
    def from_sets(cls, grid, images):
        """From {top: set of tops}, each image a box of top cells or empty."""
        boxes = {}
        for i, img in images.items():
            img = {tuple(j) for j in img}
            if not img:
                boxes[i] = None
                continue
            lo = tuple(min(j[a] for j in img) for a in range(grid.dim))
            hi = tuple(max(j[a] for j in img) for a in range(grid.dim))
            if len(img) != prod(h - l + 1 for l, h in zip(lo, hi)):
                raise ValueError(f"F({i}) is not a box")
            boxes[i] = (lo, hi)
        return cls(grid, boxes)

    @classmethod
    def from_relation(cls, grid, offsets, targets):
        """From a relation on ids in compressed form: F(i) has the ids targets[offsets[i]:offsets[i+1]],
        with the cells in id order (product order, last axis fastest)."""
        offsets, targets = np.asarray(offsets), np.asarray(targets)
        tops = grid.tops()
        if offsets.shape != (len(tops) + 1,):
            raise ValueError("offsets has one entry for each cell and one more")
        return cls.from_sets(grid, {tops[i]: {tops[int(j)] for j in targets[offsets[i]:offsets[i + 1]]}
                                    for i in range(len(tops))})

    @classmethod
    def from_function(cls, grid, lower, upper, f):
        """From a CMGDB box map f([lo_0..lo_{d-1}, hi_0..hi_{d-1}]) -> image in the same layout: F(i)
        is the set of cells whose closed box meets the closed image. CMGDB also pads each image by
        2^-50 of the width of the domain; this is not reproduced."""
        d = grid.dim
        w = [(upper[a] - lower[a]) / grid.shape[a] for a in range(d)]
        boxes = {}
        for i in grid.tops():
            y = f(cmgdb_box(i, lower, upper, grid.shape))
            boxes[i] = (tuple(ceil((y[a] - lower[a]) / w[a]) - 1 for a in range(d)),
                        tuple(floor((y[d + a] - lower[a]) / w[a]) for a in range(d)))
        return cls(grid, boxes)


def index_pair(F, S):
    """The index pair of S: P1 = S u F(S), P0 = P1 minus S, the target pair (T1, T0) =
    (P1 u F(P0), P0 u F(P0)) of the index map, and isolated, which says that F(P0) n S is empty."""
    S = set(S)
    P1 = S | F.image(S)
    P0 = P1 - S
    FP0 = F.image(P0)
    return dict(S=S, P1=P1, P0=P0, T1=P1 | FP0, T0=P0 | FP0, isolated=not FP0 & S)


# ---------------------------------------------------------------- chain selectors

def _center(lo, hi):
    return lo + 2 * ((hi - lo) // 4)


def _contract(e, K):
    """The contraction D (dD + Dd = 1 - i eps) of the cell e in the box K toward the vertex at its
    center, axis 0 first."""
    if not e:
        return {}
    x, rest, m = e[0], e[1:], _center(*K[0])
    if x & 1:
        return {}
    out = {(y,) + rest: 1 for y in range(m + 1, x, 2)}
    out.update({(y,) + rest: -1 for y in range(x + 1, m, 2)})
    for c, v in _contract(rest, K[1:]).items():
        out[(m,) + c] = out.get((m,) + c, 0) + v
    return out


class Selector:
    """A chain selector of the carrier K(c), the intersection of |F(xi)| over the top cells xi of
    `domain` that contain c, a box. A vertex goes to the vertex at the center of its carrier, and a cell
    of positive dimension to the contraction of phi(d c) in its carrier. faces = {axis: set of
    coordinates} lists facets x_axis = v of the box (v = 0 or 2 shape[axis] in cell coordinates); the
    carrier of a cell on such facets is intersected with them, so the selector maps the cells on each
    facet into that facet."""

    def __init__(self, F, domain, faces=None):
        self.grid, self.F, self.memo = F.grid, F, {}
        self.domain, self.faces = set(domain), dict(faces or {})

    def carrier(self, c):
        """The carrier of c as a box [(lo_a, hi_a)] in cell coordinates."""
        g, K = self.grid, None
        nbrs = [[(x - 1) // 2] if x & 1 else [x // 2 - 1, x // 2] for x in c]
        nbrs = [[j for j in js if 0 <= j < n] for js, n in zip(nbrs, g.shape)]
        for j in product(*nbrs):
            if j not in self.domain:
                continue
            box = self.F.boxes.get(j)
            if not box:
                raise ValueError(f"empty image of the top cell {j}")
            B = []
            for (lo, hi), n in zip(zip(*box), g.shape):
                lo, hi = max(lo, 0), min(hi, n - 1)
                if lo > hi:
                    raise ValueError(f"the image of the top cell {j} misses the grid")
                B.append((2 * lo, 2 * hi + 2))
            K = B if K is None else [(max(x[0], y[0]), min(x[1], y[1])) for x, y in zip(K, B)]
        if K is None:
            raise ValueError(f"no top cell of the domain contains {c}")
        if any(lo > hi for lo, hi in K):
            raise ValueError(f"empty carrier at {c}")
        for a, values in self.faces.items():
            if c[a] in values:
                if not K[a][0] <= c[a] <= K[a][1]:
                    raise ValueError(f"the carrier of {c} misses the facet x_{a} = {c[a] // 2}: F does "
                                     "not map N_P/G into P/G")
                K[a] = (c[a], c[a])
        return K

    def __call__(self, c):
        if c in self.memo:
            return self.memo[c]
        K = self.carrier(c)
        if cdim(c) == 0:
            out = {tuple(_center(lo, hi) for lo, hi in K): 1}
        else:
            z = {}
            for f, s in self.grid.faces(c):
                _add(z, self(f), s)
            z, out = _clean(z), {}
            for e, v in z.items():
                if any(x - (x & 1) < lo or x + (x & 1) > hi for x, (lo, hi) in zip(e, K)):
                    raise ValueError(f"the selector leaves the carrier: {e} is not in {K}")
                for e2, w in _contract(e, K).items():
                    out[e2] = out.get(e2, 0) + v * w
            out = _clean(out)
            if self.grid.boundary(out) != z:
                raise AssertionError(f"the selector is not a chain map at {c}")
        self.memo[c] = out
        return out


# ---------------------------------------------------------------- Conley indices

@dataclass
class ConleyIndex:
    """The homology of an index pair over a field with its index map: betti[k] is the dimension of
    H_k, and maps[k] the matrix of the index map on H_k, a list of betti[k] rows."""
    field: object
    betti: list
    maps: list

    def shift_class(self):
        """CMGDB's labels of the index maps by degree, the notation of its Morse graph annotations.
        Computed by CMGDB.ComputeRelativeShiftClass; the field must be F_5."""
        import CMGDB
        if _field(self.field) != 5:
            raise ValueError("CMGDB computes shift classes over F_5")
        counts = list(self.betti)
        entries = [[(i, j, int(x) % 5) for i, r in enumerate(M) for j, x in enumerate(r) if int(x) % 5]
                   for M in self.maps]
        while len(counts) > 1 and counts[-1] == 0:
            counts.pop()
            entries.pop()
        out = list(CMGDB.ComputeRelativeShiftClass(counts, [[] for _ in counts], entries)["shift_class"])
        return out + ["0"] * (len(self.betti) - len(out))


def _index(grid, phi, source, target, field):
    """The index map inc_*^{-1} phi_* of the pair source = (N, L) with target pair (N', L')."""
    src, tgt = relative(grid, *source), relative(grid, *target)
    whole = tgt.cells | tgt.zero
    for c in src.cells | src.zero:
        if not set(phi(c)) <= (tgt.zero if c in src.zero else whole):
            raise ValueError(f"phi({c}) leaves the target pair")
    Phi, Inc = (lambda c: tgt.restrict(phi(c))), (lambda c: tgt.restrict({c: 1}))
    for M in (Phi, Inc):
        for c in src.cells:
            rhs = {}
            for f, v in src.bd[c].items():
                _add(rhs, M(f), v)
            if tgt.d(M(c)) != _clean(rhs):
                raise ValueError(f"not a chain map at {c}")
    Hs, Ht = Homology(src, field), Homology(tgt, field)
    maps = []
    for k in range(grid.dim + 1):
        M = None
        if len(Hs.gens[k]) == len(Ht.gens[k]):
            M = _solve(_induced(Hs, Ht, Inc, k), _induced(Hs, Ht, Phi, k), Hs.p)
        if M is None:
            raise ValueError(f"the inclusion into the target pair is not an isomorphism on H_{k}")
        maps.append(M)
    return ConleyIndex(field, list(Hs.betti), maps)


def _pair(F, S):
    S = F.grid.multi_index(S)
    if not S:
        raise ValueError("the Morse set is empty")
    pr = index_pair(F, S)
    if not pr["isolated"]:
        raise ValueError("F(P0) meets S: (P1, P0) is not an index pair")
    return pr


def conley_index(F, S, field=5):
    """The Conley index of the Morse set S of F on the index pair (|S u F(S)|, |F(S) minus S|).

    F is a BoxMap and S a set of top cells, given by ids or multi-indices."""
    g, pr = F.grid, _pair(F, S)
    return _index(g, Selector(F, pr["P1"]), (g.cells_of(pr["P1"]), g.cells_of(pr["P0"])),
                  (g.cells_of(pr["T1"]), g.cells_of(pr["T0"])), field)


@dataclass
class Reconstruction:
    """The Conley index in X of the lifted index pair (N, L) of a Morse set.

    N and L are the top cells of N/G and L/G, and N_P the cells of N_P/G = N/G n P/G. quotient is
    the index map on H_*(N/G, L/G), relative the index map on H_*(N/G, K/G) with K = L u N_P, and
    lifted the index map on H_*(N, L): in degree k, the direct sum of quotient in degree k and
    degree times relative in degree k - m."""
    quotient: ConleyIndex
    relative: ConleyIndex
    lifted: ConleyIndex
    m: int
    degree: int
    N: set
    L: set
    N_P: set


def _lift(quotient, relative_, m, degree):
    d, p = len(quotient.betti) - 1, _field(quotient.field)
    betti, maps = [], []
    for k in range(d + m + 1):
        blocks = []
        if k <= d:
            blocks.append(quotient.maps[k])
        if 0 <= k - m <= d:
            blocks.append([[_red(degree * x, p) for x in r] for r in relative_.maps[k - m]])
        n = sum(len(B) for B in blocks)
        M, off = [[0] * n for _ in range(n)], 0
        for B in blocks:
            for i, r in enumerate(B):
                M[off + i][off:off + len(r)] = r
            off += len(B)
        betti.append(n)
        maps.append(M)
    return ConleyIndex(quotient.field, betti, maps)


def reconstructed_index(F, S, fixed, m, degree=1, field=5):
    """The Conley index in X of the lifted pair of the Morse set S, by Theorem thm:sphere-reconstruction.

    F: a BoxMap, the outer approximation of the quotient map on the grid of the box X/G.
    S: the top cells of the Morse set, by ids or multi-indices.
    fixed: the facets of the box whose union is P/G, as pairs (axis, "lower") or (axis, "upper").
    m: G/H is homeomorphic to S^m, m >= 1.
    degree: the degree of the map delta of G/H in part (ii) of the theorem.

    The selector maps N_P/G into P/G, facet by facet; if a carrier misses a facet, F does not map
    N_P/G into P/G and ValueError is raised. The result is the index map of the lifted pair for the
    maps of part (ii) of the theorem with quotient map a continuous selector of F."""
    g, m, degree = F.grid, int(m), int(degree)
    if m < 1:
        raise ValueError("m >= 1")
    faces = {}
    for axis, side in fixed:
        if not 0 <= axis < g.dim or side not in ("lower", "upper"):
            raise ValueError(f"a facet is (axis, 'lower' or 'upper'), not {(axis, side)}")
        faces.setdefault(axis, set()).add(0 if side == "lower" else 2 * g.shape[axis])
    pr = _pair(F, S)
    phi = Selector(F, pr["P1"], faces)
    on_P = lambda cells: {c for c in cells if any(c[a] in v for a, v in faces.items())}  # noqa: E731
    N, L = g.cells_of(pr["P1"]), g.cells_of(pr["P0"])
    T1, T0 = g.cells_of(pr["T1"]), g.cells_of(pr["T0"])
    quotient = _index(g, phi, (N, L), (T1, T0), field)
    relative_ = _index(g, phi, (N, L | on_P(N)), (T1, T0 | on_P(T1)), field)
    return Reconstruction(quotient, relative_, _lift(quotient, relative_, m, degree), m, degree,
                          pr["P1"], pr["P0"], on_P(N))


# ---------------------------------------------------------------- CMGDB on a uniform grid

def cmgdb_shape(depth, dim):
    """The cells per axis after depth subdivisions; the node at depth t is bisected along t % dim."""
    return tuple(2 ** ((depth + dim - 1 - a) // dim) for a in range(dim))


def cmgdb_cell(m, depth, dim):
    """The multi-index of CMGDB's cell m on the uniform grid of the given depth (Morton order)."""
    i = [0] * dim
    for t in range(depth):
        i[t % dim] = 2 * i[t % dim] + ((m >> (depth - 1 - t)) & 1)
    return tuple(i)


def cmgdb_index(i, depth):
    """CMGDB's number of the cell with multi-index i, the inverse of cmgdb_cell."""
    dim = len(i)
    k = [(depth + dim - 1 - a) // dim for a in range(dim)]
    m = 0
    for t in range(depth):
        m = 2 * m + ((i[t % dim] >> (k[t % dim] - 1 - t // dim)) & 1)
    return m


try:
    from math import fma as _fma                 # Python 3.13
except ImportError:
    def _fma(a, b, c):
        """a b + c with one rounding."""
        return float(Fraction(a) * Fraction(b) + Fraction(c))


def cmgdb_box(i, lower, upper, shape):
    """CMGDB's bounds [lo_0..lo_{d-1}, hi_0..hi_{d-1}] of the cell i, bit for bit as TreeGrid::geometry
    computes them in CMGDB 1.5.3+fork.7: lo = t U + (1 - t) L with t = i/n and hi = s L + (1 - s) U
    with s = 1 - (i + 1)/n, where clang on arm64 contracts a b + c d to fma(a, b, c d)."""
    lo, hi = [], []
    for a, n in enumerate(shape):
        t, s = i[a] / n, 1 - (i[a] + 1) / n
        lo.append(_fma(t, upper[a], (1 - t) * lower[a]))
        hi.append(_fma(s, lower[a], (1 - s) * upper[a]))
    return lo + hi


def cmgdb_cells(boxes, lower, upper, depth):
    """The multi-indices of CMGDB boxes on the uniform grid of the given depth. Each box agrees with
    the bounds of one cell (cmgdb_box) to 1e-9 of a cell width."""
    d = len(lower)
    shape = cmgdb_shape(depth, d)
    w = [(upper[a] - lower[a]) / shape[a] for a in range(d)]
    out = []
    for box in boxes:
        i = tuple(round((box[a] - lower[a]) / w[a]) for a in range(d))
        ref = cmgdb_box(i, lower, upper, shape)
        if any(not 0 <= i[a] < shape[a] for a in range(d)) or \
                any(abs(x - y) > 1e-9 * w[a % d] for a, (x, y) in enumerate(zip(box, ref))):
            raise ValueError(f"box {box} is not a cell of the uniform grid of depth {depth}")
        out.append(i)
    return out


def cmgdb_morse_set(morse_graph, v, lower, upper, depth):
    """The top cells of the Morse set v of a CMGDB Morse graph computed with subdiv_min = subdiv_max =
    depth, from its boxes, checked against the Morton order of its cell numbers."""
    a = set(cmgdb_cells(morse_graph.morse_set_boxes(v), lower, upper, depth))
    if a != {cmgdb_cell(m, depth, len(lower)) for m in morse_graph.morse_set(v)}:
        raise ValueError("the boxes and cell numbers of the Morse set disagree")
    return a


def cmgdb_map(map_graph, grid, depth):
    """The BoxMap of a CMGDB map graph on the uniform grid of the given depth."""
    d = grid.dim
    if cmgdb_shape(depth, d) != grid.shape:
        raise ValueError(f"depth {depth} does not give the grid {grid.shape}")
    return BoxMap.from_sets(grid, {cmgdb_cell(m, depth, d): {cmgdb_cell(x, depth, d)
                                                             for x in map_graph.adjacencies(m)}
                                   for m in range(prod(grid.shape))})
