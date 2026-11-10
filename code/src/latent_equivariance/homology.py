"""Cubical homology for what CMGDB does not compute.

CMGDB computes Morse graphs and Conley indices over F_5. This module computes, on a uniform grid,
homology over Z (with torsion), F_p and Q, index maps through a carrier chain selector, signed
group actions on chains, orbit and twisted complexes, isotypic components, and the reconstruction
of an index from an orbit space. It reads CMGDB's cells on a uniform grid (subdiv_min =
subdiv_max) as integer cells, so that both compute on the same index pairs.

Cells. A grid has shape[a] cells on axis a. A cell is a tuple c of integers with
c[a] = 2 lo_a + d_a, the product of the intervals [lo_a, lo_a + d_a] in grid units; axes with
d_a = 0 are degenerate. The top cell with multi-index i, CMGDB's box [lower + i w, lower + (i+1) w],
is (2 i_a + 1)_a. On a periodic axis c[a] is reduced modulo 2 shape[a]. A cell is oriented by
e_{a_1} ^ ... ^ e_{a_k} over its nondegenerate axes a_1 < ... < a_k (CMGDB's orientation), so
d c = sum_j (-1)^(j-1) ((c + e_{a_j}) - (c - e_{a_j})). Chains are dicts {cell: coefficient}.

Conventions. The index pair of a Morse set S of
a combinatorial map F is P1 = S u F(S), P0 = P1 minus S, checked F(P0) n S = empty. Its index map is
inc_*^{-1} phi_* with phi: (|P1|, |P0|) -> (|P1 u F(P0)|, |P0 u F(P0)|) a selector of the carrier
K(c) = intersection of |F(xi)| over the top cells xi containing c, and inc the inclusion, checked
to induce an isomorphism. A group acts on oriented cells with signs: the sign of the permutation of
the nondegenerate axes times the signs on them.
"""
from fractions import Fraction
from itertools import product
from math import ceil, floor, prod

__all__ = ["Grid", "Complex", "relative", "pair", "Homology", "induced", "index_map", "BoxMap",
           "Selector", "index_pair", "conley_index", "Group", "OrbitComplex", "commutes", "action",
           "isotypic_betti", "check_z2_reconstruction", "hyperplane", "reconstruct",
           "reconstructed_index", "cmgdb_shape", "cmgdb_cell", "cmgdb_index", "cmgdb_box",
           "cmgdb_cells", "cmgdb_morse_set", "cmgdb_map", "cdim"]


def cdim(c):
    return sum(x & 1 for x in c)


def _add(acc, chain, a=1):
    for c, v in chain.items():
        acc[c] = acc.get(c, 0) + a * v
    return acc


def _clean(chain):
    return {c: v for c, v in chain.items() if v}


# ---------------------------------------------------------------- grid, cells, complexes

class Grid:
    """Uniform cubical grid with shape[a] cells on axis a (cells of side 1 in grid units)."""

    def __init__(self, shape, periodic=False):
        self.shape = tuple(int(n) for n in shape)
        self.dim = len(self.shape)
        per = [periodic] * self.dim if isinstance(periodic, bool) else list(periodic)
        self.periodic = tuple(bool(p) for p in per)
        if len(self.periodic) != self.dim or min(self.shape) < 1:
            raise ValueError("bad grid")

    def norm(self, c):
        return tuple(x % (2 * n) if p else x for x, n, p in zip(c, self.shape, self.periodic))

    def norm_index(self, i):
        return tuple(x % n if p else x for x, n, p in zip(i, self.shape, self.periodic))

    def top(self, i):
        if any(not p and not 0 <= x < n for x, n, p in zip(i, self.shape, self.periodic)):
            raise ValueError(f"top cell {i} outside the grid")
        return self.norm(tuple(2 * x + 1 for x in i))

    def tops(self):
        return list(product(*(range(n) for n in self.shape)))

    def faces(self, c):
        """[(face, incidence)]."""
        out, k = [], 0
        for a, x in enumerate(c):
            if x & 1:
                s = -1 if k & 1 else 1
                out += [(self.norm(c[:a] + (x + 1,) + c[a + 1:]), s),
                        (self.norm(c[:a] + (x - 1,) + c[a + 1:]), -s)]
                k += 1
        return out

    def boundary(self, chain):
        acc = {}
        for c, v in chain.items():
            for f, s in self.faces(c):
                acc[f] = acc.get(f, 0) + s * v
        return _clean(acc)

    def closure(self, cells):
        todo = list(cells)
        S = set(todo)
        while todo:
            for f, _ in self.faces(todo.pop()):
                if f not in S:
                    S.add(f)
                    todo.append(f)
        return S

    def cells_of(self, tops):
        """|P|: all cells of the closed top cells with multi-indices in tops."""
        return self.closure([self.top(i) for i in tops])


class Complex:
    """A based chain complex over Z: cells (grid cells, graded by cdim) and bd[c] = {face: coef}.
    Cells of `zero` (the subcomplex L of a pair) are sent to 0 by restrict()."""

    def __init__(self, cells, bd, dim, zero=()):
        self.cells, self.bd, self.dim, self.zero = set(cells), bd, dim, set(zero)
        self.basis = {k: [] for k in range(dim + 1)}
        for c in sorted(self.cells):
            self.basis[cdim(c)].append(c)

    def restrict(self, chain):
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

    def check(self):
        bad = [c for c in self.cells if self.d(self.bd[c])]
        if bad:
            raise ValueError(f"d d != 0 at {bad[0]}")


def relative(grid, N, L=()):
    """C(N, L) for subcomplexes L of N (sets of cells closed under faces)."""
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


def pair(grid, P1, P0=()):
    """C(|P1|, |P0|) for sets of top cells (multi-indices) P0 in P1."""
    return relative(grid, grid.cells_of(P1), grid.cells_of(P0))


# ---------------------------------------------------------------- rings and Smith normal form

def _field(ring):
    """None for Z, 0 for Q, p for F_p."""
    if ring == "Z":
        return None
    if ring == "Q":
        return 0
    p = int(ring)
    if p < 2 or any(p % q == 0 for q in range(2, int(p ** 0.5) + 1)):
        raise ValueError(f"ring must be 'Z', 'Q' or a prime, not {ring!r}")
    return p


def _red(x, p):
    """x in the ring; over Q integers stay Python ints (three times faster than Fractions)."""
    if p is None:
        return int(x)
    if p == 0:
        return x if type(x) is int or x.denominator != 1 else x.numerator
    if isinstance(x, Fraction):
        return x.numerator * pow(x.denominator, -1, p) % p
    return int(x) % p


def _inv(x, p):
    if p is None or x in (1, -1):
        return x
    return Fraction(1) / x if p == 0 else pow(x, -1, p)


def _unit(x, p):
    return x in (1, -1) if p is None else x != 0


def _eye(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def _mul(A, B, ncols):
    return [[sum(r[t] * B[t][j] for t in range(len(r))) for j in range(ncols)] for r in A]


def _snf(A0):
    """U A V = diag(d) with d_i > 0, d_i | d_{i+1}, U and V unimodular: (d, U, Ui, V, Vi)."""
    A = [list(r) for r in A0]
    m, n = len(A), len(A[0])
    U, Ui, V, Vi = _eye(m), _eye(m), _eye(n), _eye(n)

    def swap_rows(i, j):
        A[i], A[j], U[i], U[j] = A[j], A[i], U[j], U[i]
        for r in Ui:
            r[i], r[j] = r[j], r[i]

    def swap_cols(i, j):
        for r in A + V:
            r[i], r[j] = r[j], r[i]
        Vi[i], Vi[j] = Vi[j], Vi[i]

    def add_row(i, t, q):  # row_i += q row_t
        A[i] = [x + q * y for x, y in zip(A[i], A[t])]
        U[i] = [x + q * y for x, y in zip(U[i], U[t])]
        for r in Ui:
            r[t] -= q * r[i]

    def add_col(j, t, q):  # col_j += q col_t
        for r in A + V:
            r[j] += q * r[t]
        Vi[t] = [x - q * y for x, y in zip(Vi[t], Vi[j])]

    t = 0
    while t < min(m, n):
        nz = [(abs(A[i][j]), i, j) for i in range(t, m) for j in range(t, n) if A[i][j]]
        if not nz:
            break
        _, i, j = min(nz)
        swap_rows(t, i)
        swap_cols(t, j)
        while True:
            for i in range(t + 1, m):
                if A[i][t] // A[t][t]:
                    add_row(i, t, -(A[i][t] // A[t][t]))
            for j in range(t + 1, n):
                if A[t][j] // A[t][t]:
                    add_col(j, t, -(A[t][j] // A[t][t]))
            rest = [(abs(A[i][t]), i, -1) for i in range(t + 1, m) if A[i][t]]
            rest += [(abs(A[t][j]), -1, j) for j in range(t + 1, n) if A[t][j]]
            if rest:
                _, i, j = min(rest)
                swap_rows(t, i) if i >= 0 else swap_cols(t, j)
                continue
            bad = next((i for i in range(t + 1, m) for j in range(t + 1, n) if A[i][j] % A[t][t]), None)
            if bad is None:
                break
            add_row(t, bad, 1)
        if A[t][t] < 0:
            A[t], U[t] = [-x for x in A[t]], [-x for x in U[t]]
            for r in Ui:
                r[t] = -r[t]
        t += 1
    d = [A[i][i] for i in range(t)]
    D = _mul(_mul(U, A0, n), V, n)
    assert all(D[i][j] == (d[i] if i == j and i < t else 0) for i in range(m) for j in range(n))
    return d, U, Ui, V, Vi


def _zhom(dk, dk1, n, m1):
    """Homology over Z of Z^m1 -dk1-> Z^n -dk-> Z^?: generators G (n x h), coordinates C (h x n) and
    the orders (0: free) of the generators."""
    if dk and any(any(r) for r in dk):
        d, _, _, V, Vi = _snf(dk)
    else:
        d, V, Vi = [], _eye(n), _eye(n)
    r = len(d)
    z = n - r
    K, Wt = [row[r:] for row in V], Vi[r:]
    B = _mul(Wt, dk1, m1) if m1 else [[] for _ in range(z)]
    if m1 and any(any(row) for row in B):
        d2, U2, U2i, _, _ = _snf(B)
    else:
        d2, U2, U2i = [], _eye(z), _eye(z)
    s = len(d2)
    G, C = _mul(K, U2i, z), _mul(U2, Wt, n)
    keep = [i for i in range(z) if i >= s or d2[i] > 1]
    return [[G[a][i] for i in keep] for a in range(n)], [C[i] for i in keep], \
        [d2[i] if i < s else 0 for i in keep]


# ---------------------------------------------------------------- homology

class Homology:
    """Homology of a Complex over ring = "Z", "Q" or a prime p, with generators and coordinates.

    The complex is reduced by Gaussian elimination of unit incidences (coreductions first), a chain
    homotopy equivalence f: C -> C', g: C' -> C over the ring; over Z the reduced complex is then
    solved by Smith normal forms. gens[k] are cycles of C (g of generators of C'), orders[k] their
    orders (0 for free), betti[k] the free rank or the dimension, torsion[k] the orders > 1, and
    coords(z, k) the coordinates of the class of a cycle z (through f)."""

    def __init__(self, cx, ring="Z"):
        p = _field(ring)
        self.cx, self.ring, self.p = cx, ring, p
        col = {c: {} for c in cx.cells}
        for c in cx.cells:
            for f, v in cx.bd[c].items():
                if _red(v, p):
                    col[c][f] = _red(v, p)
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

        # coreductions: b with exactly one noncritical face a (fill-in only in critical rows)
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
                if not _unit(col[b][a], p):
                    continue
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
        # the remaining (critical) complex: eliminate the remaining unit entries
        done = False
        while not done:
            done = True
            for b in sorted(col, key=lambda c: (cdim(c), c)):
                us = [f for f, x in col.get(b, {}).items() if _unit(x, p)]
                if us:
                    eliminate(min(us, key=lambda f: (len(row[f]), f)), b)
                    done = False
        self._steps = steps
        rem = {k: sorted(c for c in col if cdim(c) == k) for k in range(-1, cx.dim + 2)}
        self._rem, self._C, self.gens, self.orders = rem, {}, {}, {}
        for k in range(cx.dim + 1):
            R = rem[k]
            if p is None:
                dk = [[col[c].get(f, 0) for c in R] for f in rem[k - 1]]
                dk1 = [[col[c].get(f, 0) for c in rem[k + 1]] for f in R]
                G, C, orders = _zhom(dk, dk1, len(R), len(rem[k + 1]))
                red_gens = [_clean({R[a]: G[a][j] for a in range(len(R))}) for j in range(len(orders))]
                self._C[k] = C
            else:
                assert all(not col[c] for c in R), "nonzero differential left over a field"
                red_gens, orders = [{c: 1} for c in R], [0] * len(R)
            self.orders[k] = orders
            self.gens[k] = [self._lift(g, k) for g in red_gens]
            for g in self.gens[k]:
                assert not self._chain(cx.d(g)), "generator is not a cycle"
        self.betti = [sum(1 for o in self.orders[k] if o == 0) for k in range(cx.dim + 1)]
        self.torsion = [sorted(o for o in self.orders[k] if o) for k in range(cx.dim + 1)]

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
        """Coordinates of the class of the k-cycle z (a chain on the basis of the complex)."""
        if self._chain(self.cx.d(z)):
            raise ValueError("not a cycle")
        y = self._project(z)
        v = [y.get(c, 0) for c in self._rem[k]]
        if self.p is not None:
            return v
        return [(x % o if o else x) for x, o in
                zip((sum(a * b for a, b in zip(r, v)) for r in self._C[k]), self.orders[k])]

    def group(self, k):
        if self.p is not None:
            return str(self.betti[k])
        f = self.betti[k]
        parts = ([] if f == 0 else ["Z" if f == 1 else f"Z^{f}"]) + [f"Z/{t}" for t in self.torsion[k]]
        return " + ".join(parts) if parts else "0"

    def groups(self):
        return [self.group(k) for k in range(self.cx.dim + 1)]


def induced(Hs, Ht, chain_map, k):
    """Matrix (rows: generators of Ht) of the map on H_k induced by chain_map: cell of Hs.cx ->
    chain on the basis of Ht.cx."""
    cols = []
    for g in Hs.gens[k]:
        img = {}
        for c, v in g.items():
            _add(img, chain_map(c), v)
        cols.append(Ht.coords(_clean(img), k))
    return [list(r) for r in zip(*cols)] if cols else [[] for _ in Ht.gens[k]]


def _fsolve(M, Y, p):
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


def _zsolve(M, Y, so, to):
    """X with M X = Y modulo the target orders, reduced modulo the source orders, if M is an
    isomorphism (sum Z/so) -> (sum Z/to); else None."""
    n, m = len(so), len(to)
    if sorted(so) != sorted(to):
        return None
    if m == 0:
        return []
    d, U, _, V, _ = _snf([list(M[i]) + [to[i] * (j == i) for j in range(m)] for i in range(m)])

    def solve(y):
        w = [0] * (n + m)
        for i, x in enumerate(sum(U[i][j] * y[j] for j in range(m)) for i in range(m)):
            if (i < len(d) and x % d[i]) or (i >= len(d) and x):
                return None
            if i < len(d):
                w[i] = x // d[i]
        return [(v % o if o else v) for v, o in
                zip((sum(V[i][j] * w[j] for j in range(n + m)) for i in range(n)), so)]

    if any(solve([int(i == j) for i in range(m)]) is None for j in range(m)):
        return None
    return [list(r) for r in zip(*(solve([Y[i][j] for i in range(m)]) for j in range(n)))]


def index_map(Hs, Ht, phi, inc, k):
    """inc_*^{-1} phi_* on H_k(Hs.cx), or None if inc_* is not an isomorphism."""
    Mp, Mi = induced(Hs, Ht, phi, k), induced(Hs, Ht, inc, k)
    if Hs.p is None:
        return _zsolve(Mi, Mp, Hs.orders[k], Ht.orders[k])
    return _fsolve(Mi, Mp, Hs.p) if len(Hs.gens[k]) == len(Ht.gens[k]) else None


def index_maps(Hs, Ht, phi, inc):
    """Check the inclusion in every degree, including degrees with zero source homology.

    Return the nonzero source degrees and every failed degree. An empty source group does
    not justify omitting the inclusion check: its target group can be nonzero.
    """
    maps = {k: index_map(Hs, Ht, phi, inc, k) for k in range(Hs.cx.dim + 1)}
    return {k: value for k, value in maps.items() if Hs.gens[k] or value is None}


# ---------------------------------------------------------------- combinatorial maps, selectors

class BoxMap:
    """A combinatorial map on top cells: F(i) is the box of top cells lo <= j <= hi (inclusive,
    boxes[i] = (lo, hi) or None for empty), in the universal cover on periodic axes and clipped to
    the grid on the others, as CMGDB clips images."""

    def __init__(self, grid, boxes):
        self.grid, self.boxes, self._img = grid, dict(boxes), {}

    def __call__(self, i):
        if i not in self._img:
            b, g = self.boxes.get(i), self.grid
            rng = [range(lo, hi + 1) if g.periodic[a] else range(max(lo, 0), min(hi, g.shape[a] - 1) + 1)
                   for a, (lo, hi) in enumerate(zip(*b))] if b else [[]]
            self._img[i] = frozenset(g.norm_index(j) for j in product(*rng))
        return self._img[i]

    def image(self, tops):
        out = set()
        for i in tops:
            out |= self(i)
        return out

    @classmethod
    def from_sets(cls, grid, F):
        """From {top: set of tops} whose images are products of intervals (cyclic on periodic axes),
        such as a CMGDB MapGraph on a uniform grid."""
        boxes = {}
        for i, img in F.items():
            img = {grid.norm_index(j) for j in img}
            if not img:
                boxes[i] = None
                continue
            lo, hi = [], []
            for a, n in enumerate(grid.shape):
                v = sorted({j[a] for j in img})
                if grid.periodic[a] and len(v) < n:
                    t = max(range(len(v)), key=lambda t: ((v[(t + 1) % len(v)] - v[t]) % n, -t))
                    s = v[(t + 1) % len(v)]
                    lo.append(s)
                    hi.append(s + (v[t] - s) % n)
                else:
                    lo.append(v[0])
                    hi.append(v[-1])
            if len(img) != prod(h - l + 1 for l, h in zip(lo, hi)):
                raise ValueError(f"F({i}) is not a box")
            boxes[i] = (tuple(lo), tuple(hi))
        return cls(grid, boxes)

    @classmethod
    def from_function(cls, grid, lower, upper, f):
        """From a CMGDB box map f([lo_0..lo_{d-1}, hi_0..hi_{d-1}]) -> image in the same layout:
        F(i) = the cells whose closed box meets the closed image. CMGDB also pads the image by
        2^-50 of the domain width, and on a periodic axis it misses the cell across the seam that
        an image touches at the upper bound; neither is reproduced here."""
        d = grid.dim
        w = [(upper[a] - lower[a]) / grid.shape[a] for a in range(d)]
        boxes = {}
        for i in grid.tops():
            y = f(cmgdb_box(i, lower, upper, grid.shape))
            boxes[i] = (tuple(ceil((y[a] - lower[a]) / w[a]) - 1 for a in range(d)),
                        tuple(floor((y[d + a] - lower[a]) / w[a]) for a in range(d)))
        return cls(grid, boxes)


def index_pair(F, S):
    """P1 = S u F(S), P0 = P1 minus S, the target (T1, T0) = (P1 u F(P0), P0 u F(P0)), and
    isolated = (F(P0) n S is empty), the check of CMGDB's ComputeConleyIndexForCells."""
    S = set(S)
    P1 = S | F.image(S)
    P0 = P1 - S
    FP0 = F.image(P0)
    return dict(S=S, P1=P1, P0=P0, T1=P1 | FP0, T0=P0 | FP0, isolated=not FP0 & S)


def _center(lo, hi):
    return lo + 2 * ((hi - lo) // 4)


def _contract(e, K):
    """Box contraction D (dD + Dd = 1 - i eps) of the cell e in the box K toward its center vertex,
    axis 0 first. A reflection through the center of K commutes with D, which keeps selectors
    invariant under stabilizers."""
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
    """Chain selector of the carrier K(c) = intersection of |F(xi)| over the top cells xi of
    `domain` (default: all) containing c, a box (boxes of neighbors are aligned modulo periods by
    proximity). For an index pair, domain = P1: the carrier is then defined on |P1|, maps |P0| into
    |F(P0)|, and contains the carrier over all top cells, so it induces the same index map. A
    vertex goes to the vertex of K(c) nearest to point(c) (point: grid units -> grid units), or to
    the center of K(c); a cell of positive dimension to the contraction of phi(d c) in K(c) toward
    its center. With a group G, phi is built on orbit representatives and extended by
    phi(h_# c) = h_# phi(c), which needs F and domain equivariant (checked through the carriers) and
    phi(c) invariant under the stabilizer of c (checked; observed to hold for free orbits, sign
    changes and rotations of order 4 about a vertex)."""

    def __init__(self, grid, F, G=None, point=None, domain=None):
        self.grid, self.F, self.G, self.point, self.memo = grid, F, G, point, {}
        self.domain = None if domain is None else {grid.norm_index(i) for i in domain}

    def carrier(self, c):
        g, K = self.grid, None
        nbrs = [[(x - 1) // 2] if x & 1 else [x // 2 - 1, x // 2] for x in c]
        nbrs = [js if per else [j for j in js if 0 <= j < n]
                for js, n, per in zip(nbrs, g.shape, g.periodic)]
        for j in product(*nbrs):
            j = g.norm_index(j)
            if self.domain is not None and j not in self.domain:
                continue
            box = self.F.boxes.get(j)
            if not box or any(lo > hi for lo, hi in zip(*box)):
                raise ValueError(f"empty image of the top cell {j}")
            B = []
            for a, (lo, hi) in enumerate(zip(*box)):
                lo, hi = 2 * lo, 2 * hi + 2
                if g.periodic[a] and K is not None:
                    P = 2 * g.shape[a]
                    s = P * round((K[a][0] + K[a][1] - lo - hi) / (2 * P))
                    lo, hi = lo + s, hi + s
                elif not g.periodic[a]:
                    lo, hi = max(lo, 0), min(hi, 2 * g.shape[a])
                    if lo > hi:
                        raise ValueError(f"the image of the top cell {j} misses the grid; enlarge it "
                                         "(CMGDB projects such images when S misses the boundary)")
                B.append((lo, hi))
            K = B if K is None else [(max(x[0], y[0]), min(x[1], y[1])) for x, y in zip(K, B)]
        if K is None:
            raise ValueError(f"no top cell of the domain contains {c}")
        for a, (lo, hi) in enumerate(K):
            if lo > hi:
                raise ValueError(f"empty carrier at {c}")
            if g.periodic[a] and hi - lo >= 2 * g.shape[a]:
                raise ValueError(f"carrier of {c} wraps around axis {a}")
        return K

    def _lift(self, e, K):
        out = []
        for a, (x, (lo, hi)) in enumerate(zip(e, K)):
            o = x & 1
            if self.grid.periodic[a]:
                P = 2 * self.grid.shape[a]
                x -= P * ((x - o - lo) // P)
            if x - o < lo or x + o > hi:
                raise ValueError(f"selector leaves the carrier: {e} not in {K}")
            out.append(x)
        return tuple(out)

    def __call__(self, c):
        if c in self.memo:
            return self.memo[c]
        G = self.G
        if G is not None:
            r, h, s = G.orbit(c)
            if r != c:
                out = {e: s * v for e, v in G.act_chain(h, self(r)).items()}
                K = self.carrier(c)
                for e in out:
                    self._lift(e, K)
                self.memo[c] = out
                return out
        out = self._build(c)
        if G is not None and any(G.act_chain(h, out) != out for h in G.stabilizer(c)):
            raise ValueError(f"phi({c}) is not invariant under the stabilizer of {c}")
        self.memo[c] = out
        return out

    def _build(self, c):
        g, K = self.grid, self.carrier(c)
        if cdim(c) == 0:
            if self.point is None:
                return {g.norm(tuple(_center(lo, hi) for lo, hi in K)): 1}
            y = self.point(tuple(x / 2 for x in c))
            v = []
            for a, (lo, hi) in enumerate(K):
                x = 2 * round(y[a])
                if g.periodic[a]:
                    P = 2 * g.shape[a]
                    x += P * round(((lo + hi) / 2 - x) / P)
                v.append(min(max(x, lo), hi))
            return {g.norm(tuple(v)): 1}
        z = {}
        for f, s in g.faces(c):
            _add(z, self(f), s)
        z = _clean(z)
        out = {}
        for e, v in z.items():
            for e2, w in _contract(self._lift(e, K), K).items():
                e2 = g.norm(e2)
                out[e2] = out.get(e2, 0) + v * w
        out = _clean(out)
        if g.boundary(out) != z:
            raise AssertionError(f"selector is not a chain map at {c}")
        return out


# ---------------------------------------------------------------- groups of signed permutations

def _parity(seq):
    return -1 if sum(seq[i] > seq[j] for i in range(len(seq)) for j in range(i + 1, len(seq))) % 2 else 1


class Group:
    """Finite group of affine signed permutations g(x)_a = sign[a] x_{perm[a]} + shift[a] of grid
    units, generated by gens = [(perm, sign, shift), ...] (shifts reduced modulo the period on
    periodic axes). The grid must be invariant: a periodic axis goes to a periodic axis of the same
    size, and a non-periodic axis a onto axis perm[a] of the same size by x -> x (shift 0) or
    x -> shape[a] - x (sign -1, shift shape[a])."""

    def __init__(self, grid, gens):
        self.grid, d = grid, grid.dim
        self.gens = [self._norm(tuple(tuple(int(x) for x in part) for part in g)) for g in gens]
        for p, s, b in self.gens:
            if sorted(p) != list(range(d)) or any(x not in (1, -1) for x in s):
                raise ValueError("not a signed permutation")
            for a in range(d):
                q, n = p[a], grid.shape[a]
                ok = grid.shape[q] == n and grid.periodic[q] == grid.periodic[a]
                if ok and not grid.periodic[a]:
                    ok = (s[a], b[a]) in ((1, 0), (-1, n))
                if not ok:
                    raise ValueError(f"generator does not map the grid to itself on axis {a}")
        e = (tuple(range(d)), (1,) * d, (0,) * d)
        self.identity, self.word, self.elements, todo = e, {e: ()}, [e], [e]
        while todo:
            x = todo.pop(0)
            for t, g in enumerate(self.gens):
                y = self.compose(g, x)
                if y not in self.word:
                    self.word[y] = (t,) + self.word[x]
                    self.elements.append(y)
                    todo.append(y)
            if len(self.elements) > 10 ** 5:
                raise ValueError("group too large")
        self._orb = {}

    def __len__(self):
        return len(self.elements)

    def _norm(self, g):
        p, s, b = g
        return p, s, self.grid.norm_index(b)

    def compose(self, g, h):
        """g o h."""
        (pg, sg, bg), (ph, sh, bh) = g, h
        r = range(len(pg))
        return self._norm((tuple(ph[pg[a]] for a in r), tuple(sg[a] * sh[pg[a]] for a in r),
                           tuple(sg[a] * bh[pg[a]] + bg[a] for a in r)))

    def act(self, g, c, signed=True):
        """(cell, sign): g_# c = sign * cell."""
        p, s, b = g
        img = self.grid.norm(tuple(s[a] * c[p[a]] + 2 * b[a] for a in range(len(c))))
        if not signed:
            return img, 1
        inv = {q: a for a, q in enumerate(p)}
        tgt = [inv[q] for q in range(len(c)) if c[q] & 1]
        return img, prod(s[a] for a in tgt) * _parity(tgt)

    def act_chain(self, g, chain, signed=True):
        out = {}
        for c, v in chain.items():
            c2, s = self.act(g, c, signed)
            out[c2] = out.get(c2, 0) + s * v
        return _clean(out)

    def act_top(self, g, i):
        return tuple((x - 1) // 2 for x in self.act(g, self.grid.top(i), False)[0])

    def stabilizer(self, c):
        return [g for g in self.elements if self.act(g, c, False)[0] == c]

    def orbit(self, c):
        """(r, h, s): r the smallest cell of the orbit of c, and h_# r = s c. Checks the grid
        condition on the orbit: the stabilizer of r fixes it pointwise."""
        if c not in self._orb:
            r = min(self.act(g, c, False)[0] for g in self.elements)
            info = {}
            for g in self.elements:
                c2, s = self.act(g, r)
                if c2 == r and any(g[0][a] != a or g[1][a] != 1 for a in range(len(r)) if r[a] & 1):
                    raise ValueError(f"grid condition fails at {r}")
                if c2 in info and info[c2][2] != s:
                    raise ValueError(f"orientations along the orbit of {r} disagree")
                info.setdefault(c2, (r, g, s))
            self._orb.update(info)
        return self._orb[c]

    def fixed(self, cells):
        """The cells fixed by every element (pointwise, under the grid condition)."""
        return {c for c in cells if len(self.stabilizer(c)) == len(self.elements)}

    def character(self, values):
        """The homomorphism G -> {1, -1} with value values[t] on gens[t] (checked)."""
        chi = {g: prod(values[t] for t in self.word[g]) for g in self.elements}
        if any(chi[g] != values[t] for t, g in enumerate(self.gens)) or any(
                chi[self.compose(h, g)] != chi[h] * chi[g] for h in self.gens for g in self.elements):
            raise ValueError(f"no character with values {values} on the generators")
        return chi

    def characters(self):
        out = []
        for vals in product((1, -1), repeat=len(self.gens)):
            try:
                chi = self.character(vals)
            except ValueError:
                continue
            if chi not in out:
                out.append(chi)
        return out


class OrbitComplex(Complex):
    """C(N, L) tensored over ZG with Z_chi (chi a character with values +-1, trivial by default),
    with orientations transported along orbits: one cell r per orbit (the smallest), with its grid
    orientation, and c = s h_# r read as s chi(h) [r]. Orbits on which chi is nontrivial on the
    stabilizer are dropped, so this is the complex of (N/G, (L u N_chi)/G) twisted by chi; for
    G = Z2 and chi the sign, N_chi = N^G (prop:z2_reconstruction). For the untwisted term relative
    to N^G, put N^G into L and take chi trivial."""

    def __init__(self, cx, G, chi=None):
        self.src, self.G = cx, G
        self.chi = chi or {g: 1 for g in G.elements}
        for c in cx.cells | cx.zero:
            home = cx.cells if c in cx.cells else cx.zero
            if any(G.act(g, c, False)[0] not in home for g in G.gens):
                raise ValueError(f"the pair is not invariant at {c}")
        self._keep = {r for r in (G.orbit(c)[0] for c in cx.cells)
                      if all(self.chi[h] == 1 for h in G.stabilizer(r))}
        super().__init__(self._keep, {r: self.project(cx.bd[r]) for r in self._keep}, cx.dim)
        self.check()

    def project(self, chain):
        """The image of a chain of C(N, L) (cells of L are 0)."""
        out = {}
        for c, v in chain.items():
            if c in self.src.zero:
                continue
            if c not in self.src.cells:
                raise ValueError(f"cell {c} is not in the pair")
            r, h, s = self.G.orbit(c)
            if r in self._keep:
                out[r] = out.get(r, 0) + v * s * self.chi[h]
        return _clean(out)


def commutes(cx, G, g, signed=True):
    """Whether g_# (signed, or the plain permutation of cells) commutes with d on C(N, L)."""
    return all(cx.restrict(G.act_chain(g, cx.bd[c], signed)) ==
               cx.d(cx.restrict(G.act_chain(g, {c: 1}, signed))) for c in cx.cells)


def action(H, G, g, k, signed=True):
    """Matrix of g_* on H_k of an invariant pair."""
    return induced(H, H, lambda c: H.cx.restrict(G.act_chain(g, {c: 1}, signed)), k)


def _rank(vecs, p):
    piv, r = {}, 0
    for v0 in vecs:
        v = {i: x for i, x in ((i, _red(a, p)) for i, a in v0.items()) if x}
        while v:
            j = min(v)
            if j not in piv:
                iv = _inv(v[j], p)
                piv[j] = {t: _red(a * iv, p) for t, a in v.items()}
                r += 1
                break
            c = v[j]
            for t, a in piv[j].items():
                x = _red(v.get(t, 0) - c * a, p)
                if x:
                    v[t] = x
                else:
                    v.pop(t, None)
    return r


def isotypic_betti(cx, G, weights=None, ring="Q"):
    """Betti numbers of the subcomplex im P of C(N, L), P = sum_g w(g) g_# in Q[G], from ranks:
    rank P_k - rank (d P)_k - rank (d P)_{k+1}. For a character w with values +-1 (default: the
    trivial one) im P is the eigen-subcomplex of w; other idempotents give the other isotypic
    components, e.g. w = {1: 1/2, r^2: -1/2} the part where a rotation r of order 4 has r^2 = -1.
    Over Q or F_p, p not dividing |G|. Independent of OrbitComplex."""
    p = _field(ring)
    if p is None or (p and len(G) % p == 0):
        raise ValueError("needs a field whose characteristic does not divide |G|")
    weights = weights or {g: 1 for g in G.elements}
    rP, rdP = [], []
    for k in range(cx.dim + 1):
        P = []
        for c in cx.basis[k]:
            v = {}
            for g, w in weights.items():
                c2, s = G.act(g, c)
                v[c2] = v.get(c2, 0) + s * Fraction(w)
            P.append(cx.restrict(v))
        rP.append(_rank(P, p))
        rdP.append(_rank([cx.d(v) for v in P], p))
    rdP.append(0)
    return [rP[k] - rdP[k] - rdP[k + 1] for k in range(cx.dim + 1)]


def check_z2_reconstruction(cx, G, ring="Q"):
    """prop:z2_reconstruction for G = {1, g} over a field of characteristic != 2: H(N, L) against
    the orbit complex (+1 part) plus the twisted complex relative to N^G (-1 part), each part also
    from the eigen-subcomplexes (isotypic_betti) and from the eigenspaces of g_* on H(N, L). Also
    the untwisted term H(N/G, (L u N^G)/G), which replaces the twisted one only in special cases."""
    p = _field(ring)
    if len(G) != 2 or p is None or p == 2:
        raise ValueError("needs G of order 2 and a field of characteristic != 2")
    g = G.elements[1]
    sign = {G.identity: 1, g: -1}
    H = Homology(cx, ring)
    N = cx.cells | cx.zero
    out = dict(upstairs=H.betti, orbit=Homology(OrbitComplex(cx, G), ring).betti,
               twisted=Homology(OrbitComplex(cx, G, sign), ring).betti,
               untwisted_rel_fixed=Homology(OrbitComplex(relative(G.grid, N, cx.zero | G.fixed(N)), G),
                                            ring).betti,
               plus_subcomplex=isotypic_betti(cx, G, None, ring),
               minus_subcomplex=isotypic_betti(cx, G, sign, ring))
    for lam, key in ((1, "plus_eigen"), (-1, "minus_eigen")):
        out[key] = []
        for k in range(cx.dim + 1):
            M = action(H, G, g, k)
            n = len(M)
            out[key].append(n - _rank([{j: M[i][j] - lam * (i == j) for j in range(n)} for i in range(n)], p))
    out["holds"] = (out["upstairs"] == [a + b for a, b in zip(out["orbit"], out["twisted"])]
                    and out["plus_subcomplex"] == out["plus_eigen"] == out["orbit"]
                    and out["minus_subcomplex"] == out["minus_eigen"] == out["twisted"])
    return out


# ---------------------------------------------------------------- index pairs and orbit spaces

def conley_index(grid, F, S, ring="Z", G=None, chi=None, rel=None, point=None):
    """Homology and index map of the index pair (P1, P0) of S (top cells) under F, with the
    Selector on P1 (point: its vertex rule).
    G: compute on the orbit complexes (OrbitComplex, twisted by chi) with an equivariant selector.
    rel: function N -> subcomplex of N added to the exit sets of both pairs (for example the cells
    on a fixed set, see reconstruct). Returns the pair, the Homology H of the pair and Ht of the
    target, the index map in each degree with H != 0 and every failed degree (None if inc_*
    is not an isomorphism, including zero source degrees), and
    two flags: touches_boundary (S meets the boundary of a non-periodic axis, where the index
    depends on the treatment of the boundary; CMGDB then leaves it undefined if an image misses
    the domain) and clipped (the cells of P1 whose image box reaches the boundary of a
    non-periodic axis: their images touch or leave the grid, and what leaves is dropped, as in
    CMGDB). An image that misses the grid entirely makes the selector raise."""
    pr = index_pair(F, S)
    if not pr["isolated"]:
        raise ValueError("F(P0) meets S: (P1, P0) is not an index pair")
    phi = Selector(grid, F, G, point, domain=pr["P1"])
    src, tgt = [], []
    for out, (T1, T0) in ((src, (pr["P1"], pr["P0"])), (tgt, (pr["T1"], pr["T0"]))):
        N, L = grid.cells_of(T1), grid.cells_of(T0)
        out.append(relative(grid, N, L | rel(N) if rel else L))
    src, tgt = src[0], tgt[0]
    T1 = tgt.cells | tgt.zero
    for c in src.cells | src.zero:
        if not set(phi(c)) <= (tgt.zero if c in src.zero else T1):
            raise ValueError(f"phi({c}) leaves the target pair")
    if rel:   # the relative summand needs phi(Pbar) in the fixed part of the target, not just in T0
        PT = rel(T1)
        if any(not set(phi(c)) <= PT for c in rel(src.cells | src.zero)):
            raise ValueError("phi does not map the fixed set into the fixed set (pass point=)")
    if G is not None:
        src, tgt = OrbitComplex(src, G, chi), OrbitComplex(tgt, G, chi)
        Phi, Inc = (lambda c: tgt.project(phi(c))), (lambda c: tgt.project({c: 1}))
    else:
        Phi, Inc = (lambda c: tgt.restrict(phi(c))), (lambda c: tgt.restrict({c: 1}))
    for M in (Phi, Inc):
        for c in src.cells:
            rhs = {}
            for f, v in src.bd[c].items():
                _add(rhs, M(f), v)
            if tgt.d(M(c)) != _clean(rhs):
                raise ValueError(f"not a chain map at {c} (selector not equivariant?)")
    Hs, Ht = Homology(src, ring), Homology(tgt, ring)
    inner = [a for a in range(grid.dim) if not grid.periodic[a]]
    clipped = sorted(i for i in pr["P1"] if F.boxes.get(i) and any(
        F.boxes[i][0][a] < 0 or F.boxes[i][1][a] >= grid.shape[a] for a in inner))
    return dict(P1=pr["P1"], P0=pr["P0"], H=Hs, Ht=Ht, clipped=clipped,
                touches_boundary=any(i[a] in (0, grid.shape[a] - 1) for i in pr["S"] for a in inner),
                index_map=index_maps(Hs, Ht, Phi, Inc))


def hyperplane(axis, *values):
    """rel function: the cells of N in the hyperplanes x_axis = v (grid units), a subcomplex."""
    vals = {2 * v for v in values}
    return lambda N: {c for c in N if c[axis] in vals}


def reconstruct(H, HP, m):
    """H_k(N, L) = H_k(Nbar, Lbar) + H_{k-m}(Nbar, Lbar u Pbar) from the homology H of (Nbar, Lbar)
    and HP of (Nbar, Lbar u Pbar), Pbar the part of Nbar on the fixed set, over one ring.
    m = 1: SO(2) acting on C (prop:rotational_reduction). m = 2: SO(3) acting on R^3, observed,
    not proved. Returns [(free rank or dimension, torsion)] for k = 0, ..., dim + m."""
    out = []
    for k in range(H.cx.dim + m + 1):
        f, t = 0, []
        for h, j in ((H, k), (HP, k - m)):
            if 0 <= j <= h.cx.dim:
                f, t = f + h.betti[j], t + h.torsion[j]
        out.append((f, sorted(t)))
    return out


def reconstructed_index(grid, F, S, fixed, m, ring="Z", point=None):
    """The index upstairs of the Morse set S of the orbit-space map F, by reconstruct: the two
    summands with their index maps. fixed: rel function giving Pbar (for example hyperplane).
    The selector must map Pbar into the fixed part of the target (checked); with point the
    orbit-space map on vertices, it does when that map preserves the fixed set."""
    a = conley_index(grid, F, S, ring, point=point)
    b = conley_index(grid, F, S, ring, rel=fixed, point=point)
    return dict(groups=reconstruct(a["H"], b["H"], m), orbit_space=a, relative_fixed=b,
                status="prop:rotational_reduction" if m == 1 else "observed, not proved")


# ---------------------------------------------------------------- CMGDB on a uniform grid

def cmgdb_shape(depth, dim):
    """Cells per axis after depth subdivisions: the node at depth t is bisected along t % dim."""
    return tuple(2 ** ((depth + dim - 1 - a) // dim) for a in range(dim))


def cmgdb_cell(m, depth, dim):
    """Multi-index of CMGDB's cell m on the uniform grid of the given depth (Morton order)."""
    i = [0] * dim
    for t in range(depth):
        i[t % dim] = 2 * i[t % dim] + ((m >> (depth - 1 - t)) & 1)
    return tuple(i)


def cmgdb_index(i, depth):
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
        """a b + c with one rounding (exact rationals, then the nearest float)."""
        return float(Fraction(a) * Fraction(b) + Fraction(c))


def cmgdb_box(i, lower, upper, shape):
    """CMGDB's bounds of the cell i ([lo_0..lo_{d-1}, hi_0..hi_{d-1}]), bit for bit as
    TreeGrid::geometry computes them in the build of this project: lo = t U + (1 - t) L with
    t = i/n and hi = s L + (1 - s) U with s = 1 - (i + 1)/n, where clang (arm64) contracts
    a b + c d to fma(a, b, c d). The two roundings differ in the last bit on about a third of the
    bounds of a domain such as [-0.1, 0.7]; a build without the contraction gives the plain
    formula, which cmgdb_cells also accepts."""
    lo, hi = [], []
    for a, n in enumerate(shape):
        t, s = i[a] / n, 1 - (i[a] + 1) / n
        lo.append(_fma(t, upper[a], (1 - t) * lower[a]))
        hi.append(_fma(s, lower[a], (1 - s) * upper[a]))
    return lo + hi


def cmgdb_cells(boxes, lower, upper, depth):
    """Multi-indices of CMGDB boxes on the uniform grid of the given depth. Each box must agree with
    the bounds of one cell (cmgdb_box) to 1e-9 of a cell width, so the integer cell is unique;
    CMGDB's own boxes agree bit for bit."""
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
    """Top cells of the Morse set v of a CMGDB Morse graph computed with subdiv_min = subdiv_max =
    depth: from its boxes, checked against the Morton order of its cell indices."""
    a = set(cmgdb_cells(morse_graph.morse_set_boxes(v), lower, upper, depth))
    if a != {cmgdb_cell(m, depth, len(lower)) for m in morse_graph.morse_set(v)}:
        raise ValueError("boxes and cell indices of the Morse set disagree")
    return a


def cmgdb_map(map_graph, grid, depth):
    """BoxMap of a CMGDB MapGraph on the uniform grid of the given depth."""
    d = grid.dim
    return BoxMap.from_sets(grid, {cmgdb_cell(m, depth, d): {cmgdb_cell(x, depth, d) for x in
                                                             map_graph.adjacencies(m)}
                                   for m in range(prod(grid.shape))})
