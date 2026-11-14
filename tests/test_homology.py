"""Known-answer checks of morsegraphs.homology.

The expected values come from the paper, from hand computations, or from an independent route
(ranks of boundary matrices, cubical homology of the lifted pairs, CMGDB's own annotations).
"""
import math
import random
from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from morsegraphs.homology import (BoxMap, Complex, Grid, Homology, cmgdb_box, cmgdb_cell, cmgdb_cells,
                                  cmgdb_index, cmgdb_map, cmgdb_morse_set, cmgdb_shape, conley_index,
                                  index_pair, reconstructed_index, relative)


def morse_sets(F):
    """The recurrent strongly connected components of F (a test helper; CMGDB computes these)."""
    tops = F.grid.tops()
    number = {t: n for n, t in enumerate(tops)}
    edges = [(number[i], number[j]) for i in tops for j in F(i)]
    A = csr_matrix((np.ones(len(edges)), tuple(zip(*edges))), shape=(len(tops), len(tops)))
    count, labels = connected_components(A, directed=True, connection="strong")
    out = []
    for c in range(count):
        members = np.flatnonzero(labels == c)
        if len(members) > 1 or A[members[0], members[0]]:
            out.append({tops[i] for i in members})
    return out


def box_map(f):
    """The box map of a monotone map of each coordinate given by its values at the corners."""
    def image(r):
        d = len(r) // 2
        lo, hi = f(r[:d]), f(r[d:])
        return [min(a, b) for a, b in zip(lo, hi)] + [max(a, b) for a, b in zip(lo, hi)]
    return image


def clip(x):
    return max(min(x, 1.0), -1.0)


def doubling(r):
    return [clip(2 * r[0]), clip(2 * r[1])]


def euler_height(z):
    """An Euler step of z' = z (0.4^2 - z^2)(1 - z^2): fixes -1, -0.4, 0, 0.4, 1, increasing."""
    return z + z * (0.4 ** 2 - z ** 2) * (1 - z ** 2)


SPHERE = [(0, "lower"), (0, "upper")]


# ---------------------------------------------------------------- homology over fields

def test_doubling_index_pair():
    """N = [-1, 1], L = [-1, -1/2] u [1/2, 1] on 4 cells: the index pair of 0 for x -> 2x."""
    g = Grid([4])
    F = BoxMap.from_function(g, [-1.0], [1.0], doubling)
    pr = index_pair(F, {(1,), (2,)})
    assert (len(pr["P1"]), len(pr["P0"]), pr["isolated"]) == (4, 2, True)
    for field in (2, 5, "Q"):
        r = conley_index(F, {(1,), (2,)}, field)
        assert r.betti == [0, 1] and r.maps == [[], [[1]]]
    assert conley_index(F, [1, 2]).betti == [0, 1]          # ids give the same cells in one dimension


def test_square_relative_to_its_boundary():
    g = Grid([2, 2])
    N = g.cells_of(g.tops())
    L = {c for c in N if c[0] in (0, 4) or c[1] in (0, 4)}
    for field in (2, 3, "Q"):
        assert Homology(relative(g, N, L), field).betti == [0, 0, 1]


def test_field_dependence():
    """The minimal cell structure of RP^2, d e2 = 2 e1: F_2 sees H_1 and H_2, F_3 and Q do not."""
    e0, e1, e2 = (0, 0), (1, 0), (1, 1)
    cx = Complex([e0, e1, e2], {e0: {}, e1: {}, e2: {e1: 2}}, 2)
    assert Homology(cx, 2).betti == [1, 1, 1]
    assert Homology(cx, 3).betti == [1, 0, 0] == Homology(cx, "Q").betti


def rank(M, p):
    """The rank of an integer matrix over F_p (p > 0) or Q (p = 0), by Gaussian elimination."""
    A = [[Fraction(x) if p == 0 else x % p for x in r] for r in M]
    r = 0
    for c in range(len(A[0]) if A else 0):
        piv = next((i for i in range(r, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = 1 / A[r][c] if p == 0 else pow(A[r][c], -1, p)
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c] * inv
                A[i] = [x - f * y if p == 0 else (x - f * y) % p for x, y in zip(A[i], A[r])]
        r += 1
    return r


def test_homology_against_ranks():
    """dim H_k = dim C_k - rank d_k - rank d_{k+1} on random relative cubical complexes."""
    rnd = random.Random(1)
    for _ in range(40):
        shape = rnd.choice([(6,), (4, 4), (5, 3), (3, 3, 3), (4, 3, 2)])
        g = Grid(shape)
        P1 = [i for i in g.tops() if rnd.random() < 0.6]
        cx = relative(g, g.cells_of(P1), g.cells_of([i for i in P1 if rnd.random() < 0.3]))
        index = {k: {c: n for n, c in enumerate(cx.basis[k])} for k in cx.basis}
        for field, p in ((2, 2), (3, 3), (5, 5), ("Q", 0)):
            ranks = [0]
            for k in range(1, g.dim + 1):
                M = [[0] * len(cx.basis[k]) for _ in cx.basis[k - 1]]
                for j, c in enumerate(cx.basis[k]):
                    for f, v in cx.bd[c].items():
                        M[index[k - 1][f]][j] += v
                ranks.append(rank(M, p))
            ranks.append(0)
            expected = [len(cx.basis[k]) - ranks[k] - ranks[k + 1] for k in range(g.dim + 1)]
            assert Homology(cx, field).betti == expected


def test_repeller_in_the_plane():
    """The origin of (x, y) -> (2x, 3y) on [-4, 4]^2 with 32 x 32 cells: H_2 = F, the identity."""
    g = Grid([32, 32])
    F = BoxMap.from_function(g, [-4.0, -4.0], [4.0, 4.0], lambda r: [2 * r[0], 3 * r[1], 2 * r[2], 3 * r[3]])
    for field in (5, "Q"):
        r = conley_index(F, {(15, 15), (15, 16), (16, 15), (16, 16)}, field)
        assert r.betti == [0, 0, 1] and r.maps[2] == [[1]]


# ---------------------------------------------------------------- reconstruction in X

def test_sphere_table():
    """SO(2) on S^2 (m = 1, P/G = {-1, 1}) for an Euler step of the height equation on 8 cells."""
    g = Grid([8])
    F = BoxMap.from_function(g, [-1.0], [1.0], box_map(lambda r: [euler_height(r[0])]))
    sets = morse_sets(F)
    expected = {"N": ({(7,)}, [0, 0, 1]), "S": ({(0,)}, [0, 0, 1]), "E": ({(3,), (4,)}, [0, 1, 1]),
                "T+": ({(5,)}, [1, 1, 0]), "T-": ({(2,)}, [1, 1, 0])}
    for name, (S, betti) in expected.items():
        r = reconstructed_index(F, S, SPHERE, 1)
        assert S in sets, name
        assert r.lifted.betti == betti, name
        assert all(M == [[1]] for x in (r.quotient, r.relative) for M in x.maps if M), name
        assert r.lifted.shift_class() == ["x-1" * (b > 0) or "0" for b in betti], name
    assert reconstructed_index(F, {(7,)}, SPHERE, 1).quotient.betti == [0, 0]
    assert reconstructed_index(F, {(7,)}, SPHERE, 1).relative.betti == [0, 1]


def _cubes(g, lo, hi):
    return [i for i in g.tops() if all(lo <= x < hi for x in i)]


def test_lifted_homology_against_cubes():
    """H_k(N, L) = H_k(N/G, L/G) + H_{k-m}(N/G, K/G) for the rotations of R^(m+1) (G/H = S^m), with
    the radius as X/G: a ball relative to its boundary sphere and a shell relative to both spheres,
    against the cubical homology of the same pairs in R^(m+1)."""
    g1 = Grid([4])                                   # the radius on [0, 4]; P/G = {0}
    quotient = {"ball": (g1.cells_of([(0,), (1,), (2,)]), {(6,)}),        # [0, 3] relative to {3}
                "shell": (g1.cells_of([(1,), (2,)]), {(2,), (6,)})}       # [1, 3] relative to {1, 3}
    expected = {(1, "ball"): [0, 0, 1], (1, "shell"): [0, 1, 1],
                (2, "ball"): [0, 0, 0, 1], (2, "shell"): [0, 1, 0, 1]}
    for m in (1, 2):
        g = Grid([8] * (m + 1))                      # [1, 7]^(m+1) with the hole (3, 5)^(m+1)
        outer = g.cells_of(_cubes(g, 1, 7))
        shell = g.cells_of([i for i in _cubes(g, 1, 7) if i not in _cubes(g, 3, 5)])
        on_outer = lambda N: {c for c in N if any(x in (2, 14) for x in c)}  # noqa: E731
        upstairs = {"ball": relative(g, outer, on_outer(outer)),
                    "shell": relative(g, shell, on_outer(shell) | {c for c in shell if all(6 <= x <= 10 for x in c)})}
        for name, (N, L) in quotient.items():
            a = Homology(relative(g1, N, L)).betti
            b = Homology(relative(g1, N, L | {c for c in N if c[0] == 0})).betti
            lifted = [(a[k] if k < 2 else 0) + (b[k - m] if 0 <= k - m < 2 else 0) for k in range(m + 2)]
            assert lifted == expected[(m, name)] == Homology(upstairs[name]).betti


@pytest.mark.parametrize("swap", [False, True])
def test_reconstruction_on_a_rectangle(swap):
    """(a, r) -> (0.45 a, 2r/(1 + r)) on [-1, 1] x [0, 1], P/G = {r = 0}: the origin lifts to a
    hyperbolic fixed point of R x R^(m+1) with m + 1 unstable directions, H_{m+1} = F, and (0, 1) to
    an attracting sphere S^m, H_0 = H_m = F. With swap the axes are exchanged, so P/G is a facet on
    the first axis."""
    order = (lambda v: v[::-1]) if swap else (lambda v: v)
    f = lambda v: order([0.45 * v[0], 2 * v[1] / (1 + v[1])])                      # noqa: E731
    g = Grid(order([16, 16]))
    lower, upper = order([-1.0, 0.0]), order([1.0, 1.0])
    F = BoxMap.from_function(g, lower, upper, box_map(lambda x: f(order(list(x)))))
    S = next(s for s in morse_sets(F) if order((7, 0)) in s)
    fixed = [(0 if swap else 1, "lower")]
    for m in (1, 2):
        r = reconstructed_index(F, S, fixed, m)
        assert r.quotient.betti == [0, 0, 0] and r.relative.betti == [0, 1, 0]
        assert r.lifted.betti == [int(k == m + 1) for k in range(m + 3)]
        assert r.lifted.maps[m + 1] == [[1]] and r.N_P
        minus = reconstructed_index(F, S, fixed, m, degree=-1)
        assert minus.lifted.maps[m + 1] == [[4]] and minus.lifted.shift_class()[m + 1] == "x+1"
    sink = next(s for s in morse_sets(F) if order((7, 15)) in s)
    r = reconstructed_index(F, sink, fixed, 2)
    assert r.lifted.betti == [1, 0, 1, 0, 0] and not r.N_P


def test_rejects_bad_input():
    g = Grid([4])
    with pytest.raises(ValueError):                  # F(P0) meets S
        conley_index(BoxMap(g, {i: ((0,), (3,)) for i in g.tops()}), {(1,)})
    with pytest.raises(ValueError):
        conley_index(BoxMap(g, {i: ((0,), (3,)) for i in g.tops()}), set())
    F = BoxMap(g, {(0,): ((0,), (0,)), (1,): ((1,), (1,)), (2,): ((2,), (3,)), (3,): ((2,), (2,))})
    assert conley_index(F, {(2,), (3,)}).betti == [1, 0]
    with pytest.raises(ValueError, match="misses the facet"):    # F moves the pole off P/G
        reconstructed_index(F, {(2,), (3,)}, [(0, "upper")], 1)
    with pytest.raises(ValueError):
        reconstructed_index(F, {(2,), (3,)}, [(1, "upper")], 1)
    with pytest.raises(ValueError):
        reconstructed_index(F, {(2,), (3,)}, [(0, "upper")], 0)


# ---------------------------------------------------------------- CMGDB

def _cmgdb():
    return pytest.importorskip("CMGDB")


def test_cmgdb_cells_and_morton_order():
    """phase_space_box of every cell against the Morton decoding of its number, on uniform grids."""
    CMGDB = _cmgdb()
    for dim, depth, lo, hi in [(1, 3, [-1.0], [1.0]), (2, 5, [-1.0, 0.0], [1.0, 3.0]),
                               (2, 6, [0.0, -0.5], [1.0, 0.5]), (3, 7, [-1.0, -2.0, 0.5], [1.0, 2.0, 1.5])]:
        mg, _ = CMGDB.ComputeMorseGraph(CMGDB.Model(depth, depth, depth, 10000, lo, hi, lambda r: list(r)))
        n = math.prod(cmgdb_shape(depth, dim))
        cells = cmgdb_cells([mg.phase_space_box(m) for m in range(n)], lo, hi, depth)
        assert cells == [cmgdb_cell(m, depth, dim) for m in range(n)]
        assert all(cmgdb_index(i, depth) == m for m, i in enumerate(cells))
        assert all(mg.phase_space_box(m) == cmgdb_box(i, lo, hi, cmgdb_shape(depth, dim))
                   for m, i in enumerate(cells))


def _neg2(r):
    return [clip(-2 * r[1]), clip(-2 * r[0])]


def _product(r):
    f = lambda x: 1.5 * x - 0.5 * x ** 3            # noqa: E731
    return [f(r[0]), 0.5 * r[1], f(r[2]), 0.5 * r[3]]


def _rotate(r):
    xs, ys = [-2 * r[1], -2 * r[3]], [2 * r[0], 2 * r[2]]
    return [clip(min(xs)), clip(min(ys)), clip(max(xs)), clip(max(ys))]


@pytest.mark.parametrize("name", ["doubling", "minus_2x", "product", "rotation_expansion"])
def test_cmgdb_f5_indices(name):
    """For every Morse set of CMGDB with a defined annotation, the F_5 index map on CMGDB's own pair
    (its map graph, converted exactly) has CMGDB's shift class."""
    CMGDB = _cmgdb()
    f, lo, hi, depth = {"doubling": (doubling, [-1.0], [1.0], 4), "minus_2x": (_neg2, [-1.0], [1.0], 4),
                        "product": (_product, [-1.5, -1.0], [1.5, 1.0], 8),
                        "rotation_expansion": (_rotate, [-1.0, -1.0], [1.0, 1.0], 6)}[name]
    mg, mapg = CMGDB.ComputeConleyMorseGraph(CMGDB.Model(depth, depth, depth, 10000, lo, hi, f))
    g = Grid(cmgdb_shape(depth, len(lo)))
    F = cmgdb_map(mapg, g, depth)
    defined = 0
    for v in range(mg.num_vertices()):
        ann = list(mg.annotations(v))
        if not ann:
            continue
        mine = conley_index(F, cmgdb_morse_set(mg, v, lo, hi, depth)).shift_class()
        n = max(len(ann), len(mine))
        assert ann + ["0"] * (n - len(ann)) == mine + ["0"] * (n - len(mine)), (name, v)
        defined += 1
    assert defined > 0


def test_relation_in_id_order():
    """BoxMap.from_relation on the relation in id order (last axis fastest) gives cmgdb_map."""
    CMGDB = _cmgdb()
    depth, lo, hi = 8, [-1.5, -1.0], [1.5, 1.0]
    mg, mapg = CMGDB.ComputeConleyMorseGraph(CMGDB.Model(depth, depth, depth, 10000, lo, hi, _product))
    g = Grid(cmgdb_shape(depth, 2))
    offsets, targets = (np.asarray(x) for x in mapg.csr_view())
    ids = [g.ids([cmgdb_cell(m, depth, 2)])[0] for m in range(len(offsets) - 1)]
    rows = sorted((ids[m], sorted(ids[int(t)] for t in targets[offsets[m]:offsets[m + 1]]))
                  for m in range(len(offsets) - 1))
    grid_offsets = np.cumsum([0] + [len(t) for _, t in rows])
    grid_targets = np.concatenate([t for _, t in rows])
    F, H = BoxMap.from_relation(g, grid_offsets, grid_targets), cmgdb_map(mapg, g, depth)
    assert all(F(i) == H(i) for i in g.tops())
    v = next(v for v in mg.vertices() if mg.annotations(v))
    S = cmgdb_morse_set(mg, v, lo, hi, depth)
    assert conley_index(F, g.ids(S)).shift_class() == list(mg.annotations(v))
