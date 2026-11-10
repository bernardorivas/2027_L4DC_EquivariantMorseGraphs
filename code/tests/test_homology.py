"""Known-answer checks of latent_equivariance.homology.

Each check returns its expected value (written from the paper, a hand computation or an
independent route) and the computed one; pytest asserts equality. Run this file as a script to
write a record of every check to a JSON file (default `homology-results.json`):

    python tests/test_homology.py [OUTPUT]
"""
import json
import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import networkx as nx
import pytest
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form

from latent_equivariance.homology import (BoxMap, Grid, Group, Homology, OrbitComplex, action,
                                          check_z2_reconstruction, cmgdb_box, cmgdb_cell,
                                          cmgdb_cells, cmgdb_index, cmgdb_map, cmgdb_morse_set,
                                          cmgdb_shape, commutes, conley_index, hyperplane,
                                          isotypic_betti, pair, reconstruct, reconstructed_index,
                                          relative)

CHECKS = {}


def test_character_respects_identity_and_repeated_generators():
    grid = Grid([2])
    reflection = ((0,), (-1,), (2,))
    trivial = ((0,), (1,), (0,))
    with_identity = Group(grid, [reflection, trivial])
    assert with_identity.character([-1, 1])[with_identity.gens[0]] == -1
    with pytest.raises(ValueError):
        with_identity.character([1, -1])
    repeated = Group(grid, [reflection, reflection])
    assert repeated.character([-1, -1])[repeated.gens[0]] == -1
    with pytest.raises(ValueError):
        repeated.character([1, -1])


def check(fn):
    CHECKS[fn.__name__] = fn
    return fn


def morse_sets(F, tops):
    """Recurrent strongly connected components of F (test helper; CMGDB computes these)."""
    D = nx.DiGraph()
    D.add_nodes_from(tops)
    D.add_edges_from((i, j) for i in tops for j in F(i))
    return [set(s) for s in nx.strongly_connected_components(D)
            if len(s) > 1 or D.has_edge(next(iter(s)), next(iter(s)))]


def doubling(r):
    return [max(min(2 * r[0], 1.0), -1.0), min(max(2 * r[1], -1.0), 1.0)]


C_KLEIN, OMEGA = 0.6 / (4 * math.pi), 0.1     # 4 pi c = 0.6, so f^2 repels with 1.6^2 > 2


def klein_h(y):
    return y + C_KLEIN * math.sin(4 * math.pi * y)


def klein_f2(r):
    """Box map of f^2, f(x, y) = (x + omega, h(y)), h increasing."""
    return [r[0] + 2 * OMEGA, klein_h(klein_h(r[1])), r[2] + 2 * OMEGA, klein_h(klein_h(r[3]))]


# ---------------------------------------------------------------- signs (check_signs.py)

@check
def doubling_on_a_mirror():
    """N = [-1, 1], L = [-1, -1/2] u [1/2, 1] (4 cells): the index pair of 0 for x -> 2x."""
    g = Grid([4])
    G = Group(g, [((0,), (-1,), (4,))])          # x -> -x is v -> 4 - v in grid units
    m = G.elements[1]
    F = BoxMap.from_function(g, [-1.0], [1.0], doubling)
    r = conley_index(g, F, {(1,), (2,)}, "Z")
    H = r["H"]
    expected = dict(pair=[4, 2], Z=["0", "Z"], F2=[0, 1], Q=[0, 1], signed_commutes=True,
                    plain_commutes=False, signed_action_H1=[[-1]], plain_image_of_cycle=[[1]],
                    index_map_Z={1: [[1]]}, index_map_F2={1: [[1]]})
    computed = dict(pair=[len(r["P1"]), len(r["P0"])], Z=H.groups(), F2=Homology(H.cx, 2).betti,
                    Q=Homology(H.cx, "Q").betti, signed_commutes=commutes(H.cx, G, m),
                    plain_commutes=commutes(H.cx, G, m, signed=False),
                    signed_action_H1=action(H, G, m, 1),
                    plain_image_of_cycle=action(H, G, m, 1, signed=False),
                    index_map_Z=r["index_map"], index_map_F2=conley_index(g, F, {(1,), (2,)}, 2)["index_map"])
    return dict(source="check_signs.py (1); next-steps 1.2", expected=expected, computed=computed)


def plain_orbit_dd(cx, G):
    """Largest |entry| of d d on the orbit complex with untransported (plain) orientations."""
    rep = {c: G.orbit(c)[0] for c in cx.cells}
    bd = {r: {} for r in set(rep.values())}
    for r in bd:
        for f, v in cx.bd[r].items():
            bd[r][rep[f]] = bd[r].get(rep[f], 0) + v
    worst = 0
    for r, b in bd.items():
        acc = {}
        for f, v in b.items():
            for f2, w in bd[f].items():
                acc[f2] = acc.get(f2, 0) + v * w
        worst = max([worst] + [abs(x) for x in acc.values()])
    return worst


@check
def klein_bottle():
    """T^2 on a 4 x 8 grid and iota(x, y) = (x + 1/2, -y); K = T^2/iota."""
    g = Grid([4, 8], periodic=True)
    G = Group(g, [((0, 1), (1, -1), (2, 0))])
    iota = G.elements[1]
    T = pair(g, g.tops())
    K, Kw = OrbitComplex(T, G), OrbitComplex(T, G, G.character([-1]))
    expected = dict(signed_commutes=True, plain_commutes=False, plain_orbit_dd=2,
                    K_Z=["Z", "Z + Z/2", "0"], K_F2=[1, 2, 1], K_Q=[1, 1, 0], twisted_Q=[0, 1, 1],
                    torus_Q=[1, 2, 1], z2_reconstruction_Q=True)
    computed = dict(signed_commutes=commutes(T, G, iota), plain_commutes=commutes(T, G, iota, False),
                    plain_orbit_dd=plain_orbit_dd(T, G), K_Z=Homology(K).groups(),
                    K_F2=Homology(K, 2).betti, K_Q=Homology(K, "Q").betti,
                    twisted_Q=Homology(Kw, "Q").betti, torus_Q=Homology(T, "Q").betti,
                    z2_reconstruction_Q=check_z2_reconstruction(T, G)["holds"])
    return dict(source="ex-klein.tex; check_signs.py (2)", expected=expected, computed=computed)


@check
def klein_index_of_R0():
    """The Morse set of R0 = {y = 0} of f^2 on an 8 x 16 grid of [0, 1] x [-1/2, 1/2]."""
    nx_, ny_ = 8, 16
    g = Grid([nx_, ny_], periodic=True)
    G = Group(g, [((0, 1), (1, -1), (nx_ // 2, 0))])
    iota, sign = G.elements[1], G.character([-1])
    F = BoxMap.from_function(g, [0.0, -0.5], [1.0, 0.5], klein_f2)
    S = next(s for s in morse_sets(F, g.tops()) if (0, ny_ // 2) in s)
    up = conley_index(g, F, S, "Z")
    orb = {ring: conley_index(g, F, S, ring, G=G) for ring in ("Z", 2, "Q")}
    tw = conley_index(g, F, S, "Q", G=G, chi=sign)
    expected = dict(rows=[7, 8], F_equivariant=True, up_Z=["0", "Z", "Z"], iota_on_H={1: [[-1]], 2: [[-1]]},
                    up_index_map={1: [[1]], 2: [[1]]}, orbit_Z=["0", "Z/2", "0"], orbit_index_map_Z={1: [[1]]},
                    orbit_F2=[0, 1, 1], orbit_index_map_F2={1: [[1]], 2: [[1]]}, orbit_Q=[0, 0, 0],
                    twisted_Q=[0, 1, 1], twisted_index_map_Q={1: [[1]], 2: [[1]]}, z2_reconstruction_Q=True)
    computed = dict(rows=sorted({j for _, j in S}),
                    F_equivariant=all(F(G.act_top(iota, i)) == {G.act_top(iota, j) for j in F(i)}
                                      for i in g.tops()),
                    up_Z=up["H"].groups(), iota_on_H={k: action(up["H"], G, iota, k) for k in (1, 2)},
                    up_index_map=up["index_map"], orbit_Z=orb["Z"]["H"].groups(),
                    orbit_index_map_Z=orb["Z"]["index_map"], orbit_F2=orb[2]["H"].betti,
                    orbit_index_map_F2=orb[2]["index_map"], orbit_Q=orb["Q"]["H"].betti,
                    twisted_Q=tw["H"].betti, twisted_index_map_Q=tw["index_map"],
                    z2_reconstruction_Q=check_z2_reconstruction(up["H"].cx, G)["holds"])
    return dict(source="ex-klein.tex", expected=expected, computed=computed)


@check
def minus_identity_on_a_square():
    """-I on [-1, 1]^2 (2 x 2 cells) relative to the boundary."""
    g = Grid([2, 2])
    G = Group(g, [((0, 1), (-1, -1), (2, 2))])
    N = g.cells_of(g.tops())
    L = {c for c in N if c[0] in (0, 4) or c[1] in (0, 4)}
    cx = relative(g, N, L)
    z2 = check_z2_reconstruction(cx, G)
    expected = dict(upstairs=[0, 0, 1], sign_part=[0, 0, 0], untwisted_rel_fixed=[0, 1, 1], holds=True,
                    twisted_Z=["0", "Z/2", "0"], untwisted_rel_fixed_Z=["0", "Z", "Z"])
    computed = dict(upstairs=z2["upstairs"], sign_part=z2["twisted"], untwisted_rel_fixed=z2["untwisted_rel_fixed"],
                    holds=z2["holds"], twisted_Z=Homology(OrbitComplex(cx, G, G.character([-1]))).groups(),
                    untwisted_rel_fixed_Z=Homology(OrbitComplex(relative(g, N, L | G.fixed(N)), G)).groups())
    return dict(source="equivariant.tex after prop:z2_reconstruction; Z values by hand", expected=expected,
                computed=computed)


@check
def antipodal_circle():
    """The antipodal map of a circle with two vertices and two edges."""
    g = Grid([2], periodic=True)
    G = Group(g, [((0,), (1,), (1,))])
    cx = pair(g, g.tops())
    T = OrbitComplex(cx, G, G.character([-1]))
    expected = dict(twisted_boundary={(1,): {(0,): -2}}, twisted_Z=["Z/2", "0"], twisted_Q=[0, 0],
                    twisted_F2=[1, 1], orbit_Z=["Z", "Z"], z2_reconstruction_Q=True)
    computed = dict(twisted_boundary={c: b for c, b in T.bd.items() if b}, twisted_Z=Homology(T).groups(),
                    twisted_Q=Homology(T, "Q").betti, twisted_F2=Homology(T, 2).betti,
                    orbit_Z=Homology(OrbitComplex(cx, G)).groups(),
                    z2_reconstruction_Q=check_z2_reconstruction(cx, G)["holds"])
    return dict(source="lean-formalization.md, item 7", expected=expected, computed=computed)


# ---------------------------------------------------------------- orbit spaces of continuous groups

@check
def sphere_table():
    """Table 1 of ex-sphere.tex by prop:rotational_reduction (m = 1), K = 8 intervals of [-1, 1],
    h(z) = z + H(z) (an Euler step with the zeros and stabilities of H, c = 1, z_t = 0.4)."""
    K, zt = 8, 0.4
    h = lambda z: z + z * (zt ** 2 - z ** 2) * (1 - z ** 2)
    g = Grid([K])
    F = BoxMap.from_function(g, [-1.0], [1.0], lambda r: [h(r[0]), h(r[1])])
    point = lambda v: (K / 2 * (h(2 * v[0] / K - 1) + 1),)    # h in grid units; fixes the poles
    sets = morse_sets(F, g.tops())
    named = {"N": {(7,)}, "S": {(0,)}, "E": {(3,), (4,)}, "T+": {(5,)}, "T-": {(2,)}}
    expected, computed = {}, {}
    for nm, S in named.items():
        r = reconstructed_index(g, F, S, hyperplane(0, 0, K), 1, "Z", point=point)
        computed[nm] = dict(is_morse_set=S in sets, ranks=[f for f, _ in r["groups"]],
                            maps_identity=all(M == [[1]] for x in ("orbit_space", "relative_fixed")
                                              for M in r[x]["index_map"].values()))
    for nm, ranks in (("N", [0, 0, 1]), ("S", [0, 0, 1]), ("E", [0, 1, 1]), ("T+", [1, 1, 0]), ("T-", [1, 1, 0])):
        expected[nm] = dict(is_morse_set=True, ranks=ranks, maps_identity=True)
    N = g.cells_of(g.tops())
    computed["S2"] = [f for f, _ in reconstruct(Homology(relative(g, N)), Homology(relative(g, N, hyperplane(0, 0, K)(N))), 1)]
    expected["S2"] = [1, 0, 1]
    # the reflection z -> -z on the pair of E: the -1 part, untwisted because the halves are disjoint
    E = conley_index(g, F, named["E"], "Q", point=point)
    z2 = check_z2_reconstruction(E["H"].cx, Group(g, [((0,), (-1,), (K,))]))
    computed["E_reflection"] = dict(orbit=z2["orbit"], twisted=z2["twisted"], untwisted=z2["untwisted_rel_fixed"])
    expected["E_reflection"] = dict(orbit=[0, 0], twisted=[0, 1], untwisted=[0, 1])
    return dict(source="ex-sphere.tex, tab:sphere and the last paragraph", expected=expected, computed=computed)


def _ring_cells(g, lo, hi):
    """Top cells of the cube [lo, hi)^d of g."""
    return [i for i in g.tops() if all(lo <= x < hi for x in i)]


@check
def rotation_groups_against_cubes():
    """reconstruct with m = 1 (SO(2) on C) and m = 2 (SO(3) on R^3, observed) against cubical
    homology upstairs: disk or ball relative to its boundary, annulus or shell relative to both."""
    g1 = Grid([4])                               # the radius on [0, 4] in grid units; 0 is fixed
    bar = {"disk": (g1.cells_of([(0,), (1,), (2,)]), {(6,)}),          # [0, 3] rel {3}
           "annulus": (g1.cells_of([(1,), (2,)]), {(2,), (6,)})}        # [1, 3] rel {1, 3}
    expected = {"m1_disk": [0, 0, 1], "m1_annulus": [0, 1, 1], "m2_disk": [0, 0, 0, 1], "m2_annulus": [0, 1, 0, 1]}
    computed = {}
    for m in (1, 2):
        g = Grid([8] * (m + 1))                  # upstairs: [1, 7]^d, the hole (3, 5)^d
        outer = g.cells_of(_ring_cells(g, 1, 7))
        shell = g.cells_of([i for i in _ring_cells(g, 1, 7) if i not in _ring_cells(g, 3, 5)])
        on_outer = lambda N: {c for c in N if any(x in (2, 14) for x in c)}
        up = {"disk": relative(g, outer, on_outer(outer)),
              "annulus": relative(g, shell, on_outer(shell) | {c for c in shell if all(6 <= x <= 10 for x in c)})}
        for nm, (N, L) in bar.items():
            f = reconstruct(Homology(relative(g1, N, L)), Homology(relative(g1, N, L | hyperplane(0, 0)(N))), m)
            computed[f"m{m}_{nm}"] = [x for x, _ in f]
            computed[f"m{m}_{nm}_cubes"] = Homology(up[nm]).betti
    expected.update({k + "_cubes": v for k, v in list(expected.items())})
    return dict(source="prop:rotational_reduction (m = 1); m = 2 observed, not proved",
                expected=expected, computed=computed)


@check
def chafee_type_z2xz2():
    """sigma = -I and rho = diag(1, -1) on [-2, 2]^2 (16 x 16 cells), F of (x, y) -> (2x, 3y): the
    origin has n = 2 unstable directions, so sigma acts by (-1)^n = 1 and rho by
    (-1)^(n(n+3)/2) = -1 on H_2 (ex-chafee.tex); fixed cells on both axes."""
    g = Grid([16, 16])
    G = Group(g, [((0, 1), (-1, -1), (16, 16)), ((0, 1), (1, -1), (0, 16))])
    sigma, rho = G.elements[1], G.elements[2]
    F = BoxMap.from_function(g, [-2.0, -2.0], [2.0, 2.0], lambda r: [2 * r[0], 3 * r[1], 2 * r[2], 3 * r[3]])
    S = {(7, 7), (7, 8), (8, 7), (8, 8)}
    up = conley_index(g, F, S, "Q")
    expected = dict(upstairs=[0, 0, 1], index_map={2: [[1]]}, sigma=[[1]], rho=[[-1]])
    computed = dict(upstairs=up["H"].betti, index_map=up["index_map"], sigma=action(up["H"], G, sigma, 2),
                    rho=action(up["H"], G, rho, 2))
    for chi in G.characters():
        key = f"chi(sigma)={chi[sigma]},chi(rho)={chi[rho]}"
        r = conley_index(g, F, S, "Q", G=G, chi=chi)
        computed[key] = dict(subcomplex=isotypic_betti(up["H"].cx, G, chi), orbit_complex=r["H"].betti,
                             index_map=r["index_map"])
        part = [0, 0, 1] if (chi[sigma], chi[rho]) == (1, -1) else [0, 0, 0]
        expected[key] = dict(subcomplex=part, orbit_complex=part, index_map={2: [[1]]} if part[2] else {})
    return dict(source="ex-chafee.tex, sign rule for n = 2", expected=expected, computed=computed)


@check
def rotation_of_order_four():
    """Isotypic components of Z4 = <r>, r(x, y) = (-y, x), on four corner squares (H_0 = Q[Z4])."""
    g = Grid([4, 4])
    G = Group(g, [((1, 0), (-1, 1), (4, 0))])
    r2 = G.compose(G.elements[1], G.elements[1])
    cx = pair(g, [(0, 0), (3, 0), (3, 3), (0, 3)])
    chars = G.characters()
    expected = dict(order=4, characters=2, trivial=[1, 0, 0], sign=[1, 0, 0], r2_minus1=[2, 0, 0])
    computed = dict(order=len(G), characters=len(chars), trivial=isotypic_betti(cx, G, chars[0]),
                    sign=isotypic_betti(cx, G, chars[1]),
                    r2_minus1=isotypic_betti(cx, G, {G.identity: Fraction(1, 2), r2: Fraction(-1, 2)}))
    return dict(source="regular representation of Z4", expected=expected, computed=computed)


# ---------------------------------------------------------------- independent routes

def sympy_homology(cx):
    """Free ranks and torsion from sympy's Smith normal forms of the dense boundary matrices."""
    idx = {k: {c: n for n, c in enumerate(cx.basis[k])} for k in cx.basis}
    facs = {}
    for k in range(1, cx.dim + 1):
        rows, cols = len(cx.basis[k - 1]), len(cx.basis[k])
        facs[k] = []
        if rows and cols:
            M = sp.zeros(rows, cols)
            for j, c in enumerate(cx.basis[k]):
                for f, v in cx.bd[c].items():
                    M[idx[k - 1][f], j] += v
            D = smith_normal_form(M, domain=sp.ZZ)
            facs[k] = [abs(int(D[i, i])) for i in range(min(D.shape)) if D[i, i] != 0]
    return [(len(cx.basis[k]) - len(facs.get(k, [])) - len(facs.get(k + 1, [])),
             sorted(d for d in facs.get(k + 1, []) if d > 1)) for k in range(cx.dim + 1)]


@check
def engine_against_sympy():
    """Homology over Z against sympy, and over F_2, F_3, Q against universal coefficients."""
    rnd = random.Random(1)
    complexes = []
    for _ in range(60):
        shape = rnd.choice([(6,), (4, 4), (5, 3), (3, 3, 3), (4, 3, 2)])
        g = Grid(shape, [rnd.random() < 0.4 for _ in shape])
        P1 = [i for i in g.tops() if rnd.random() < 0.6]
        complexes.append(pair(g, P1, [i for i in P1 if rnd.random() < 0.3]))
    g = Grid([4, 8], periodic=True)
    G = Group(g, [((0, 1), (1, -1), (2, 0))])
    complexes += [OrbitComplex(pair(g, g.tops()), G, chi) for chi in (None, G.character([-1]))]
    g = Grid([2, 2, 2])                          # antipodal map of the boundary of a cube: RP^2
    G = Group(g, [((0, 1, 2), (-1, -1, -1), (2, 2, 2))])
    N = g.cells_of(g.tops())
    sphere = relative(g, {c for c in N if any(x in (0, 4) for x in c)})
    complexes += [OrbitComplex(sphere, G, chi) for chi in (None, G.character([-1]))]
    bad, torsion = 0, 0
    for cx in complexes:
        H, ref = Homology(cx), sympy_homology(cx)
        bad += [(H.betti[k], H.torsion[k]) for k in range(cx.dim + 1)] != ref
        torsion += any(t for _, t in ref)
        for p in (2, 3):
            uct = [ref[k][0] + sum(t % p == 0 for t in ref[k][1]) + (sum(t % p == 0 for t in ref[k - 1][1]) if k else 0)
                   for k in range(cx.dim + 1)]
            bad += Homology(cx, p).betti != uct
        bad += Homology(cx, "Q").betti != [f for f, _ in ref]
    return dict(source="sympy smith_normal_form; universal coefficient theorem",
                expected=dict(complexes=64, with_torsion=4, mismatches=0),
                computed=dict(complexes=len(complexes), with_torsion=torsion, mismatches=bad))


# ---------------------------------------------------------------- CMGDB

def _cmgdb():
    return pytest.importorskip("CMGDB")


@check
def cmgdb_cells_and_morton_order():
    """phase_space_box of every cell against the Morton decoding of its index, on uniform grids."""
    CMGDB = _cmgdb()
    cases = [(1, 3, [-1.0], [1.0]), (2, 5, [-1.0, 0.0], [1.0, 3.0]), (2, 6, [0.0, -0.5], [1.0, 0.5]),
             (3, 7, [-1.0, -2.0, 0.5], [1.0, 2.0, 1.5])]
    ok = []
    for dim, depth, lo, hi in cases:
        model = CMGDB.Model(depth, depth, depth, 10000, lo, hi, lambda r: list(r))
        mg, _ = CMGDB.ComputeMorseGraph(model)
        n = math.prod(cmgdb_shape(depth, dim))
        cells = cmgdb_cells([mg.phase_space_box(m) for m in range(n)], lo, hi, depth)
        ok.append(cells == [cmgdb_cell(m, depth, dim) for m in range(n)]
                  and all(cmgdb_index(i, depth) == m for m, i in enumerate(cells))
                  and all(mg.phase_space_box(m) == cmgdb_box(i, lo, hi, cmgdb_shape(depth, dim))
                          for m, i in enumerate(cells)))
    return dict(source="CMGDB TreeGrid::geometry, depth % dim", expected=[True] * len(cases), computed=ok)


def _shift_class(CMGDB, index_maps, betti):
    """CMGDB's shift-class strings of our F_5 index maps (a complex with zero boundary)."""
    counts = list(betti)
    entries = [[(i, j, int(x) % 5) for i, row in enumerate(index_maps.get(k, [])) for j, x in enumerate(row)
                if int(x) % 5] for k in range(len(counts))]
    while len(counts) > 1 and counts[-1] == 0:
        counts.pop()
        entries.pop()
    return CMGDB.ComputeRelativeShiftClass(counts, [[] for _ in counts], entries)["shift_class"]


def _neg2(r):
    return [max(min(-2 * r[1], 1.0), -1.0), min(max(-2 * r[0], -1.0), 1.0)]


def _product(r):
    f = lambda x: 1.5 * x - 0.5 * x ** 3
    return [f(r[0]), 0.5 * r[1], f(r[2]), 0.5 * r[3]]


def _rotate(r):
    xs, ys = [-2 * r[1], -2 * r[3]], [2 * r[0], 2 * r[2]]
    c = lambda v: max(min(v, 1.0), -1.0)
    return [c(min(xs)), c(min(ys)), c(max(xs)), c(max(ys))]


@check
def cmgdb_f5_indices():
    """For every Morse set of CMGDB on a uniform grid with a defined annotation, our F_5 index map on
    CMGDB's own pair (its MapGraph, converted exactly) has CMGDB's shift class."""
    CMGDB = _cmgdb()
    cases = {"doubling": (doubling, [-1.0], [1.0], 4, None),
             "minus_2x": (_neg2, [-1.0], [1.0], 4, None),
             "circle_doubling": (lambda r: [2 * r[0], 2 * r[1]], [0.0], [1.0], 5, [True]),
             "klein_f2_seam_off_circles": (klein_f2, [0.0, -0.125], [1.0, 0.875], 8, [True, True]),
             "product": (_product, [-1.5, -1.0], [1.5, 1.0], 8, None),
             "rotation_expansion": (_rotate, [-1.0, -1.0], [1.0, 1.0], 6, None)}
    expected, computed, info = {}, {}, {}
    for nm, (f, lo, hi, depth, per) in cases.items():
        args = (depth, depth, depth, 10000, lo, hi) + ((per,) if per else ()) + (f,)
        mg, mapg = CMGDB.ComputeConleyMorseGraph(CMGDB.Model(*args))
        g = Grid(cmgdb_shape(depth, len(lo)), per or False)
        F = cmgdb_map(mapg, g, depth)
        agree, undefined, maps = 0, 0, set()
        for v in range(mg.num_vertices()):
            ann = mg.annotations(v)
            if not ann:
                undefined += 1
                continue
            r = conley_index(g, F, cmgdb_morse_set(mg, v, lo, hi, depth), 5)
            mine = _shift_class(CMGDB, r["index_map"], r["H"].betti)
            agree += ann[:len(mine)] == mine and all(s == "0" for s in ann[len(mine):])
            maps |= set(mine) - {"0"}
        computed[nm] = agree == mg.num_vertices() - undefined
        expected[nm] = True
        info[nm] = dict(depth=depth, sets=mg.num_vertices(), undefined_in_cmgdb=undefined,
                        defined_and_equal=agree, nonzero_shift_classes=sorted(maps))
    return dict(source="CMGDB ComputeConleyMorseGraph annotations; ComputeRelativeShiftClass",
                expected=expected, computed=computed, info=info)


@check
def cmgdb_periodic_seam():
    """Characterization of CMGDB 1.5.3+fork.7.dev0: on a periodic axis an image box whose upper bound
    is the upper bound of the domain does not cover the cell across the seam (one whose lower bound
    is the lower bound does). It splits an invariant circle on the seam into two Morse sets."""
    CMGDB = _cmgdb()
    rot = lambda r: [r[0] + 0.25, r[1] + 0.25]
    mg, mapg = CMGDB.ComputeMorseGraph(CMGDB.Model(2, 2, 2, 10000, [0.0], [1.0], [True], rot))
    g = Grid([4], True)
    F = BoxMap.from_function(g, [0.0], [1.0], rot)
    sets = {}
    for lo in (-0.5, -0.125):
        mg2, _ = CMGDB.ComputeMorseGraph(CMGDB.Model(8, 8, 8, 10000, [0.0, lo], [1.0, lo + 1], [True, True], klein_f2))
        sets[str(lo)] = mg2.num_vertices()
    expected = dict(cmgdb_F_of_cell_2=[(2,), (3,)], closed_cover_of_cell_2=[(0,), (2,), (3,)],
                    klein_morse_sets={"-0.5": 5, "-0.125": 4})
    computed = dict(cmgdb_F_of_cell_2=sorted(cmgdb_cell(j, 2, 1) for j in mapg.adjacencies(cmgdb_index((2,), 2))),
                    closed_cover_of_cell_2=sorted(F((2,))), klein_morse_sets=sets)
    return dict(source="observed; TreeGrid::coverAccept shifts a periodic image only if upper > bound",
                expected=expected, computed=computed)


# ---------------------------------------------------------------- pytest and the record

@pytest.mark.parametrize("name", list(CHECKS))
def test_known_answer(name):
    r = CHECKS[name]()
    assert r["computed"] == r["expected"]


def test_group_law_and_signs():
    """act(x y, c) = x_# y_# c with signs, for a group of order 48 on a grid with a periodic axis."""
    g = Grid([4, 4, 6], [False, False, True])
    G = Group(g, [((1, 0, 2), (1, -1, 1), (0, 4, 0)), ((0, 1, 2), (-1, 1, -1), (4, 0, 3)),
                  ((0, 1, 2), (1, 1, 1), (0, 0, 1))])
    cx = pair(g, g.tops())
    cells = random.Random(0).sample(sorted(cx.cells), 60)
    assert len(G) == 48 and all(commutes(cx, G, e) for e in G.elements)
    for x in G.elements:
        for y in G.elements:
            for c in cells:
                c1, s1 = G.act(y, c)
                c2, s2 = G.act(x, c1)
                assert G.act(G.compose(x, y), c) == (c2, s1 * s2)


def test_index_map_solve_with_mixed_torsion():
    from latent_equivariance.homology import _zsolve
    so = [0, 2, 4]                               # Z + Z/2 + Z/4
    Y = [[1, 0, 0], [1, 1, 0], [3, 0, 3]]
    assert _zsolve([[1, 0, 0], [0, 1, 0], [0, 0, 1]], Y, so, so) == Y
    P = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]        # the target basis in another order
    assert _zsolve(P, [[1, 0, 0], [3, 0, 3], [1, 1, 0]], so, [0, 4, 2]) == Y
    assert _zsolve([[2, 0, 0], [0, 1, 0], [0, 0, 1]], Y, so, so) is None   # not an isomorphism


def test_rejects_bad_input():
    g = Grid([4])
    with pytest.raises(ValueError):
        Group(Grid([2, 2]), [((1, 0), (1, 1), (0, 0))]).orbit((1, 1))   # a swap fixes a square, not pointwise
    with pytest.raises(ValueError):
        conley_index(g, BoxMap(g, {i: ((0,), (3,)) for i in g.tops()}), {(1,)})   # F(P0) meets S
    h = lambda z: z + z * (0.16 - z ** 2) * (1 - z ** 2)
    g8 = Grid([8])
    F8 = BoxMap.from_function(g8, [-1.0], [1.0], lambda r: [h(r[0]), h(r[1])])
    with pytest.raises(ValueError):                  # the center rule moves the pole off the fixed set
        reconstructed_index(g8, F8, {(7,)}, hyperplane(0, 0, 8), 1)
    F = BoxMap.from_function(g, [-1.0], [1.0], doubling)
    r = conley_index(g, F, {(1,), (2,)})        # images of the four cells touch or leave [-1, 1]
    assert r["clipped"] == [(0,), (1,), (2,), (3,)] and not r["touches_boundary"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k) if not isinstance(k, str) else k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in (sorted(x) if isinstance(x, (set, frozenset)) else x)]
    if isinstance(x, Fraction):
        return int(x) if x.denominator == 1 else str(x)
    return x


def main(path=None):
    import platform
    out = dict(module="code/src/latent_equivariance/homology.py", python=platform.python_version(),
               date=time.strftime("%Y-%m-%d"), checks={})
    try:
        from importlib.metadata import version
        out["cmgdb"] = version("CMGDB")
    except Exception:
        out["cmgdb"] = None
    for name, fn in CHECKS.items():
        t0 = time.time()
        try:
            r = fn()
            r["passed"] = r["computed"] == r["expected"]
        except pytest.skip.Exception as e:
            r = dict(passed=None, skipped=str(e))
        r["seconds"] = round(time.time() - t0, 2)
        out["checks"][name] = _jsonable(r)
        print(f"{name:32s} {r['passed']}  {r['seconds']} s")
    path = Path(path or "homology-results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)
    return all(r["passed"] is not False for r in out["checks"].values())


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else None) else 1)
