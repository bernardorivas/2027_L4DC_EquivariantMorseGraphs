"""Voltage reconstruction against independent finite-graph enumeration."""
from itertools import permutations

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from latent_equivariance.equivariant_graph import (forward_closure, morse_decomposition,
                                                  quotient_from_relation)


def reference(a):
    graph = nx.from_scipy_sparse_array(a, create_using=nx.DiGraph)
    comps = [tuple(sorted(c)) for c in nx.strongly_connected_components(graph)
             if len(c) > 1 or graph.has_edge(next(iter(c)), next(iter(c)))]
    comps.sort()
    relation = {(i, j) for i, c in enumerate(comps) for j, d in enumerate(comps)
                if i != j and nx.has_path(graph, c[0], d[0])}
    order_graph = nx.DiGraph()
    order_graph.add_nodes_from(range(len(comps)))
    order_graph.add_edges_from(relation)
    return comps, relation, set(nx.transitive_reduction(order_graph).edges())


def test_nonabelian_nonfree_random_relations():
    elements = list(permutations(range(3)))
    mul = np.asarray([[elements.index(tuple(g[h[i]] for i in range(3))) for h in elements]
                      for g in elements])
    # A natural three-point orbit, a regular six-point orbit, and a fixed point.
    action = np.asarray([list(g) + (3 + mul[k]).tolist() + [9] for k, g in enumerate(elements)])
    rng = np.random.default_rng(421)
    for _ in range(32):
        raw = rng.random((10, 10)) < .06
        equiv = np.zeros_like(raw)
        for p in action:
            equiv[np.ix_(p, p)] |= raw
        a = csr_matrix(equiv)
        quotient = quotient_from_relation(a, action, mul)
        assert (quotient.lift_relation() != a).nnz == 0
        report = quotient.lift_morse_graph()
        comps, relation, hasse = reference(a)
        assert [tuple(c) for c in report["lifted"].components] == comps
        assert set(report["lifted"].order) == relation
        assert set(report["lifted"].hasse) == hasse
        for component, base, stab in zip(report["base_components"], report["bases"], report["stabilizers"]):
            c = set(report["lifted"].components[component].tolist())
            assert set(stab) == {g for g in range(6) if set(action[g, list(c)]) == c}


def test_ineffective_action_and_quotient_loop():
    a = csr_matrix([[1, 1], [0, 1]])
    q = quotient_from_relation(a, [[0, 1], [0, 1]], [[0, 1], [1, 0]])
    assert q.edge_voltages(0, 1) == (0, 1)
    assert q.lift_morse_graph()["stabilizers"] == ((0, 1), (0, 1))
    swapped = quotient_from_relation(csr_matrix([[0, 1], [1, 0]]),
                                     [[0, 1], [1, 0]], [[0, 1], [1, 0]])
    assert swapped.adjacency[0, 0]
    assert tuple(swapped.lift_morse_graph()["lifted"].components[0]) == (0, 1)


def test_all_comparable_pairs_are_needed():
    # Vertex order p1,p2,q1,q2,r1,r2. The direct q1->p2 edge is lost by Hasse-only lifting.
    edges = [(i, i) for i in range(6)] + [(2, 4), (3, 5), (4, 0), (5, 1), (3, 0), (2, 1)]
    rows, cols = zip(*edges)
    a = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))
    q = quotient_from_relation(a, [list(range(6)), [1, 0, 3, 2, 5, 4]], [[0, 1], [1, 0]])
    report = q.lift_morse_graph()
    assert len(report["quotient"].hasse) == 2
    assert len(report["walk_voltages"]) == 3
    assert (2, 1) in report["lifted"].hasse
    assert set(report["lifted"].hasse) == reference(a)[2]


def test_outside_escape_and_forward_closure():
    a = csr_matrix([[1, 1, 0], [0, 1, 1], [0, 0, 0]])
    report = morse_decomposition(a, outside=2)
    assert len(report.components) == 2
    assert report.escapes == (True, True)
    assert list(forward_closure(a, [0])) == [0, 1, 2]


def test_attractor_lattice_and_meet_correspondence():
    from latent_equivariance.equivariant_graph import attractor_lattice, omega_limit
    # Two recurrent vertices share a terminal image. Their set intersection is not an attractor.
    a = csr_matrix([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    q = quotient_from_relation(a, [[0, 1, 2], [1, 0, 2]], [[0, 1], [1, 0]])
    up = attractor_lattice(a)["elements"]
    down = attractor_lattice(q.adjacency)["elements"]
    assert frozenset({0, 2}) in up and frozenset({1, 2}) in up
    assert omega_limit(a, {2}) == frozenset()
    invariant = [s for s in up if {int(q.action[1, v]) for v in s} == s]
    assert {frozenset(q.orbit_index[list(s)]) for s in invariant} == set(down)
    for s in invariant:
        for t in invariant:
            meet = omega_limit(a, s & t)
            projected = frozenset(q.orbit_index[list(meet)])
            assert projected == omega_limit(q.adjacency,
                set(q.orbit_index[list(s)]) & set(q.orbit_index[list(t)]))


def test_noncommuting_voltages_multiply_in_walk_order():
    elements = list(permutations(range(3)))
    mul = np.asarray([[elements.index(tuple(g[h[i]] for i in range(3))) for h in elements]
                      for g in elements])
    action = np.asarray([np.concatenate([6 * q + mul[h] for q in range(3)])
                         for h in range(6)])
    g1, g2 = elements.index((1, 0, 2)), elements.index((0, 2, 1))
    rows, cols = [], []
    for h in range(6):
        rows.extend([h, 6 + h])
        cols.extend([6 + mul[h, g1], 12 + mul[h, g2]])
    # Self-loops make each vertex a recurrent component without changing connecting voltages.
    rows.extend(range(18))
    cols.extend(range(18))
    a = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(18, 18))
    q = quotient_from_relation(a, action, mul)
    assert q.edge_voltages(0, 1) == (g1,)
    assert q.edge_voltages(1, 2) == (g2,)
    report = q.lift_morse_graph()
    assert report["walk_voltages"][0, 2] == (int(mul[g1, g2]),)
    assert mul[g1, g2] != mul[g2, g1]
    assert (q.lift_relation() != a).nnz == 0
