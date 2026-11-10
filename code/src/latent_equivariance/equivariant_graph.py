"""Quotient digraphs with voltages, recurrent components, and the full lifted order.

Arrows point from a source Morse component to a reachable target. The associated poset orders
that target below the source. The outside node is retained as an escape and is never recurrent.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components


@dataclass
class MorseDecomposition:
    components: tuple
    order: tuple
    hasse: tuple
    escapes: tuple
    labels: np.ndarray


def morse_decomposition(adjacency, outside=None):
    """Recurrent SCCs, all strict reachability relations, their reduction, and escape flags."""
    a = csr_matrix(adjacency, dtype=bool)
    if a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    ncomp, labels = connected_components(a, directed=True, connection="strong")
    counts = np.bincount(labels, minlength=ncomp)
    recurrent = counts > 1
    recurrent[labels[np.flatnonzero(a.diagonal())]] = True
    if outside is not None:
        if not 0 <= outside < a.shape[0]:
            raise ValueError("invalid outside vertex")
        if a.indptr[outside + 1] != a.indptr[outside]:
            raise ValueError("outside vertex must have no outgoing edges")
        recurrent[labels[outside]] = False
    comps = [np.flatnonzero(labels == i) for i in np.flatnonzero(recurrent)]
    comps.sort(key=lambda c: int(c[0]))
    relation, escapes = set(), []
    component_ids = [labels[c[0]] for c in comps]
    for i, c in enumerate(comps):
        reached = breadth_first_order(a, int(c[0]), directed=True, return_predecessors=False)
        reached_components = set(labels[reached].tolist())
        relation.update((i, j) for j, cj in enumerate(component_ids)
                        if i != j and cj in reached_components)
        escapes.append(outside is not None and labels[outside] in reached_components)
    hasse = {(i, j) for i, j in relation
             if not any((i, k) in relation and (k, j) in relation for k in range(len(comps)))}
    return MorseDecomposition(tuple(comps), tuple(sorted(relation)), tuple(sorted(hasse)),
                              tuple(escapes), labels)


def forward_closure(adjacency, vertices):
    """All vertices reachable from the given set, including any outside vertex."""
    a = csr_matrix(adjacency)
    seen = np.zeros(a.shape[0], dtype=bool)
    todo = [int(i) for i in vertices]
    if any(i < 0 or i >= len(seen) for i in todo):
        raise ValueError("vertex outside graph")
    seen[todo] = True
    while todo:
        v = todo.pop()
        targets = a.indices[a.indptr[v]:a.indptr[v + 1]]
        for w in targets[~seen[targets]]:
            seen[w] = True
            todo.append(int(w))
    return np.flatnonzero(seen)


def index_pair(adjacency, recurrent_set, outside=None):
    """Combinatorial source/target pair and its isolation check in vertex indices."""
    a = csr_matrix(adjacency)
    s = set(int(v) for v in recurrent_set)
    if any(v < 0 or v >= a.shape[0] for v in s):
        raise ValueError("vertex outside graph")
    image = lambda vs: set(int(w) for v in vs for w in a.indices[a.indptr[v]:a.indptr[v + 1]])
    p1 = s | image(s)
    p0 = p1 - s
    fp0 = image(p0)
    t1, t0 = p1 | fp0, p0 | fp0
    return dict(S=s, P1=p1, P0=p0, T1=t1, T0=t0, isolated=not bool(fp0 & s),
                escaped=outside is not None and outside in t1)


class QuotientGraph:
    """Quotient relation with one voltage bit set per group element.

The action is stored as permutations of the upstairs vertices. Multiplication[a,b] is a*b,
acting as a after b. A walk with voltages g1,...,gk therefore reaches h*g1*...*gk.
This representation also supports nonfree vertex actions and ineffective abstract actions.
"""

    def __init__(self, adjacency, voltages, action, multiplication, identity,
                 representatives, orbit_index, outside=None, upstairs_outside=None):
        self.adjacency = csr_matrix(adjacency, dtype=bool)
        self.voltages = csr_matrix(voltages, dtype=np.uint64)
        self.action = np.asarray(action, dtype=np.int64)
        self.multiplication = np.asarray(multiplication, dtype=np.int64)
        self.identity = int(identity)
        self.representatives = np.asarray(representatives, dtype=np.int64)
        self.orbit_index = np.asarray(orbit_index, dtype=np.int64)
        self.outside, self.upstairs_outside = outside, upstairs_outside
        self.group_size = len(self.action)
        self.stabilizers = tuple(tuple(np.flatnonzero(
            self.action[:, rep] == rep).tolist()) for rep in self.representatives)

    def edge_voltages(self, source, target):
        mask = int(self.voltages[source, target])
        return tuple(g for g in range(self.group_size) if mask & (1 << g))

    def morse_decomposition(self):
        return morse_decomposition(self.adjacency, self.outside)

    def lift_relation(self, max_edges=20_000_000):
        """Recover the whole relation from voltage edges, without an upstairs map evaluation."""
        rows, cols = [], []
        for qi, rep in enumerate(self.representatives):
            start, end = self.voltages.indptr[qi:qi + 2]
            for qj, mask in zip(self.voltages.indices[start:end], self.voltages.data[start:end]):
                target = self.representatives[qj]
                for g in range(self.group_size):
                    if int(mask) & (1 << g):
                        for h in range(self.group_size):
                            rows.append(int(self.action[h, rep]))
                            cols.append(int(self.action[self.multiplication[h, g], target]))
                if len(rows) > max_edges:
                    raise MemoryError("lifted relation exceeds the edge budget")
        n = self.action.shape[1]
        relation = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
        relation.sum_duplicates()
        return relation

    def lift_morse_graph(self, max_edges=20_000_000):
        """Compute component stabilizers and all comparable-pair voltages, then the coset order.

The auxiliary lifted relation is reconstructed solely from the quotient voltage data. Every
comparable quotient pair is inspected, including those absent from the quotient Hasse diagram.
"""
        relation = self.lift_relation(max_edges)
        quotient = self.morse_decomposition()
        lifted = morse_decomposition(relation, self.upstairs_outside)
        membership = np.full(relation.shape[0], -1, dtype=np.int64)
        for i, c in enumerate(lifted.components):
            membership[c] = i
        bases, base_components, stabilizers = [], [], []
        for qc in quotient.components:
            base = int(self.representatives[int(qc[0])])
            ci = int(membership[base])
            if ci < 0:
                raise AssertionError("recurrent quotient component did not lift recurrently")
            stab = tuple(g for g in range(self.group_size)
                         if membership[self.action[g, base]] == ci)
            bases.append(base)
            base_components.append(ci)
            stabilizers.append(stab)
        component_cosets = []
        for q, base in enumerate(bases):
            found = {}
            for h in range(self.group_size):
                found.setdefault(int(membership[self.action[h, base]]), h)
            if len(found) * len(stabilizers[q]) != self.group_size:
                raise AssertionError("component stabilizer and coset count disagree")
            component_cosets.extend((ci, q, h) for ci, h in found.items())
        walks = {}
        for source, target in quotient.order:
            reachable = set(breadth_first_order(relation, bases[source], directed=True,
                                                return_predecessors=False).tolist())
            walks[source, target] = tuple(g for g in range(self.group_size)
                                          if int(self.action[g, bases[target]]) in reachable)
        inverse = [int(np.flatnonzero(row == self.identity)[0]) for row in self.multiplication]
        reconstructed_order = set()
        for source_ci, source_q, k in component_cosets:
            for target_ci, target_q, h in component_cosets:
                if (source_q, target_q) in walks:
                    displacement = int(self.multiplication[inverse[k], h])
                    if displacement in walks[source_q, target_q]:
                        reconstructed_order.add((source_ci, target_ci))
        if reconstructed_order != set(lifted.order):
            raise AssertionError("voltage/coset order disagrees with the reconstructed relation")
        return dict(quotient=quotient, lifted=lifted, bases=tuple(bases),
                    base_components=tuple(base_components), stabilizers=tuple(stabilizers),
                    component_cosets=tuple(sorted(component_cosets)), walk_voltages=walks)


def _voltage_csr(rows, cols, elements, shape):
    if not rows:
        return csr_matrix(shape, dtype=np.uint64)
    key = np.asarray(rows, dtype=np.int64) * shape[1] + np.asarray(cols, dtype=np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    bits = np.left_shift(np.uint64(1), np.asarray(elements, dtype=np.uint64)[order])
    starts = np.r_[0, 1 + np.flatnonzero(key[1:] != key[:-1])]
    masks = np.bitwise_or.reduceat(bits, starts)
    keys = key[starts]
    return csr_matrix((masks, (keys // shape[1], keys % shape[1])), shape=shape)


def quotient_from_relation(adjacency, action, multiplication, identity=0, outside=None,
                           check=True):
    """Form voltages of a finite equivariant digraph, including nonfree actions.

This general route is intended for independent small-graph checks. Large grids use quotient_graph.
"""
    a = csr_matrix(adjacency, dtype=bool)
    action, multiplication = np.asarray(action, dtype=np.int64), np.asarray(multiplication, dtype=np.int64)
    n, ng = a.shape[0], len(action)
    if a.shape != (n, n) or action.shape != (ng, n) or not 1 <= ng <= 64:
        raise ValueError("invalid relation/action or more than 64 group elements")
    if multiplication.shape != (ng, ng) or np.any(multiplication < 0) or np.any(multiplication >= ng):
        raise ValueError("invalid multiplication table")
    if not 0 <= identity < ng or not np.array_equal(action[identity], np.arange(n)):
        raise ValueError("invalid identity action")
    if check:
        for g in range(ng):
            if not np.array_equal(np.sort(action[g]), np.arange(n)):
                raise ValueError("group action is not a permutation")
            if (a[action[g]][:, action[g]] != a).nnz:
                raise ValueError("relation is not equivariant")
            for h in range(ng):
                if not np.array_equal(action[multiplication[g, h]], action[g, action[h]]):
                    raise ValueError("multiplication and action disagree")
        if not np.array_equal(multiplication[identity], np.arange(ng)) or not np.array_equal(
                multiplication[:, identity], np.arange(ng)):
            raise ValueError("invalid group identity")
        for g in range(ng):
            if not np.any((multiplication[g] == identity) & (multiplication[:, g] == identity)):
                raise ValueError("group element has no inverse")
            for h in range(ng):
                if not np.array_equal(multiplication[multiplication[g, h]],
                                      multiplication[g, multiplication[h]]):
                    raise ValueError("multiplication is not associative")
    canonical = action.min(axis=0)
    reps, orbit_index = np.unique(canonical, return_inverse=True)
    qoutside = None
    if outside is not None:
        if any(action[:, outside] != outside) or a.indptr[outside] != a.indptr[outside + 1]:
            raise ValueError("outside must be fixed and have no outgoing edges")
        qoutside = int(orbit_index[outside])
    rows, cols, elements = [], [], []
    for qi, rep in enumerate(reps):
        for target in a.indices[a.indptr[rep]:a.indptr[rep + 1]]:
            qj = int(orbit_index[target])
            for g in np.flatnonzero(action[:, reps[qj]] == target):
                rows.append(qi)
                cols.append(qj)
                elements.append(int(g))
    voltage = _voltage_csr(rows, cols, elements, (len(reps), len(reps)))
    return QuotientGraph(voltage.astype(bool), voltage, action, multiplication, identity,
                         reps, orbit_index, qoutside, outside)


def quotient_graph(outer):
    """Quotient and voltages directly from representative boxes, without full map evaluation."""
    grid, orbits = outer.equivariant_grid, outer.orbits
    ng, n = len(grid.group), len(orbits.cell_ids)
    if ng > 64:
        raise ValueError("voltage masks support at most 64 group elements")
    nr = len(orbits.representatives)
    action = np.stack([np.r_[outer.positions(grid.act_ids(g, orbits.cell_ids)), n]
                       for g in range(ng)])
    reps = np.r_[outer.positions(orbits.representatives), n]
    orbit_index = np.r_[orbits.orbit_index, nr]
    rows, cols, elements = [], [], []
    for qi, row in enumerate(reps[:-1]):
        ids, escaped = outer.image_ids(int(row))
        for target in outer.positions(ids):
            qj = int(orbits.orbit_index[target])
            for g in np.flatnonzero(action[:, reps[qj]] == target):
                rows.append(qi)
                cols.append(qj)
                elements.append(int(g))
        if escaped:
            for g in range(ng):
                rows.append(qi)
                cols.append(nr)
                elements.append(g)
    voltage = _voltage_csr(rows, cols, elements, (nr + 1, nr + 1))
    return QuotientGraph(voltage.astype(bool), voltage, action, grid.multiplication,
                         grid.identity, reps, orbit_index, nr, n)


def omega_limit(adjacency, vertices, outside=None):
    """Finite-graph omega limit: descendants of reachable recurrent components."""
    a = csr_matrix(adjacency)
    reachable = set(forward_closure(a, vertices).tolist())
    recurrent = morse_decomposition(a, outside)
    seeds = [int(v) for c in recurrent.components if int(c[0]) in reachable for v in c]
    return frozenset(forward_closure(a, seeds).tolist())


def attractor_lattice(adjacency, outside=None, max_elements=65536):
    """Finite attractors generated by recurrent-component forward closures.

Returns every attractor, including the empty one. An attractor containing outside is explicitly
flagged and cannot be used as a no-escape block. Set intersection is not the lattice meet:
use omega_limit(adjacency, A & B, outside). Exponential growth is stopped at max_elements.
"""
    a = csr_matrix(adjacency)
    recurrent = morse_decomposition(a, outside)
    generators = [frozenset(forward_closure(a, c).tolist()) for c in recurrent.components]
    elements = {frozenset()}
    for generator in generators:
        new = {element | generator for element in elements}
        elements |= new
        if len(elements) > max_elements:
            raise MemoryError("attractor lattice exceeds the configured element budget")
    ordered = tuple(sorted(elements, key=lambda s: (len(s), tuple(sorted(s)))))
    return dict(elements=ordered, generators=tuple(generators),
                escaped=tuple(outside is not None and outside in s for s in ordered))
