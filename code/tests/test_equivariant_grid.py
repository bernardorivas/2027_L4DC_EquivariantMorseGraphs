"""Independent cell enumeration and admitted-action checks."""
from itertools import product

import numpy as np
import pytest

from latent_equivariance.equivariant_grid import EquivariantGrid, build_outer_approximation
from latent_equivariance.equivariant_graph import index_pair, quotient_graph
from latent_equivariance.homology import BoxMap, conley_index, index_pair as old_index_pair


def test_sign_grid_and_orbits():
    grid = EquivariantGrid.diagonal_signs((8, 6), (-2., -1.), (2., 1.), [(-1, -1), (1, -1)])
    orbits = grid.orbits()
    assert len(orbits.representatives) == 12
    assert all(stab == (grid.identity,) for stab in orbits.stabilizers)
    for c in grid.grid.cells_of(grid.grid.tops()):
        grid.group.orbit(c)
    for h in range(len(grid.group)):
        ids = np.arange(grid.size)
        expected = [grid.group.act_top(grid.group.elements[h], tuple(i)) for i in grid.indices(ids)]
        assert np.array_equal(grid.act_indices(h, grid.indices(ids)), expected)
    for args in [((3,), (-1.,), (1.,)), ((4,), (-1.,), (2.,))]:
        with pytest.raises(ValueError):
            EquivariantGrid.diagonal_signs(*args, [(-1,)])
    with pytest.raises(ValueError):
        EquivariantGrid((4, 4), (-1., -1.), (1., 1.), [((1, 0), (1, 1), (0, 0))])


def test_klein_grid_condition_and_closed_cells():
    for shape in [(2, 8), (4, 1), (8, 30), (12, 48)]:
        grid = EquivariantGrid.klein(shape)
        assert grid.closed_cell_injective == (shape[0] >= 4)
        for c in grid.grid.cells_of(grid.grid.tops()):
            grid.group.orbit(c)
        assert len(grid.orbits().representatives) == grid.size // 2
    with pytest.raises(ValueError):
        EquivariantGrid.klein((3, 8))


def test_representative_boxes_agree_with_full_map():
    grid = EquivariantGrid.diagonal_signs((8, 6), (-2., -1.), (2., 1.), [(-1, -1), (1, -1)])
    factors = np.asarray([.37, .38])
    outer = build_outer_approximation(grid, lambda lo, hi: (lo * factors, hi * factors), padding=1e-9)
    direct = BoxMap.from_function(grid.grid, grid.lower, grid.upper,
        lambda box: list(np.asarray(box[:2]) * factors - 1e-9) +
                    list(np.asarray(box[2:]) * factors + 1e-9))
    a = outer.adjacency()
    assert outer.evaluated == 12
    for row, idx in enumerate(grid.indices(outer.cell_ids)):
        expected = {int(grid.ids(np.asarray(t))) for t in direct(tuple(idx))}
        assert set(a.indices[a.indptr[row]:a.indptr[row + 1]]) == expected
    q = quotient_graph(outer)
    assert (q.lift_relation() != a).nnz == 0
    report = q.lift_morse_graph()
    assert len(report["lifted"].components) == len(report["quotient"].components) == 1
    assert report["stabilizers"] == (tuple(range(4)),)


def test_active_domain_escape_and_invalid_bounds():
    grid = EquivariantGrid.diagonal_signs((8,), (-1.,), (1.,), [(-1,)])
    outer = build_outer_approximation(grid, lambda lo, hi: (lo * .1, hi * .1),
                                      active_ids=[0, 7], padding=1e-9)
    assert outer.adjacency()[0, outer.outside]
    assert outer.adjacency()[1, outer.outside]
    with pytest.raises(ValueError, match="outside image"):
        outer.local_box_map()
    with pytest.raises(ValueError, match="invariant"):
        grid.orbits([0])
    with pytest.raises(ValueError, match="empty interval"):
        build_outer_approximation(grid, lambda lo, hi: (hi, lo))
    with pytest.raises(ValueError, match="invalid bounds"):
        build_outer_approximation(grid, lambda lo, hi: (lo * np.nan, hi))


def test_pair_quotient_lift_and_local_homology():
    grid = EquivariantGrid.diagonal_signs((16,), (-1.,), (1.,), [(-1,)])
    # The odd increasing map has two attracting fixed points and a repelling zero.
    f = lambda x: x + .35 * x * (.49 - x * x)
    outer = build_outer_approximation(grid, lambda lo, hi: (f(lo), f(hi)), padding=1e-10)
    q = quotient_graph(outer)
    a = outer.adjacency()
    bm = outer.local_box_map()
    for component in q.morse_decomposition().components:
        lifted = set(np.flatnonzero(np.isin(q.orbit_index, component)).tolist())
        up = index_pair(a, lifted, outer.outside)
        down = index_pair(q.adjacency, component, q.outside)
        for key in ("S", "P1", "P0", "T1", "T0"):
            assert {int(q.orbit_index[i]) for i in up[key]} == down[key]
        assert up["isolated"] == down["isolated"]
        old = old_index_pair(bm, {tuple(x) for x in grid.indices(list(lifted))})
        assert up["isolated"] == old["isolated"]
        if up["isolated"] and not up["escaped"]:
            result = conley_index(grid.grid, bm, old["S"], ring="Z")
            assert all(m is not None for m in result["index_map"].values())
