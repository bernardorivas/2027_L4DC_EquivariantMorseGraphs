"""Geometric intersections, Euclidean distances, and residual-domain regressions."""
import numpy as np
import pytest

from latent_equivariance.certificates import (
    CubicalBlock, generated_block_family, principal_block_family, box_complement_distance,
    tolerance_2d, sampled_tolerance_upper, sample_residuals, residual_values,
    representative_indices, block_representative_indices, separate_attractor_blocks,
)
from latent_equivariance.equivariant_grid import EquivariantGrid, build_outer_approximation


def plain_grid(shape=(3, 3), lower=(0., 0.), upper=(3., 3.)):
    return EquivariantGrid.diagonal_signs(shape, lower, upper, [])


def block_from_indices(grid, indices):
    return CubicalBlock.from_top_ids(grid, grid.ids(np.asarray(indices)))


def test_geometric_lattice_retains_shared_edge_and_vertex():
    grid = plain_grid((2, 2), (0, 0), (2, 2))
    a = block_from_indices(grid, [(0, 0)])
    b = block_from_indices(grid, [(1, 0)])
    face = a.intersection(b)
    assert not face.empty_set and not face.pure and len(face.top_ids) == 0
    assert face.cells.sum() == 3
    assert face.contains([[1, .5], [1, 0], [1, 1], [np.nextafter(1., 0.), .5]]).tolist() == [True, True, True, False]
    vertex = a.intersection(block_from_indices(grid, [(1, 1)]))
    assert vertex.cells.sum() == 1 and vertex.contains([[1, 1]])[0]
    family = generated_block_family([a, b])
    assert len(family.blocks) == 5 and len(family.join_irreducibles) == 3
    indices = {x.fingerprint: i for i, x in enumerate(family.blocks)}
    assert family.predecessors[indices[a.fingerprint]] == indices[face.fingerprint]
    assert family.predecessors[indices[face.fingerprint]] == 0
    assert family.manifest()['hash'] == family.fingerprint
    assert generated_block_family([a, b]).fingerprint == family.fingerprint
    with pytest.raises(MemoryError):
        generated_block_family([a, b], max_elements=4)


def test_closed_cell_membership_does_not_clip_or_fatten_faces():
    grid = plain_grid((2, 2), (-.7, -.3), (.9, 1.1))
    b = block_from_indices(grid, [(0, 0)])
    lo, hi = grid.cell_bounds([0])
    x, y = hi[0]
    points = [[x, y], [np.nextafter(x, np.inf), y], [x, np.nextafter(y, np.inf)],
              [grid.lower[0]-1, grid.lower[1]], [np.nan, 0]]
    assert b.contains(points).tolist() == [True, False, False, False, False]
    invalid = b.cells.copy()
    invalid[0, 0] = False
    with pytest.raises(ValueError, match='closed under faces'):
        CubicalBlock(grid, invalid)
    with pytest.raises(ValueError):
        b.cells[0, 0] = False


def test_principal_downsets_generate_frozen_chain_and_predecessors():
    grid = plain_grid((3, 1), (0, 0), (3, 1))
    def enclosure(lo, hi):
        i = lo[:, 0].astype(int)
        return np.column_stack([np.maximum(i-1, 0)+.25, np.full(len(i), .25)]), np.column_stack([i+.75, np.full(len(i), .75)])
    outer = build_outer_approximation(grid, enclosure, padding=1e-8)
    family = principal_block_family(outer)
    assert [len(x.top_ids) for x in family.blocks] == [0, 1, 2, 3]
    assert family.join_irreducibles == (1, 2, 3)
    assert family.predecessors == {1: 0, 2: 1, 3: 2}
    assert [x['downset'] for x in family.candidates] == [[0], [0, 1], [0, 1, 2]]
    assert not any(x['escaped'] for x in family.candidates)
    for block in family.blocks[1:]:
        assert tolerance_2d(outer, block)['lower'] == pytest.approx(.25-1e-8)
    manifest = family.manifest()
    family.predecessors[3] = 0
    with pytest.raises(ValueError, match='frozen block family changed'):
        family.manifest()
    assert manifest['predecessors']['3'] == 2


def test_escaped_required_candidate_is_retained():
    grid = EquivariantGrid.diagonal_signs((2, 2), (-1, -1), (1, 1), [(-1, 1), (1, -1)])
    outer = build_outer_approximation(grid, lambda lo, hi: (2*lo, 2*hi), padding=1e-8)
    family = principal_block_family(outer)
    assert len(family.candidates) == 1 and family.candidates[0]['escaped']
    assert family.blocks[-1].top_ids.size == 4
    report = tolerance_2d(outer, family.blocks[-1])
    assert report['escaped'] and not report['positive'] and report['lower'] == 0


def test_euclidean_corner_distance_and_boxes_crossing_a_hole():
    grid = plain_grid((5, 5), (0, 0), (5, 5))
    b = block_from_indices(grid, [(i, j) for i in range(5) for j in range(5) if i < 2 or j < 2])
    assert box_complement_distance(b, [[1.4, 1.4]], [[1.6, 1.6]]) == pytest.approx(np.sqrt(.4**2+.4**2))
    assert box_complement_distance(b, [[3, 3]], [[3.1, 3.1]]) == 0
    hole = block_from_indices(grid, [(i, j) for i in range(5) for j in range(5) if (i, j) != (2, 2)])
    assert hole.contains([[1.5, 1.5], [3.5, 3.5]]).all()
    assert box_complement_distance(hole, [[1.5, 1.5]], [[3.5, 3.5]]) == 0
    # Lower-dimensional additions cannot supply a positive interior distance.
    isolated = b.intersection(block_from_indices(grid, [(2, 2)]))
    assert not isolated.pure and box_complement_distance(isolated, [[2, 2]], [[2, 2]]) == 0


def test_anisotropic_rectangle_distance_with_lossless_pruning():
    grid = plain_grid((20, 10), (-5, -2), (5, 2))
    block = CubicalBlock.from_top_ids(grid, np.arange(grid.size))
    rng = np.random.default_rng(20261106)
    centers = rng.uniform([-4, -1.5], [4, 1.5], size=(500, 2))
    half = rng.uniform(0, .1, size=(500, 2))
    lo, hi = centers-half, centers+half
    expected = np.min(np.column_stack([lo[:, 0]+5, 5-hi[:, 0], lo[:, 1]+2, 2-hi[:, 1]]))
    assert box_complement_distance(block, lo, hi) == pytest.approx(expected)
    assert sampled_tolerance_upper(block, centers, centers)['upper'] == pytest.approx(np.min(np.column_stack([centers[:, 0]+5, 5-centers[:, 0], centers[:, 1]+2, 2-centers[:, 1]])))
    with pytest.raises(ValueError, match='sources'):
        sampled_tolerance_upper(block, [[100, 0]], [[0, 0]])


def test_nonpure_tolerance_fails_without_erasing_shared_faces():
    grid = plain_grid((2, 1), (0, 0), (2, 1))
    outer = build_outer_approximation(grid, lambda lo, hi: (lo*.1+.4, hi*.1+.4), padding=1e-8)
    face = block_from_indices(grid, [(0, 0)]).intersection(block_from_indices(grid, [(1, 0)]))
    result = tolerance_2d(outer, face)
    assert result['lower'] == 0 and not result['positive']
    assert 'lower-dimensional' in result['reason']
    assert face.cells.sum() == 3


def test_residual_domains_stabilizers_and_empty_samples():
    grid = EquivariantGrid.diagonal_signs((2, 2), (-1, -1), (1, 1), [(-1, 1), (1, -1)])
    right = block_from_indices(grid, [(1, 0), (1, 1)])
    whole = CubicalBlock.from_top_ids(grid, np.arange(4))
    z = np.array([[.2, .2], [-.2, .2], [2., 0], [.3, -.2]])
    truth = np.array([[.1, .2], [.1, .2], [100., 100.], [.1, .2]])
    pred = np.array([[-.1, .2], [40., 0.], [0., 0.], [10., 10.]])
    domain = np.array([True, True, True, False])
    report = sample_residuals(right, z, pred, truth, domain_mask=domain, tolerance_lower=.1)
    assert report['selected_indices'].tolist() == [0]
    assert report['maxima'] == pytest.approx(dict(plain=.2, modulo_group=0, stabilizer=.2))
    assert not report['eligible']['modulo_group'] and not any(report['passed'].values())
    with pytest.raises(ValueError, match='preserve the block'):
        sample_residuals(right, z, pred, truth, domain_mask=domain, subgroup=range(len(grid.group)))
    full = sample_residuals(whole, z[:1], pred[:1], truth[:1], domain_mask=np.array([True]), tolerance_lower=.1)
    assert full['passed']['modulo_group'] and not full['passed']['plain']
    absent = sample_residuals(right, z, pred, truth, domain_mask=np.zeros(4, bool), tolerance_lower=1.)
    assert absent['count'] == 0 and absent['maxima']['plain'] is None and not absent['passed']['plain']


def test_full_state_orbits_preserve_fibers_and_actual_block_domain():
    grid = EquivariantGrid.diagonal_signs((2, 2), (-1, -1), (1, 1), [(-1, 1), (1, -1)])
    right = block_from_indices(grid, [(1, 0), (1, 1)])
    states = np.array([[.2, .3, 1.], [.2, -.3, 1.], [.2, .3, 2.], [-.2, .3, 1.]])
    def action(g, points):
        return points*np.array([*grid.group.elements[g][1], 1])
    chosen = block_representative_indices(right, states, states[:, :2], action, domain_mask=np.ones(4, bool))
    assert chosen.tolist() == [0, 2]
    assert representative_indices(states, action, range(len(grid.group))).tolist() == [0, 2]
    with pytest.raises(ValueError, match='does not preserve'):
        block_representative_indices(right, states, states[:, :2], action, domain_mask=np.ones(4, bool), subgroup=range(len(grid.group)))


def test_invalid_subgroup_and_nonfinite_domain_fail():
    grid = EquivariantGrid.diagonal_signs((2, 2), (-1, -1), (1, 1), [(-1, 1), (1, -1)])
    with pytest.raises(ValueError, match='subgroup'):
        residual_values(grid, [[0, 0]], [[0, 0]], [1])
    block = CubicalBlock.from_top_ids(grid, np.arange(4))
    with pytest.raises(ValueError, match='finite'):
        sample_residuals(block, [[np.nan, 0]], [[0, 0]], [[0, 0]], domain_mask=np.array([True]))


def test_separate_attractors_use_actual_block_stabilizers():
    grid = EquivariantGrid.diagonal_signs((4, 2), (-1, -1), (1, 1), [(-1, 1), (1, -1)])
    def enclosure(lo, hi):
        return np.column_stack([.8*np.tanh(2*lo[:, 0]), np.zeros(len(lo))]), np.column_stack([.8*np.tanh(2*hi[:, 0]), np.zeros(len(hi))])
    outer = build_outer_approximation(grid, enclosure, padding=1e-8)
    candidates = separate_attractor_blocks(outer)
    assert len(candidates) == 2
    for candidate in candidates:
        assert not candidate['escaped'] and len(candidate['stabilizer']) == 2
        assert len(candidate['block'].top_ids) == 2
        assert tolerance_2d(outer, candidate['block'])['positive']
    family = principal_block_family(outer)
    assert all(len(b.stabilizer) == 4 for b in family.blocks)
    assert len(family.blocks) == 3


def test_block_hashes_and_families_bind_the_group_action():
    trivial = plain_grid((2, 2), (-1, -1), (1, 1))
    reflected = EquivariantGrid.diagonal_signs((2, 2), (-1, -1), (1, 1), [(-1, 1)])
    a = CubicalBlock.from_top_ids(trivial, [0])
    b = CubicalBlock.from_top_ids(reflected, [0])
    assert np.array_equal(a.cells, b.cells) and a.fingerprint != b.fingerprint
    with pytest.raises(ValueError, match='group actions'):
        generated_block_family([a, b])
    outer = build_outer_approximation(trivial, lambda lo, hi: (.2*lo, .2*hi), padding=1e-8)
    # A supplied stale quotient is not accepted by the public API.
    with pytest.raises(TypeError):
        principal_block_family(outer, object())
    with pytest.raises(TypeError):
        separate_attractor_blocks(outer, object())
