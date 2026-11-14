"""Quotient metrics, closures on a cell relation, tolerances, attracting regions and residuals."""
import numpy as np
import pytest

from morsegraphs import grids, regions


def interval(count):
    return grids.cells(count, [-1.], [1.])


def test_euclidean_distance_between_boxes():
    metric = regions.Euclidean()
    a = (np.array([0., 0.]), np.array([1., 1.]))
    b = (np.array([4., 5.]), np.array([6., 6.]))
    assert metric.lower(*a, *b) == pytest.approx(5.0)
    assert metric.lower(*a, np.array([.5, -2.]), np.array([.6, 9.])) == 0
    assert metric.point([0., 0.], [3., 4.]) == pytest.approx(5.0)


def test_sphere_height_is_the_distance_between_circles_of_latitude():
    metric = regions.SphereHeight()
    rng = np.random.default_rng(1)
    z, w = rng.uniform(-1, 1, (2, 200))
    angle = rng.uniform(0, 2 * np.pi, (64, 1))
    p = np.stack((np.sqrt(1 - z**2), np.zeros_like(z), z), axis=-1)
    q = np.stack((np.sqrt(1 - w**2) * np.cos(angle), np.sqrt(1 - w**2) * np.sin(angle),
                  np.broadcast_to(w, (64, 200))), axis=-1)
    exact = metric.point(z[:, None], w[:, None])
    assert np.all(exact <= np.linalg.norm(p[None] - q, axis=-1).min(axis=0) + 1e-12)
    assert metric.point([[1.]], [[-1.]]) == pytest.approx(2.0)
    a, b = (np.array([-.2]), np.array([.1])), (np.array([.5]), np.array([.9]))
    pairs = metric.point(np.linspace(-.2, .1, 301)[:, None, None], np.linspace(.5, .9, 301)[None, :, None])
    assert metric.lower(*a, *b) == pytest.approx(pairs.min())
    with pytest.raises(ValueError):
        metric.point([[1.2]], [[.5]])


def test_closures():
    # 0 -> 1 -> 2 -> 2 and 3 -> 3, 3 -> 1, 4 -> 3
    offsets, targets = np.array([0, 1, 2, 3, 5, 6]), np.array([1, 2, 2, 3, 1, 3])
    assert regions.forward_closure(offsets, targets, [3]).tolist() == [False, True, True, True, False]
    back_offsets, back_targets = regions.reverse(offsets, targets)
    assert sorted(back_targets[back_offsets[1]:back_offsets[2]]) == [0, 3]
    assert regions.is_forward_invariant(offsets, targets, [1, 2])
    assert not regions.is_forward_invariant(offsets, targets, [0, 2])


@pytest.mark.parametrize("seed", range(10))
def test_closures_against_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    count = int(rng.integers(2, 11))
    edges = rng.random((count, count)) < rng.uniform(.05, .35)
    offsets = np.concatenate(([0], np.cumsum(edges.sum(axis=1))))
    targets = np.concatenate([np.flatnonzero(row) for row in edges]).astype(np.int64)
    reach = edges | np.eye(count, dtype=bool)
    for _ in range(count):
        reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)
    seeds = rng.random(count) < .4
    assert np.array_equal(regions.forward_closure(offsets, targets, seeds), reach[seeds].any(axis=0))


def test_tolerance_of_a_contraction_on_an_interval():
    boxes = interval(8)
    images = boxes / 2
    metric = regions.Euclidean()
    faces = np.array([[True, True]])
    assert regions.tolerance(boxes, images, [2, 3, 4, 5], metric, faces)["value"] == pytest.approx(.25)
    assert regions.tolerance(boxes, images, np.ones(8, bool), metric, faces)["value"] == float("inf")
    window = regions.tolerance(boxes, images, np.ones(8, bool), metric, np.array([[False, False]]))
    assert window["value"] == pytest.approx(.5)
    assert regions.tolerance(boxes, images, [4, 5, 6, 7], metric, faces)["value"] == 0
    assert regions.tolerance(boxes, images, [], metric, faces)["value"] == float("inf")


def brute_tolerance(boxes, mask, images, faces):
    k = boxes.shape[1] // 2
    lower, upper = boxes[:, :k], boxes[:, k:]
    domain_lower, domain_upper = lower.min(axis=0), upper.max(axis=0)
    best = np.inf
    for image in images[mask]:
        low = np.where(faces[:, 0], np.maximum(image[:k], domain_lower), image[:k])
        high = np.where(faces[:, 1], np.minimum(image[k:], domain_upper), image[k:])
        for other_low, other_high in zip(lower[~mask], upper[~mask]):
            gap = np.maximum(0, np.maximum(other_low - high, low - other_high))
            best = min(best, float(np.linalg.norm(gap)))
        for axis in range(k):
            if not faces[axis, 0]:
                best = min(best, max(0.0, float(low[axis] - domain_lower[axis])))
            if not faces[axis, 1]:
                best = min(best, max(0.0, float(domain_upper[axis] - high[axis])))
    return best


@pytest.mark.parametrize("seed", range(20))
def test_tolerance_against_brute_force_in_the_plane(seed):
    rng = np.random.default_rng(seed)
    boxes = grids.cells(4, [0., 0.], [1., 1.])
    mask = rng.random(len(boxes)) < .5
    mask[0] = True
    center = rng.uniform(.1, .9, (len(boxes), 2))
    radius = rng.uniform(.01, .2, (len(boxes), 2))
    images = np.column_stack((center - radius, center + radius))
    faces = rng.random((2, 2)) < .5
    result = regions.tolerance(boxes, images, mask, regions.Euclidean(), faces, chunk=int(rng.integers(1, 5)))
    assert result["value"] == pytest.approx(brute_tolerance(boxes, mask, images, faces), abs=1e-14)


def test_window_exit_removes_a_region():
    boxes = interval(8)
    images = np.tile([-1.2, -.85], (len(boxes), 1))
    graph = grids.morse_graph(boxes, np.clip(images, -2, 2))
    assert len(graph.minimal) == 1
    inside = regions.attracting_regions(graph, np.array([[True, True]]))
    assert np.all(inside >= 0)
    window = regions.attracting_regions(graph, np.array([[False, False]]))
    assert np.all(window < 0)


def test_unavailable_cell_and_its_predecessors_are_removed():
    boxes = interval(8)
    images = np.full_like(boxes, -.875)
    images[1] = .125
    available = np.ones(8, bool)
    available[4] = False
    graph = grids.morse_graph(boxes, images)
    labels = regions.attracting_regions(graph, np.array([[False, False]]), available)
    assert set(np.flatnonzero(labels < 0)) == {1, 4}
    rows, _ = regions.compare(graph, labels, [[-.875]], [[-.875]], [[-.8]], regions.Euclidean(),
                              np.array([[False, False]]))
    assert rows[0]["tolerance"] == pytest.approx(.125)
    assert rows[0]["passed"] and rows[0]["cells"] == 6


def test_residual_on_the_closed_region_includes_boundary_samples():
    boxes = interval(4)
    images = np.full_like(boxes, -.75)
    graph = grids.morse_graph(boxes, images)
    labels = regions.attracting_regions(graph, np.array([[True, True]]))
    assert np.all(labels == graph.minimal[0])
    rows, residual = regions.compare(graph, labels, [[0.], [.25]], [[-.1], [-.25]], [[.1], [-.24]],
                                     regions.Euclidean(), np.array([[True, True]]))
    assert residual == pytest.approx([.2, .01])
    assert rows[0]["samples"] == 2 and rows[0]["max_residual"] == pytest.approx(.2)
    assert rows[0]["tolerance"] == float("inf")
