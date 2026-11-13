"""Quotient metrics, closures on a cell relation, tolerance bounds and block membership."""
import numpy as np
import pytest

from latent_equivariance.quotient_certificates import (
    CellIndex, Euclidean, SphereHeight, compare, forward_closure, is_forward_invariant,
    region_of_attraction, residual_modulo_group, residual_upper, reverse, tolerance_lower,
)


def interval_cells(count, lower=-1.0, upper=1.0):
    vertices = np.linspace(lower, upper, count + 1)
    return vertices[:-1, None], vertices[1:, None]


def relation(lower, upper, image_lower, image_upper):
    """Each cell is related to the closed cells that meet its image box."""
    offsets, targets = [0], []
    for low, high in zip(image_lower, image_upper):
        hit = np.flatnonzero(np.all((upper >= low) & (lower <= high), axis=1))
        targets.extend(hit)
        offsets.append(len(targets))
    return np.array(offsets), np.array(targets)


def test_euclidean_distances_between_boxes():
    metric = Euclidean()
    a = (np.array([0., 0.]), np.array([1., 1.]))
    b = (np.array([4., 5.]), np.array([6., 6.]))
    assert metric.lower(*a, *b) == pytest.approx(5.0)  # gaps 3 and 4
    assert metric.upper(*a, *b) == pytest.approx(np.hypot(6, 6))
    assert metric.lower(*a, np.array([.5, -2.]), np.array([.6, 9.])) == 0
    assert metric.point([0., 0.], [3., 4.]) == pytest.approx(5.0)
    rng = np.random.default_rng(0)
    x = rng.uniform(a[0], a[1], (2000, 2))
    y = rng.uniform(b[0], b[1], (2000, 2))
    sampled = np.linalg.norm(x[:, None] - y[None], axis=-1)
    assert metric.lower(*a, *b) <= sampled.min() and sampled.max() <= metric.upper(*a, *b)


def test_sphere_height_is_the_distance_between_circles_of_latitude():
    metric = SphereHeight()
    rng = np.random.default_rng(1)
    z, w = rng.uniform(-1, 1, (2, 200))
    angle = rng.uniform(0, 2 * np.pi, (64, 1))
    p = np.stack((np.sqrt(1 - z**2), np.zeros_like(z), z), axis=-1)
    q = np.stack((np.sqrt(1 - w**2) * np.cos(angle), np.sqrt(1 - w**2) * np.sin(angle),
                  np.broadcast_to(w, (64, 200))), axis=-1)
    over_rotations = np.linalg.norm(p[None] - q, axis=-1).min(axis=0)
    # The rotation by angle 0 is among the samples only approximately.
    exact = metric.point(z[:, None], w[:, None])
    assert np.all(exact <= over_rotations + 1e-12)
    assert np.allclose(exact, np.hypot(z - w, np.sqrt(1 - z**2) - np.sqrt(1 - w**2)))
    assert metric.point([[1.]], [[-1.]]) == pytest.approx(2.0)
    assert np.all(exact >= np.abs(z - w) - 1e-15)


def test_sphere_height_distances_between_intervals():
    metric = SphereHeight()
    a = (np.array([-.2]), np.array([.1]))
    b = (np.array([.5]), np.array([.9]))
    grid_a, grid_b = np.linspace(*a[0], *a[1], 301), np.linspace(*b[0], *b[1], 301)
    pairs = metric.point(grid_a[:, None, None], grid_b[None, :, None])
    assert metric.lower(*a, *b) == pytest.approx(pairs.min())
    assert metric.upper(*a, *b) == pytest.approx(pairs.max())
    assert metric.lower(*b, *a) == pytest.approx(pairs.min())
    assert metric.lower(*a, np.array([0.]), np.array([.3])) == 0


def test_closures_and_region_of_attraction():
    # 0 -> 1 -> 2 -> 2 and 3 -> 3, 3 -> 1, 4 -> 3: Morse sets {2} and {3}.
    offsets = np.array([0, 1, 2, 3, 5, 6])
    targets = np.array([1, 2, 2, 3, 1, 3])
    assert forward_closure(offsets, targets, [3]).tolist() == [False, True, True, True, False]
    assert forward_closure(offsets, targets, [2]).tolist() == [False, False, True, False, False]
    back_offsets, back_targets = reverse(offsets, targets)
    assert sorted(back_targets[back_offsets[1]:back_offsets[2]]) == [0, 3]
    assert is_forward_invariant(offsets, targets, [1, 2])
    assert not is_forward_invariant(offsets, targets, [0, 2])
    region = region_of_attraction(offsets, targets, [2], [[2], [3]])
    assert region.tolist() == [True, True, True, False, False]
    assert region_of_attraction(offsets, targets, [1, 2, 3], [[2], [3]]).all()
    with pytest.raises(ValueError):
        region_of_attraction(offsets, targets, [0], [[2], [3]])


def test_tolerance_of_a_contraction_on_an_interval():
    lower, upper = interval_cells(8)  # cells of side 1/4 on [-1, 1]
    image_lower, image_upper = lower / 2, upper / 2
    metric = Euclidean()
    middle = [2, 3, 4, 5]  # [-1/2, 1/2], image [-1/4, 1/4]
    result = tolerance_lower(lower, upper, middle, image_lower, image_upper, metric)
    assert result["lower"] == pytest.approx(.25) and result["positive"]
    assert result["worst_cell"] in (2, 5) and result["cells"] == 4
    # The whole interval: nothing lies outside when its ends are faces of the orbit space.
    everything = tolerance_lower(lower, upper, np.ones(8, bool), image_lower, image_upper, metric)
    assert everything["lower"] == float("inf") and everything["worst_cell"] is None
    # As a window of a larger orbit space the outside counts as complement.
    window = tolerance_lower(lower, upper, np.ones(8, bool), image_lower, image_upper, metric,
                             boundary_faces=[[False, False]])
    assert window["lower"] == pytest.approx(.5)
    one_sided = tolerance_lower(lower, upper, [4, 5, 6, 7], image_lower, image_upper, metric)
    assert one_sided["lower"] == 0 and not one_sided["positive"]  # the image of [0, 1/4] touches 0
    assert tolerance_lower(lower, upper, [], image_lower, image_upper, metric)["lower"] == float("inf")


def test_tolerance_cuts_image_boxes_at_faces_of_the_orbit_space():
    lower, upper = interval_cells(4)  # [-1, -1/2], ..., [1/2, 1]
    image_lower = np.array([[-1.2], [-.9], [.6], [.7]])
    image_upper = np.array([[-.8], [-.6], [.9], [1.3]])
    metric = SphereHeight()
    result = tolerance_lower(lower, upper, [3], image_lower, image_upper, metric)
    assert result["lower"] == pytest.approx(metric.point([[.7]], [[.5]]))
    # In a window of a line, an image box that crosses the upper face leaves the window.
    leaving = tolerance_lower(lower, upper, [3], image_lower, image_upper, Euclidean(),
                              boundary_faces=[[True, False]])
    assert leaving["lower"] == 0 and not leaving["positive"]
    with pytest.raises(ValueError):  # no height of the sphere exceeds 1
        tolerance_lower(lower, upper, [3], image_lower, image_upper, metric,
                        boundary_faces=[[True, False]])
    with pytest.raises(ValueError):  # an image box beyond a face of the orbit space
        tolerance_lower(lower, upper, [3], image_lower + 2, image_upper + 2, metric)
    with pytest.raises(ValueError):  # a missing cell
        tolerance_lower(lower[[0, 1, 3]], upper[[0, 1, 3]], [2], image_lower[[0, 1, 3]],
                        image_upper[[0, 1, 3]], metric)


def test_positive_tolerance_agrees_with_forward_invariance_in_the_plane():
    rng = np.random.default_rng(2)
    edges = np.linspace(-1, 1, 9)
    corners = np.array([(x, y) for x in edges[:-1] for y in edges[:-1]])
    order = rng.permutation(len(corners))  # cells in no particular order
    lower, upper = corners[order], corners[order] + .25
    center = (lower + upper) / 2
    image_lower, image_upper = center * .4 - .05, center * .4 + .05
    offsets, targets = relation(lower, upper, image_lower, image_upper)
    attractor = forward_closure(offsets, targets, np.all(np.abs(center) < .25, axis=1))
    assert is_forward_invariant(offsets, targets, attractor)
    result = tolerance_lower(lower, upper, attractor, image_lower, image_upper, Euclidean(),
                             chunk=3)
    assert result["positive"]
    x = rng.uniform(lower[attractor], upper[attractor])
    assert CellIndex(lower, upper).points_in(x * .4, attractor).all()
    brute = min(Euclidean.lower(a, b, c, d) for a, b in zip(image_lower[attractor], image_upper[attractor])
                for c, d in zip(lower[~attractor], upper[~attractor]))
    assert result["lower"] == pytest.approx(brute)


def test_cell_index_finds_cells_in_any_order():
    edges = np.linspace(0, 1, 5)
    corners = np.array([(x, y) for x in edges[:-1] for y in edges[:-1]])
    order = np.random.default_rng(3).permutation(16)
    lower, upper = corners[order], corners[order] + .25
    index = CellIndex(lower, upper)
    inside = index.containing(np.array([[.3, .6]]))[0]
    assert set(inside) == {int(np.flatnonzero(np.all(lower == [.25, .5], axis=1))[0])}
    corner = set(index.containing(np.array([[.5, .5]]))[0])
    expected = {int(i) for i in np.flatnonzero(np.all((lower <= .5) & (upper >= .5), axis=1))}
    assert corner == expected and len(corner) == 4
    assert set(index.containing(np.array([[0., 0.]]))[0]) - {-1} == {int(np.flatnonzero(np.all(lower == 0, axis=1))[0])}
    assert set(index.containing(np.array([[2., .1]]))[0]) == {-1}
    left = np.flatnonzero(upper[:, 0] <= .5)
    assert index.points_in(np.array([[.5, .9], [.51, .9], [.1, .1]]), left).tolist() == [True, False, True]
    assert index.boxes_meet(np.array([[.6, .1], [.4, .1]]), np.array([[.9, .2], [.9, .2]]), left).tolist() == [False, True]
    with pytest.raises(ValueError):
        CellIndex(lower[:-1], upper[:-1])


def test_residuals_and_comparison():
    metric = Euclidean()
    predicted = np.array([[0., 0.], [1., 1.], [2., 2.]])
    encoded = np.array([[0., .1], [1., 1.3], [5., 6.]])
    residual = residual_modulo_group(predicted, encoded, metric)
    assert residual == pytest.approx([.1, .3, 5.])
    member = np.array([True, True, False])
    passed = compare(.5, residual, member, "sampled")
    assert passed["passed"] and passed["max_residual"] == pytest.approx(.3) and passed["count"] == 2
    assert passed["margin"] == pytest.approx(.2) and passed["label"].startswith("sampled")
    assert not compare(.3, residual, member, "sampled")["passed"]  # the comparison is strict
    assert not compare(1., residual, np.zeros(3, bool), "enclosed")["passed"]
    with pytest.raises(ValueError):
        compare(1., residual, member, "proved")
    bound = residual_upper(predicted - .1, predicted + .1, encoded - .1, encoded + .1, metric)
    assert np.all(bound >= residual) and bound[0] == pytest.approx(np.hypot(.2, .3))


def brute_tolerance(lower, upper, mask, image_lower, image_upper, faces):
    """Nearest points of each image box to each cell outside and to each outside half-space."""
    domain_lower, domain_upper = lower.min(axis=0), upper.max(axis=0)
    best = np.inf
    for low, high in zip(image_lower[mask], image_upper[mask]):
        low = np.where(faces[:, 0], np.maximum(low, domain_lower), low)
        high = np.where(faces[:, 1], np.minimum(high, domain_upper), high)
        for other_low, other_high in zip(lower[~mask], upper[~mask]):
            gap = np.maximum(0, np.maximum(other_low - high, low - other_high))
            best = min(best, float(np.hypot(*gap)))
        for axis in range(lower.shape[1]):
            if not faces[axis, 0]:
                best = min(best, max(0.0, float(low[axis] - domain_lower[axis])))
            if not faces[axis, 1]:
                best = min(best, max(0.0, float(domain_upper[axis] - high[axis])))
    return best


@pytest.mark.parametrize("seed", range(20))
def test_tolerance_against_brute_force_in_the_plane(seed):
    rng = np.random.default_rng(seed)
    shape = rng.integers(2, 6, 2)
    xs, ys = (np.linspace(0, 1, n + 1) for n in shape)
    corners = np.array([(x, y) for x in xs[:-1] for y in ys[:-1]])
    order = rng.permutation(len(corners))
    lower, upper = corners[order], corners[order] + 1 / shape
    mask = rng.random(len(lower)) < .5
    mask[0] = True
    center = rng.uniform(.1, .9, lower.shape)
    radius = rng.uniform(.01, .2, lower.shape)
    image_lower, image_upper = center - radius, center + radius
    faces = rng.random((2, 2)) < .5
    result = tolerance_lower(lower, upper, mask, image_lower, image_upper, Euclidean(),
                             boundary_faces=faces, chunk=int(rng.integers(1, 5)))
    expected = brute_tolerance(lower, upper, mask, image_lower, image_upper, faces)
    assert result["lower"] == pytest.approx(expected, abs=1e-14)
    if result["positive"]:  # then every image box lies in the union of the cells of the set
        points = rng.uniform(image_lower[mask], image_upper[mask])
        assert CellIndex(lower, upper).points_in(points, mask).all()


@pytest.mark.parametrize("seed", range(20))
def test_closures_against_brute_force_on_random_relations(seed):
    rng = np.random.default_rng(100 + seed)
    count = int(rng.integers(2, 11))
    edges = rng.random((count, count)) < rng.uniform(.05, .35)
    offsets = np.concatenate(([0], np.cumsum(edges.sum(axis=1))))
    targets = np.concatenate([np.flatnonzero(row) for row in edges]).astype(np.int64)
    reach = edges | np.eye(count, dtype=bool)
    for _ in range(count):
        reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)
    paths = (edges.astype(int) @ reach.astype(int)) > 0  # paths of positive length
    seeds = rng.random(count) < .4
    assert np.array_equal(forward_closure(offsets, targets, seeds), reach[seeds].any(axis=0))
    back_offsets, back_targets = reverse(offsets, targets)
    for cell in range(count):
        assert sorted(back_targets[back_offsets[cell]:back_offsets[cell + 1]]) \
            == list(np.flatnonzero(edges[:, cell]))
    # Morse sets: classes of cells that lie on a cycle, under mutual reachability.
    recurrent = np.flatnonzero(np.diag(paths))
    morse_sets, seen = [], set()
    for cell in recurrent:
        if cell not in seen:
            members = [int(c) for c in recurrent if reach[cell, c] and reach[c, cell]]
            seen.update(members)
            morse_sets.append(members)
    for members in morse_sets:
        attractor = forward_closure(offsets, targets, members)
        assert is_forward_invariant(offsets, targets, attractor)
        region = region_of_attraction(offsets, targets, attractor, morse_sets)
        outside = [m for m in morse_sets if not attractor[m].all()]
        expected = np.array([not any(reach[cell, m].any() for m in outside)
                             for cell in range(count)])
        assert np.array_equal(region, expected)
        assert all(region[m].all() == attractor[m].all() for m in morse_sets)


@pytest.mark.parametrize("seed", range(10))
def test_cell_index_against_brute_force(seed):
    rng = np.random.default_rng(200 + seed)
    dimension = int(rng.integers(1, 4))
    shape = rng.integers(1, 5, dimension)
    origin, side = rng.uniform(-3, 3, dimension), rng.choice([.1, .25, 1 / 3, 1.], dimension)
    grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    integer = np.column_stack([g.ravel() for g in grids])
    order = rng.permutation(len(integer))
    lower = origin + integer[order] * side
    upper = lower + side
    index = CellIndex(lower, upper)
    vertices = origin + rng.integers(-1, shape + 2, (200, dimension)) * side
    points = np.concatenate((vertices, rng.uniform(origin - side, origin + (shape + 1) * side,
                                                   (200, dimension)),
                             np.where(rng.random((200, dimension)) < .5, vertices[:200],
                                      vertices[:200] + rng.uniform(0, 1, (200, dimension)) * side)))
    cells = rng.random(len(lower)) < .5
    inside = np.all((points[:, None] >= lower[None] - 1e-12) & (points[:, None] <= upper[None] + 1e-12),
                    axis=2)
    assert np.array_equal(index.points_in(points, cells), inside[:, cells].any(axis=1))
    low = rng.uniform(origin - side, origin + (shape + 1) * side, (100, dimension))
    high = low + rng.uniform(0, 2, (100, dimension)) * side
    meets = np.all((high[:, None] >= lower[None]) & (low[:, None] <= upper[None]), axis=2)
    assert np.array_equal(index.boxes_meet(low, high, cells, slack=0), meets[:, cells].any(axis=1))
    with pytest.raises(ValueError):
        index.points_in(points, cells, slack=.6)


def test_inputs_that_are_refused():
    offsets, targets = np.array([0, 1, 2]), np.array([1, 1])
    with pytest.raises(ValueError):
        forward_closure(offsets, targets, [-1])
    with pytest.raises(ValueError):
        forward_closure(offsets, targets, np.array([1, 0, 0]))  # a mask must be boolean
    with pytest.raises(ValueError):
        SphereHeight().point([[1.2]], [[.5]])
    with pytest.raises(ValueError):
        SphereHeight().point([[.1, .2]], [[.5, .1]])
    with pytest.raises(ValueError):
        compare(.5, [.1, .2], [0, 1], "sampled")
    outside = SphereHeight().lower(np.array([.2]), np.array([.4]), np.array([-np.inf]),
                                   np.array([-.5]))
    assert float(outside) == pytest.approx(float(SphereHeight().point([[.2]], [[-.5]])[0]))
    assert Euclidean.lower(np.array([0., 0.]), np.array([1., 1.]), np.array([-np.inf, -np.inf]),
                           np.array([-2., np.inf])) == pytest.approx(2.)
