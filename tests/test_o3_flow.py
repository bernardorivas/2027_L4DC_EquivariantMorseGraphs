"""Section 5.2: the O(3) flow, its explicit enclosures, data, networks and Morse graphs."""
from fractions import Fraction
import json

import numpy as np
import pytest

from morsegraphs import o3_flow as m
from morsegraphs import training


def rotations(seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for reflection in (False, True):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix[:, 0] *= np.linalg.det(matrix) * (-1 if reflection else 1)
        out.append(matrix)
    return out


def act(matrix, x):
    y = np.array(x, float, copy=True)
    y[..., 1:] = y[..., 1:] @ matrix.T
    return y


def test_samples_lie_in_X_and_include_axes_and_equilibria():
    x = m.sample(64, 5)
    assert x.shape == (64, 4)
    ar = m.orbit(x)
    assert np.all(np.abs(ar[:, 0]) <= 1.5) and np.all(ar[:, 1] <= 1.5 + 1e-12)
    assert np.allclose(ar[-4:], m.CONTROLS, atol=1e-15)
    assert np.any(ar[:-4, 0] == 0) and np.any(ar[:-4, 1] == 0)
    assert np.array_equal(x, m.sample(64, 5))
    with pytest.raises(ValueError):
        m.sample(7, 0)


def test_generate_and_load_data(tmp_path):
    settings = m.settings_with(**{"data.sizes": [16, 8, 8]})
    values = m.generate_data(tmp_path, settings)
    loaded = m.load_data(tmp_path)
    assert set(loaded) == {"train_x", "validation_x", "test_x"}
    assert all(np.array_equal(values[k], loaded[k]) for k in loaded)
    assert loaded["train_x"].shape == (16, 11, 4) and loaded["test_x"].shape == (8, 11, 4)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["sizes"] == [16, 8, 8] and metadata["split_seeds"] == settings["data"]["split_seeds"]
    assert metadata["sha256"] == training.digest(tmp_path / "trajectories.npz")
    x = loaded["train_x"]
    assert np.allclose(x[:, 3], m.physical_step(x[:, 0], 3), rtol=0, atol=1e-14)


def test_closed_form_map_is_equivariant_and_reduces_to_the_quotient_map():
    x = m.sample(256, 11)
    for matrix in rotations():
        assert np.allclose(m.physical_step(act(matrix, x)), act(matrix, m.physical_step(x)), atol=1e-14)
    assert np.allclose(m.orbit(m.physical_step(x)), m.quotient_step(m.orbit(x)), atol=1e-14)
    assert np.allclose(m.physical_step(m.physical_step(x), 9), m.physical_step(x, 10), atol=1e-14)
    assert np.allclose(m.quotient_step(m.CONTROLS), m.CONTROLS, atol=1e-15)
    assert np.allclose(m.quotient_step([[.5, 1.]]), [[np.sqrt(3) / 2, 3 / np.sqrt(33)]])


def test_explicit_enclosures_contain_sampled_images():
    rng = np.random.default_rng(4)
    low = np.column_stack((rng.uniform(-1.5, 1.3, 200), rng.uniform(0, 1.3, 200)))
    boxes = np.column_stack((low, low + rng.uniform(0, .2, (200, 2))))
    boxes[:, 2:] = np.minimum(boxes[:, 2:], [1.5, 1.5])
    boxes[0] = [-.1, 0., .1, .2]  # a box that meets a = 0 and r = 0
    images = m.reference_images(boxes)
    t = rng.uniform(0, 1, (200, 50, 2))
    t[:, :4] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    points = boxes[:, None, :2] + (boxes[:, None, 2:] - boxes[:, None, :2]) * t
    values = m.quotient_step(points)
    # The binary64 evaluation of the map is itself off by a few units in the last place.
    assert np.all(values >= images[:, None, :2] - 1e-15) and np.all(values <= images[:, None, 2:] + 1e-15)
    # olF(1/2, 1) = (sqrt(3)/2, 3/sqrt(33)), enclosed by neighboring binary64 numbers.
    low_a, low_r, high_a, high_r = map(Fraction, m.reference_images([[.5, 1., .5, 1.]])[0])
    assert low_a**2 < Fraction(3, 4) < high_a**2 and low_r**2 < Fraction(9, 33) < high_r**2
    assert np.nextafter(float(low_a), 2.) == float(high_a) and np.nextafter(float(low_r), 2.) == float(high_r)
    # Away from a = 0 the extremes are at corners, so the enclosures are tight to rounding.
    away = (boxes[:, 0] > 0) | (boxes[:, 2] < 0)
    sampled = values.max(axis=1) - values.min(axis=1)
    assert np.all((images[:, 2:] - images[:, :2])[away] <= sampled[away] + 1e-14)
    with pytest.raises(ValueError):
        m.reference_images([[-2., 0., 0., 1.]])


def test_reference_graph_order_and_annotations():
    graph = m.reference_graph()
    assert len(graph.boxes) == 2**12
    role = {row["node"]: row["role"] for row in graph.nodes}
    assert sorted(role.values()) == ["A+", "A-", "C", "O"]
    annotations = {row["role"]: row["annotations"] for row in graph.nodes}
    assert annotations == {"O": ["0", "0", "0"], "C": ["0", "x-1", "0"],
                           "A-": ["x-1", "0", "0"], "A+": ["x-1", "0", "0"]}
    assert sorted((role[a], role[b]) for a, b in graph.edges) == [("C", "A+"), ("C", "A-"), ("O", "C")]
    assert m.groups(graph) == {name: [node] for node, name in role.items()}
    agreement = m.compare_with_reference(m.merge(graph), graph)
    assert agreement["same_order"] and agreement["same_annotations"]
    with pytest.raises(ValueError):
        m.groups(m.label_roles(m.reference_graph(), m.ANCHORS, radius=.5))


@pytest.fixture(scope="module")
def small_model():
    np.random.seed(0)
    import objax
    objax.random.DEFAULT_GENERATOR.seed(0)
    return m.InvariantDynamics(width=4, layers=1)


def test_networks_are_invariant_and_keep_the_axes(small_model):
    encode, decode, advance = m.functions(small_model)
    x = m.sample(128, 2)
    codes = encode(x)
    for matrix in rotations(7):
        assert np.max(np.abs(encode(act(matrix, x)) - codes)) < 1e-12
    assert np.all(np.sign(codes[:, 0]) == np.sign(x[:, 0])) and np.all(codes[:, 1] >= 0)
    z = np.column_stack((np.r_[np.zeros(20), np.linspace(-2, 2, 20)], np.r_[np.linspace(0, 2, 20), np.zeros(20)]))
    images = advance(z)
    assert np.all(images[:20, 0] == 0) and np.all(images[20:, 1] == 0)
    assert np.all((images >= m.LOWER) & (images <= m.UPPER))
    decoded = decode(z)
    assert np.all((decoded >= m.PHYSICAL_LOWER) & (decoded <= m.PHYSICAL_UPPER))


def test_training_smoke(tmp_path):
    settings = m.settings_with(**{"data.sizes": [32, 16, 16], "training.width": 4, "training.layers": 1,
                                  "training.steps": 3, "training.batch": 8, "training.validate_every": 1})
    data = m.generate_data(tmp_path / "data", settings)
    progress = training.Progress(tmp_path / "run", "ex2_o3_flow", "seed0", 3)
    model = m.train(tmp_path / "run" / "seed0", data, 0, settings, progress)
    config = json.loads((tmp_path / "run/seed0/config.json").read_text())
    assert {"seed", "best_step", "best_validation", "seconds", "settings", "versions"} <= set(config)
    assert len(json.loads((tmp_path / "run/seed0/history.json").read_text())) == 4
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "trained"
    loaded = m.load(tmp_path / "run/seed0")
    assert m.validation_objective(loaded, data) == config["best_validation"]
    assert m.validation_objective(model, data) == config["best_validation"]


def test_merge_groups_an_extra_morse_set_with_its_reference_set():
    # On 2^7 x 2^7 cells the reference graph has a fifth Morse set, of trivial index,
    # next to C; it lies in the region of C, and the merged set has the index of C.
    graph = m.reference_graph(7)
    assert len(graph.nodes) == 5
    assert sorted(row["role"] for row in graph.nodes) == ["A+", "A-", "C", "C", "O"]
    merged = m.merge(graph)
    assert [row["original_nodes"] for row in merged.nodes if row["role"] == "C"] == [[2, 3]]
    agreement = m.compare_with_reference(merged, m.reference_graph())
    assert agreement["same_order"] and agreement["same_annotations"]
