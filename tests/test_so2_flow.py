"""Section 5.1: the reference map and its graph, the data generator, the networks and training."""
import copy
from pathlib import Path
import json
import shutil

import numpy as np
import objax
import pytest
from scipy.integrate import solve_ivp

from morsegraphs import so2_flow, training


def small(**sizes):
    settings = copy.deepcopy(so2_flow.DEFAULTS)
    settings["data"]["sizes"] = dict(train=64, validation=16, test=16, **sizes)
    return settings


@pytest.fixture(scope="module")
def model():
    np.random.seed(0)
    objax.random.DEFAULT_GENERATOR.seed(0)
    options = so2_flow.DEFAULTS["training"]
    return so2_flow.SphereInterval(options["width"], options["layers"], options["hidden"])


@pytest.fixture(scope="module")
def fitted():
    """The selected fit of data/ex1_so2_flow, whose encoder is increasing."""
    folder = Path(__file__).resolve().parents[1] / "data/ex1_so2_flow"
    if not (folder / "selection.json").exists():
        pytest.skip("no stored fit")
    selected = json.loads((folder / "selection.json").read_text())["all"]["selected"]
    return so2_flow.load(folder / "models" / selected)


def height_flow(z, duration):
    answer = solve_ivp(lambda _, y: y * (.16 - y**2) * (1 - y**2), (0., duration), np.atleast_1d(z),
                       method="DOP853", rtol=1e-13, atol=1e-15)
    return answer.y[:, -1]


def test_enclosures_of_the_time_ten_map():
    heights = np.array([-.97, -.6, -.41, -.2, -1e-3, 1e-3, .05, .39, .41, .8, .999])
    for z, value in zip(heights, height_flow(heights, 10.), strict=True):
        low, high = so2_flow.time_ten(z)
        assert low <= high and high - low < 1e-12
        assert low - 1e-10 <= value <= high + 1e-10
        assert so2_flow.time_ten(-z) == (-high, -low)
    for z in (-1., 0., 1.):
        assert so2_flow.time_ten(z) == (z, z)
    images = so2_flow.time_ten_images(np.array([[-.5, -.4], [.1, .2]]))
    assert images[0, 0] <= so2_flow.time_ten(-.5)[0] and images[1, 1] >= so2_flow.time_ten(.2)[1]


def test_reference_graph_order_and_annotations():
    graph = so2_flow.reference_graph()
    assert len(graph.boxes) == 2**7
    nodes = sorted(graph.nodes, key=lambda row: row["hull"][0])
    assert [row["annotations"] for row in nodes] == [["0", "0"], ["x-1", "0"], ["0", "x-1"], ["x-1", "0"], ["0", "0"]]
    south, low, equator, high, north = [row["node"] for row in nodes]
    assert sorted(map(tuple, graph.edges)) == sorted([(south, low), (equator, low), (equator, high), (north, high)])
    assert graph.minimal == sorted([low, high])
    hulls = [row["hull"] for row in nodes]
    assert hulls[0][0] == -1 and hulls[4][1] == 1
    assert hulls[1][0] <= -.4 <= hulls[1][1] and hulls[2][0] < 0 < hulls[2][1] and hulls[3][0] <= .4 <= hulls[3][1]


def test_data_generator(tmp_path):
    settings = small()
    metadata = so2_flow.generate_data(tmp_path / "a", settings)
    data = so2_flow.load_data(tmp_path / "a")
    assert so2_flow.generate_data(tmp_path / "b", settings)["sha256"] == metadata["sha256"]
    with pytest.raises(FileExistsError):
        so2_flow.generate_data(tmp_path / "a", settings)
    for split, count in settings["data"]["sizes"].items():
        for t in (0, 1, 10):
            p, w = data[f"{split}_p{t}"], data[f"{split}_w{t}"]
            assert p.shape == (count, 3) and w.shape == (count,)
            assert np.allclose(np.linalg.norm(p, axis=1), 1, rtol=0, atol=1e-14)
            assert np.allclose(p[:, 2], np.tanh(w), rtol=0, atol=1e-15)
        assert np.sum(1 - np.abs(data[f"{split}_p0"][:, 2]) <= 1e-2) >= count // 4
    # The states at time one follow the vector field V on S^2.
    starts = data["train_p0"][:8]
    for start, later in zip(starts, data["train_p1"][:8], strict=True):
        answer = solve_ivp(lambda _, p: so2_flow.vector_field(p), (0., 1.), start, method="DOP853",
                           rtol=1e-12, atol=1e-14)
        assert np.allclose(answer.y[:, -1], later, rtol=0, atol=1e-9)
    # Q(z) = w^21 (1 - w)^4 / |w - 0.16|^25, w = z^2, grows by exp(6.72 t) along the flow.
    def log_q(w):
        z = np.tanh(w)
        return 42 * np.log(np.abs(z)) - 8 * np.log(np.cosh(w)) - 25 * np.log(np.abs(z**2 - .16))
    w0, w1 = data["train_w0"], data["train_w1"]
    z = np.abs(np.tanh(w0))
    away = ((z > .05) & (z < .35)) | ((z > .45) & (z < .95))
    assert away.sum() > 10
    assert np.allclose(log_q(w1[away]) - log_q(w0[away]), 6.72, rtol=0, atol=1e-6)


def test_meridian_samples_follow_the_reference_map():
    analysis = dict(so2_flow.DEFAULTS["analysis"], region_samples=257, pole_samples_per_end=8)
    theta, starts, images = so2_flow.meridian_samples(analysis)
    assert len(theta) == 257 + 16 and np.all(np.diff(theta) > 0)
    assert np.allclose(starts[:, 1], 0) and np.allclose(np.linalg.norm(images, axis=1), 1)
    for z, image in zip(starts[::16, 2], images[::16], strict=True):
        low, high = so2_flow.time_ten(z)
        assert low - 1e-10 <= image[2] <= high + 1e-10
    radius = np.hypot(images[:, 0], images[:, 1])
    inside = radius > 1e-6
    assert np.allclose(np.arctan2(images[inside, 1], images[inside, 0]), 6. - 2 * np.pi)


def test_networks_are_invariant_and_fix_the_poles(model):
    maps = so2_flow.functions(model)
    rng = np.random.default_rng(3)
    p = rng.normal(size=(256, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    code = maps["encode"](p)
    for angle in (.3, 2.1, -1.4):
        assert np.allclose(maps["encode"](so2_flow.rotate(p, angle)), code, rtol=0, atol=1e-12)
    assert np.allclose(maps["encode"](np.array([[0., 0., -1.], [0., 0., 1.]])), [-1., 1.], rtol=0, atol=1e-15)
    for name in ("decode", "advance", "time_ten"):
        assert maps[name](np.array([-1., 1.])).tolist() == [-1., 1.]
    assert np.allclose(maps["height_map"](p[:, 2]), maps["encode"](so2_flow.meridian(p[:, 2])))


def test_learned_graph_and_regions_run(fitted):
    model = fitted
    settings = copy.deepcopy(so2_flow.DEFAULTS)
    settings["analysis"].update(cells=16, image_samples=5, region_samples=129, pole_samples_per_end=8)
    graph, merged = so2_flow.learned_graph(model, settings)
    assert len(graph.boxes) == 16 and np.all(np.abs(graph.images) <= 1)
    if merged is not None:
        assert {row["role"] for row in merged.nodes} == set(so2_flow.REFERENCE_HEIGHTS)
    rows, details = so2_flow.attracting_regions(model, graph, settings)
    assert [row["metric"] for row in rows] == ["euclidean"] * len(graph.minimal) + ["chordal"] * len(graph.minimal)
    assert details["labels"].shape == (16,) and len(details["codes"]) == 129 + 16
    lifted = so2_flow.lifted_grid(model, settings)
    assert np.all(np.diff(lifted["physical_boundaries"]) > 0)
    assert np.max(np.abs(lifted["encoder_inverse_residual"])) < 1e-10


def test_training_smoke(tmp_path):
    settings = small()
    settings["training"].update(steps=3, batch=16, validate_every=1)
    data = so2_flow.trajectories(settings["data"])
    progress = training.Progress(tmp_path / "seed0", "ex1_so2_flow", "seed0", 3)
    trained = so2_flow.train(tmp_path / "seed0", data, 0, settings, progress)
    config = training.read(tmp_path / "seed0/config.json")
    assert {"seed", "best_step", "best_validation", "seconds", "settings", "versions"} <= set(config)
    assert config["settings"] == settings and config["seed"] == 0
    assert [row["step"] for row in training.read(tmp_path / "seed0/history.json")] == [0, 1, 2, 3]
    assert training.read(tmp_path / "seed0/status.json")["status"] == "trained"
    assert np.isclose(so2_flow.objective_value(trained, data), config["best_validation"], rtol=1e-12)
    reloaded = so2_flow.load(tmp_path / "seed0")
    u = np.linspace(-1, 1, 33)
    assert np.array_equal(so2_flow.functions(reloaded)["advance"](u), so2_flow.functions(trained)["advance"](u))
    with pytest.raises(FileExistsError):
        so2_flow.train(tmp_path / "seed0", data, 0, settings)


def test_reference_roles_and_indices_in_the_sphere():
    graph = so2_flow.reference_graph()
    roles = so2_flow.reference_roles(graph)
    assert sorted(roles.values()) == sorted(so2_flow.ORDER)
    assert so2_flow.role_edges(graph) == [["S", "A-"], ["C", "A-"], ["C", "A+"], ["N", "A+"]]
    comparison = so2_flow.compare_graphs(graph, graph)
    assert comparison["agree"] and all(row["agree"] for row in comparison["nodes"])
    indices = so2_flow.reconstructed_indices({"reference": graph})["reference"]
    assert {role: row["in_X"] for role, row in indices.items()} == so2_flow.PROVED
    assert indices["S"]["fixed_point_cells"] == 1 and indices["C"]["fixed_point_cells"] == 0
    assert so2_flow.homology_degrees(so2_flow.PROVED["C"]) == "H_1 = H_2 = F"


def test_compare_records_is_exact_on_graphs_and_relative_elsewhere():
    saved = {"graphs": {"nodes": [{"hull": [0.5, 0.75], "annotations": ["x-1", "0"]}]},
             "regions": {"rows": [{"tolerance": 0.08, "cells": 45, "passed": True}]}}
    assert so2_flow.compare_records(saved, saved) == (7, [])
    close = copy.deepcopy(saved)
    close["regions"]["rows"][0]["tolerance"] *= 1 + 1e-12
    assert so2_flow.compare_records(close, saved)[1] == []
    far = copy.deepcopy(saved)
    far["regions"]["rows"][0]["tolerance"] *= 1 + 1e-6
    assert [d[0] for d in so2_flow.compare_records(far, saved)[1]] == ["/regions/rows/0/tolerance"]
    moved = copy.deepcopy(saved)
    moved["graphs"]["nodes"][0]["hull"][0] += 1e-15
    assert [d[0] for d in so2_flow.compare_records(moved, saved)[1]] == ["/graphs/nodes/0/hull/0"]
    renamed = copy.deepcopy(saved)
    renamed["graphs"]["nodes"][0]["annotations"] = ["0", "0"]
    assert len(so2_flow.compare_records(renamed, saved)[1]) == 1


def test_height_flow_and_bands():
    z = np.array([-1., -.7, -.4, 0., .2, .4, .9, 1.])
    values = so2_flow.height_flow(z, 3.)
    assert np.allclose(values, height_flow(z, 3.), rtol=0, atol=1e-10)
    assert values[[0, 3, 7]].tolist() == [-1., 0., 1.]
    graph = so2_flow.reference_graph()
    bands = so2_flow.morse_bands(graph, np.linspace(-1, 1, len(graph.boxes) + 1))
    assert sorted(role for _, _, role in bands) == sorted(so2_flow.ORDER)
    equator = [band for band in bands if band[2] == "C"]
    assert equator == [(-0.015625, 0.015625, "C")]


@pytest.mark.skipif(not all(shutil.which(tool) for tool in ("dot", "pdftoppm")), reason="needs Graphviz and Poppler")
def test_appendix_height_panels(tmp_path):
    panels = so2_flow.figure_sphere_height(tmp_path)
    assert list(panels) == ["dynamics/sphere-trajectories", "dynamics/sphere-height-map"]
    for path in panels.values():
        assert path.suffix == ".pdf" and path.exists() and path.with_suffix(".svg").exists()
