"""Section 5.3: the closed loop, the data generator, the reference graphs, the prescribed merge,
the equivariance of the networks, and a short training run."""
import importlib
import json

import numpy as np
import pytest

from morsegraphs import grids, training


@pytest.fixture(autouse=True)
def float32():
    """The networks of this example run in float32; other examples may enable float64."""
    import jax
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def ac():
    return importlib.import_module("morsegraphs.attitude_control")


SMALL = dict(width=8, scalar_width=4, knots=9)


def vertical(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])


@pytest.mark.parametrize("policy", ["nominal", "coning"])
def test_closed_loop_identities_and_equivariance(ac, policy):
    x = ac.sample(66, 5, policy, eta_radius=3.)[:-2]
    r, q, eta, qdot, moment = ac.components(x, policy)
    wdot = np.einsum("...ij,...j->...i", r, ac.rhs(x, policy)[:, 10:13])
    assert np.allclose(wdot - ac.derivative_rate(q, qdot, policy), -2 * eta, atol=1e-11)
    assert np.allclose(qdot, ac.field(q, policy) + np.cross(eta, q), atol=1e-12)
    g = vertical(.4)
    _, tq, teta, _, tmoment = ac.components(ac.group_action(x, g), policy)
    assert np.allclose(tq, q @ g.T, atol=1e-12) and np.allclose(teta, eta @ g.T, atol=1e-12)
    assert np.allclose(tmoment, moment, atol=1e-11)  # the body moment is invariant
    assert np.allclose(ac.observe(ac.group_action(x, g)), ac.observation_action(ac.observe(x), g), atol=1e-12)
    with pytest.raises(ValueError):
        ac.group_action(x, np.diag([1., -1., -1.]))


def test_height_flow_and_axis_flow(ac):
    assert np.allclose(ac.height_flow(np.array([-1., .25, 1.]), 1., "coning"), [-1., .25, 1.], atol=1e-14)
    assert np.all(np.diff(ac.height_flow(np.array([-.9, 0., .9]), 1., "nominal")) > 0)
    z = ac.height_flow(np.array([-.99, .9]), 30., "coning")
    assert np.allclose(z, .25, atol=1e-6)  # C attracts every nonpolar axis
    q = np.array([[.6, 0., .8], [0., -.8, -.6]])
    image = ac.axis_flow(q, .7, "coning")
    assert np.allclose(np.linalg.norm(image, axis=1), 1.)
    g = vertical(1.1)
    assert np.allclose(ac.axis_flow(q @ g.T, .7, "coning"), image @ g.T)


def test_generate_data_is_deterministic_and_on_the_invariant_set(ac, tmp_path):
    settings = ac.DEFAULTS | dict(policies=["coning"], trajectories=dict(train=6, validation=3, test=4), intervals=3)
    metadata = ac.generate_data(tmp_path / "a", settings)
    again = ac.generate_data(tmp_path / "b", settings)
    assert {k: v["sha256"] for k, v in metadata["files"].items()} == {k: v["sha256"] for k, v in again["files"].items()}
    with pytest.raises(FileExistsError):
        ac.generate_data(tmp_path / "a", settings)
    data = ac.load_data(tmp_path / "a", "coning")
    assert data["train"].shape == (6, 4, 18) and data["test"].shape == (4, 4, 18)
    with np.load(tmp_path / "a/coning-test.npz") as saved:
        initial = saved["initial_state"]
        assert np.allclose(saved["times"], [0., .1, .2, .3])
    assert np.array_equal(ac.observe(initial), data["test"][:, 0])
    assert np.allclose(ac.components(initial, "coning")[2], 0., atol=1e-12)  # eta = 0
    assert data["test"][-2, 0, 14] == pytest.approx(-1.) and data["test"][-1, 0, 14] == pytest.approx(1.)
    # On eta = 0 the observed axis follows V_a.
    exact = ac.axis_flow(data["test"][:, 0, 12:15], .3, "coning")
    assert np.max(np.abs(data["test"][:, 3, 12:15] - exact)) < 1e-9
    assert metadata["files"]["coning-test.npz"]["seed"] == settings["data_seeds"]["test"]


@pytest.mark.parametrize("policy", ["nominal", "coning"])
def test_reference_graph_order_and_annotations(ac, policy):
    graph = ac.reference_graph(policy)
    rows = {tuple(row["cell_ids"]): row for row in graph.nodes}
    south, north = rows[(0,)], rows[(127,)]
    assert south["annotations"] == ["0", "0"]
    if policy == "nominal":
        assert len(rows) == 2 and north["annotations"] == ["x-1", "0"]
        assert graph.edges == [[south["node"], north["node"]]]
    else:
        circle = rows[(79, 80)]  # the cells of z = 1/4
        assert len(rows) == 3 and north["annotations"] == ["0", "0"] and circle["annotations"] == ["x-1", "0"]
        assert sorted(graph.edges) == sorted([[south["node"], circle["node"]], [north["node"], circle["node"]]])
    merged = ac.prescribed_merge(graph, policy)
    assert [(row["cell_ids"], row["annotations"]) for row in merged.nodes] == \
        [(row["cell_ids"], row["annotations"]) for row in graph.nodes]


def coning_like(ac, change=None):
    """The time-two map of the coning heights on 2^4 cells, with the images of some cells changed."""
    boxes = grids.cells(16, [-1.], [1.])
    images = grids.sampled_hulls(lambda z: ac.height_flow(z, 2., "coning"), boxes, 33, 1e-9, [-1.], [1.])
    for cell, image in (change or {}).items():
        images[cell] = image
    return grids.morse_graph(boxes, images)


def test_prescribed_merge_of_the_coning_structure(ac):
    plain = ac.prescribed_merge(coning_like(ac), "coning")
    assert [(row["cell_ids"], row["annotations"]) for row in plain.nodes] == \
        [([9, 10], ["x-1", "0"]), ([0], ["0", "0"]), ([15], ["0", "0"])]
    # The cell [.375, .5] maps into itself and into C: a fourth Morse set, with trivial index.
    graph = coning_like(ac, {11: [.3, .45]})
    assert sorted((row["cell_ids"], row["annotations"]) for row in graph.nodes)[2] == ([11], ["0", "0"])
    merged = ac.prescribed_merge(graph, "coning")
    rows = {tuple(row["cell_ids"]): row for row in merged.nodes}
    assert sorted(rows) == [(0,), (9, 10, 11), (15,)] and rows[(9, 10, 11)]["annotations"] == ["x-1", "0"]
    assert sorted(len(row["original_nodes"]) for row in merged.nodes) == [1, 1, 2]
    assert merged.minimal == [rows[(9, 10, 11)]["node"]]
    assert sorted(merged.edges) == sorted([[rows[(0,)]["node"], rows[(9, 10, 11)]["node"]],
                                           [rows[(15,)]["node"], rows[(9, 10, 11)]["node"]]])
    with pytest.raises(ValueError):
        ac.prescribed_merge(coning_like(ac, {13: [.66, .7]}), "coning")  # a second attractor between the poles
    with pytest.raises(ValueError):
        ac.prescribed_merge(ac.reference_graph("nominal"), "coning")  # the north pole attracts


def test_networks_are_equivariant_and_preserve_the_sphere(ac):
    import jax.numpy as jnp
    model = ac.AttitudeModel("coning", seed=3, settings=SMALL)
    assert all(v.dtype == jnp.float32 for group in model.params for v in group)
    encode, decode, advance, coordinate, inverse, orbit_decode, latitude = model.functions()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 18)).astype(np.float32)
    g = vertical(.9)
    gx = ac.observation_action(x, g).astype(np.float32)
    q = np.asarray(encode(jnp.asarray(x)))
    assert np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-6)
    assert np.allclose(np.asarray(encode(jnp.asarray(gx))), q @ g.T, atol=1e-5)
    gq = (q @ g.T).astype(np.float32)
    assert np.allclose(np.asarray(advance(jnp.asarray(gq))), np.asarray(advance(jnp.asarray(q))) @ g.T, atol=1e-5)
    assert np.allclose(np.asarray(decode(jnp.asarray(gq))), np.asarray(decode(jnp.asarray(q))) @ g.T, atol=1e-5)
    assert np.allclose(np.linalg.norm(np.asarray(advance(jnp.asarray(q))), axis=1), 1., atol=1e-6)
    assert np.asarray(latitude(jnp.array([-1., 1.]))).tolist() == [-1., 1.]
    heights = jnp.linspace(-1., 1., 101)
    u = np.asarray(coordinate(heights))
    assert u[0] == -1. and u[-1] == 1. and np.all(np.diff(u) > 0)
    assert np.allclose(np.asarray(inverse(jnp.asarray(u))), np.asarray(heights), atol=1e-6)
    p = model.prepare(model.params)
    lower = np.asarray(model.quotient(p, coordinate(jnp.asarray(q[:, 2]))))
    assert np.allclose(np.asarray(coordinate(advance(jnp.asarray(q))[:, 2])), lower, atol=2e-6)
    native = ac.AttitudeModel("coning", seed=3, settings=SMALL, backend="native")
    native.params = model.params
    assert np.allclose(np.asarray(native.functions()[0](jnp.asarray(x))), q, atol=3e-5)


def test_float64_is_refused(ac):
    import jax
    jax.config.update("jax_enable_x64", True)
    with pytest.raises(RuntimeError):
        ac.AttitudeModel("nominal", settings=SMALL)


def test_training_smoke(ac, tmp_path):
    settings = ac.DEFAULTS | SMALL | dict(policies=["nominal"], trajectories=dict(train=16, validation=8, test=8),
                                          intervals=11, steps=4, warmup_steps=2, validate_every=2, batch=8)
    ac.generate_data(tmp_path / "data", settings)
    data = ac.load_data(tmp_path / "data", "nominal")
    progress = training.Progress(tmp_path / "run", "ex3_attitude_control", "nominal-seed0", settings["steps"])
    model = ac.train(tmp_path / "run/model", data, 0, settings, progress)
    config = json.loads((tmp_path / "run/model/config.json").read_text())
    assert {"seed", "best_step", "best_validation", "seconds", "settings", "versions", "policy"} <= set(config)
    assert config["settings"]["steps"] == 4 and config["policy"] == "nominal"
    history = json.loads((tmp_path / "run/model/history.json").read_text())
    assert [row["step"] for row in history] == [0, 1, 2, 4]
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "trained"
    loaded = ac.load(tmp_path / "run/model")
    assert all(np.array_equal(a, b) for group_a, group_b in zip(loaded.params, model.params)
               for a, b in zip(group_a, group_b))
    assert config["best_validation"] == min(row["validation"]["objective"] for row in history)
