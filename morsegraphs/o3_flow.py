"""Section 5.2: an O(3)-equivariant flow on R^4 and its learned orbit-space dynamics.

The flow of
    a' = a (1 - a^2),    v' = (1 - 2 a^2 - |v|^2) v
on X = [-3/2, 3/2] x B^3_{3/2} commutes with (a, v) -> (a, Q v), Q in O(3). F is its
time-log(3) map. In the orbit coordinates (a, r) = (a, |v|) of X/G, with D = 1 + 8 a^2,
    F(a, v) = (3a / sqrt(D), 3v / sqrt(D (D + 8 |v|^2))),
    olF(a, r) = (3a / sqrt(D), 3r / sqrt(D (D + 8 r^2))).
The repelling origin O connects to the image (0, 1) of the saddle sphere
C = {a = 0, |v| = 1}, which connects to the sinks A_- = (-1, 0) and A_+ = (1, 0).

A joint fit learns an invariant encoder E: X -> Z = [-2, 2] x [0, 2], a time-log(3) map
f of Z and an orbit decoder d: Z -> X/G. CMGDB computes the Morse graph of the data
map built from the pairs (E(x), f(E(x))) on 2^6 x 2^6 cells, where a cell without
samples maps outside the box, and of the sampled image hulls of f on the same grid.
The reference graph uses explicit binary64 enclosures of olF on 2^6 x 2^6 cells of
X/G. All learned computations are float64 on the CPU.

The last sections hold the indices in X reconstructed from the orbit space, the panels
of the figures of Section 5.2 and its appendix, the record of every result that the
notebook `ex2_o3_flow.ipynb` checks, and the steps of that notebook, each of which
returns one table, one line or the saved panels of a figure.

Import this module after `device.configure`, if that is called at all: it imports
JAX and enables float64.
"""
import copy
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.metadata import version
import logging
from math import inf, nextafter, sqrt
from pathlib import Path
import time
import warnings
import weakref

# tqdm, which EMLP imports, warns in a notebook that progress bars need ipywidgets; none is shown.
warnings.filterwarnings("ignore", message="IProgress not found")
# fontTools logs a warning about font timestamps when Matplotlib embeds fonts in a PDF.
logging.getLogger("fontTools").setLevel(logging.ERROR)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import objax
from emlp.groups import O
from emlp.nn.objax import EMLP
from emlp.reps import T, V
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import CMGDB

from . import grids, homology, plotting, regions, training
from .so2_flow import Line, Table, compare_records, weights_digest

TAU = float(np.log(3.))
LOWER, UPPER = np.array([-2., 0.]), np.array([2., 2.])
PHYSICAL_LOWER, PHYSICAL_UPPER = np.array([-1.5, 0.]), np.array([1.5, 1.5])
# The face r = 0 of the box is a face of the orbit space (the fixed points of the
# reflections of v); the other faces of Z and of X/G are not.
FACES = np.array([[False, False], [True, False]])
ANCHORS = {"O": [0., 0.], "C": [0., 1.], "A-": [-1., 0.], "A+": [1., 0.]}
CONTROLS = np.array([ANCHORS[name] for name in ("O", "C", "A-", "A+")])
NETWORKS = ("encoder", "decoder", "dynamics")
SPLITS = ("train", "validation", "test")
HORIZONS = 11

DEFAULTS = {
    "experiment": "joint-quotient-v1",
    "seeds": [0, 1, 2],
    "data": {"sizes": [16384, 4096, 4096], "split_seeds": [20261112, 20261113, 20261114]},
    "training": {
        "width": 32,
        "layers": 2,
        "steps": 20000,
        "batch": 512,
        "learning_rate_start": 0.002,
        "learning_rate_end": 0.0001,
        "validate_every": 100,
    },
    "recipe": {
        "encoder_gain_log_bound": 0.2,
        "decoder_gain_log_bound": 0.35,
        "time_gain_log_bound": 3.0,
    },
    "analysis": {
        "depths": [6],
        "primary_depth": 6,
        "physical_section_points_per_axis": 513,
        "data_map": {
            "source_sizes": [257, 513, 1025],
            "primary_source_size": 1025,
            "map_empty": "outside",
            "domain_padding": False,
            "padding": False,
        },
        # Not in the current config.json, where these are fixed in the code.
        "tolerances": {"samples_per_axis": 1024},
        "sampled_hulls": {"depth": 6, "samples_per_axis": 5, "padding": 1e-9, "group_radius_cells": 4},
    },
}


# The system

def orbit(x):
    """The orbit coordinates (a, |v|) of states (a, v1, v2, v3)."""
    x = np.asarray(x)
    return np.stack((x[..., 0], np.linalg.norm(x[..., 1:], axis=-1)), axis=-1)


def section(ar):
    """The representative (a, r, 0, 0) of each orbit (a, r)."""
    ar = np.asarray(ar)
    return np.concatenate((ar, np.zeros((*ar.shape[:-1], 2))), axis=-1)


def physical_step(x, steps=1):
    """The time-(steps log 3) map of the flow, in closed form."""
    x = np.asarray(x)
    a, v = x[..., :1], x[..., 1:]
    factor = 9.**steps
    d = 1 + (factor - 1)*a*a
    radial = d + (factor - 1)*np.sum(v*v, axis=-1, keepdims=True)
    return np.concatenate((np.sqrt(factor)*a/np.sqrt(d), np.sqrt(factor)*v/np.sqrt(d*radial)), axis=-1)


def quotient_step(ar):
    """The quotient map olF in the coordinates (a, r)."""
    ar = np.asarray(ar, float)
    d = 1 + 8*ar[..., :1]**2
    return np.concatenate((3*ar[..., :1]/np.sqrt(d), 3*ar[..., 1:]/np.sqrt(d*(d + 8*ar[..., 1:]**2))), axis=-1)


def verify_physics(count=64, seed=2229):
    """Compare the closed-form map with DOP853 on the four-dimensional vector field."""
    points = sample(count, seed)
    def rhs(_, y):
        a, v = y[0], y[1:]
        return np.r_[a*(1 - a*a), (1 - 2*a*a - v@v)*v]
    values = []
    for x in points:
        solution = solve_ivp(rhs, (0., 10*TAU), x, t_eval=[TAU, 10*TAU], method="DOP853", rtol=2e-13, atol=2e-14)
        if not solution.success:
            raise RuntimeError(solution.message)
        values.append(solution.y.T)
    values = np.asarray(values)
    exact = np.stack([physical_step(points, k) for k in (1, 10)], axis=1)
    errors = np.linalg.norm(values - exact, axis=-1)
    composition = np.max(np.abs(physical_step(physical_step(points, 1), 9) - physical_step(points, 10)))
    return dict(independent_method="DOP853 on the four-dimensional vector field", starts=count,
                maximum_state_error_one_step=float(errors[:, 0].max()),
                maximum_state_error_ten_steps=float(errors[:, 1].max()),
                composition_maximum=float(composition), time_step=TAU,
                passed=bool(errors.max() < 1e-9 and composition < 1e-11))


# Explicit enclosures of the quotient map

def _square(x):
    return Fraction.from_float(float(x))**2


@lru_cache(maxsize=None)
def _sqrt_enclosure(q):
    """Neighboring binary64 numbers around sqrt(q), found by exact comparisons of squares."""
    if q == 0:
        return 0., 0.
    if q < 0:
        raise ValueError("The radicand must be nonnegative.")
    trial = sqrt(float(q))
    if _square(trial) == q:
        return trial, trial
    if _square(trial) < q:
        lower = trial
        upper = nextafter(lower, inf)
        while _square(upper) < q:
            lower, upper = upper, nextafter(upper, inf)
        if _square(upper) == q:
            return upper, upper
    else:
        upper = trial
        lower = nextafter(upper, 0.)
        while _square(lower) > q:
            upper, lower = lower, nextafter(lower, 0.)
        if _square(lower) == q:
            return lower, lower
    assert _square(lower) < q < _square(upper) and nextafter(lower, inf) == upper
    return lower, upper


def _scalar(a):
    magnitude = _sqrt_enclosure(9*a**2/(1 + 8*a**2))
    return (-magnitude[1], -magnitude[0]) if a < 0 else magnitude


def _radial(abs_a, r):
    d = 1 + 8*abs_a**2
    return _sqrt_enclosure(9*r**2/(d*(d + 8*r**2)))


@lru_cache(maxsize=None)
def _box_image(box):
    a_lo, r_lo, a_hi, r_hi = map(Fraction.from_float, map(float, box))
    if not (-Fraction(3, 2) <= a_lo <= a_hi <= Fraction(3, 2) and 0 <= r_lo <= r_hi <= Fraction(3, 2)):
        raise ValueError("Boxes must lie in [-3/2, 3/2] x [0, 3/2].")
    abs_max = max(abs(a_lo), abs(a_hi))
    abs_min = 0 if a_lo <= 0 <= a_hi else min(abs(a_lo), abs(a_hi))
    # The first coordinate increases with a; the second increases with r and decreases with |a|.
    return (_scalar(a_lo)[0], _radial(abs_max, r_lo)[0], _scalar(a_hi)[1], _radial(abs_min, r_hi)[1])


def reference_images(boxes):
    """Binary64 boxes containing the images of olF on the boxes (a_lo, r_lo, a_hi, r_hi).

    Each endpoint is a binary64 neighbor of an exact square root, found by comparing
    rational squares, so the boxes contain the exact images.
    """
    boxes = np.asarray(boxes, float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all() or np.any(boxes[:, :2] > boxes[:, 2:]):
        raise ValueError("Boxes are finite rows (a_lo, r_lo, a_hi, r_hi).")
    return np.asarray([_box_image(tuple(box)) for box in boxes], float).reshape(-1, 4)


def reference_graph(depth=6):
    """The Morse graph of olF on 2^depth x 2^depth cells of X/G, from explicit enclosures."""
    boxes = grids.cells(2**depth, PHYSICAL_LOWER, PHYSICAL_UPPER)
    graph = grids.morse_graph(boxes, reference_images(boxes))
    label_roles(graph, ANCHORS)
    return graph


# Data

def sample(count, seed):
    """Starts in X: uniform a, r and direction, with extra starts near the axes and the
    equilibrium orbits, starts exactly on the axes, and the four equilibrium orbits."""
    if count < 8:
        raise ValueError("Each data split needs at least eight starts.")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    a, r = rng.uniform(-1.5, 1.5, count), rng.uniform(0, 1.5, count)
    k = count//8
    a[:k] = rng.choice([-1., 1.], k)*10.**rng.uniform(-8, -1, k)
    r[k:2*k] = 10.**rng.uniform(-8, -1, k)
    centers = np.array([[0., 0.], [0., 1.], [-1., 0.], [1., 0.]])
    perturb = 10.**rng.uniform(-6, -1, (2*k, 2))*rng.choice([-1., 1.], (2*k, 2))
    near = centers[np.arange(2*k) % 4] + perturb
    a[2*k:4*k], r[2*k:4*k] = near[:, 0], np.abs(near[:, 1])
    axis = np.arange(4*k, 4*k + max(4, k//4))
    a[axis[::2]] = 0.
    r[axis[1::2]] = 0.
    a[-4:], r[-4:] = centers[:, 0], centers[:, 1]
    return np.column_stack((a, r[:, None]*directions))


def trajectories(starts):
    """The orbits of the starts at times 0, tau, ..., 10 tau, of shape (count, 11, 4)."""
    return np.stack([physical_step(starts, k) for k in range(HORIZONS)], axis=1)


def generate_data(folder, settings=DEFAULTS):
    """Write `trajectories.npz` (train_x, validation_x, test_x) and `metadata.json`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sizes, seeds = settings["data"]["sizes"], settings["data"]["split_seeds"]
    values = {f"{name}_x": trajectories(sample(count, seed))
              for name, count, seed in zip(SPLITS, sizes, seeds, strict=True)}
    path = folder/"trajectories.npz"
    np.savez_compressed(path, **values)
    training.write(folder/"metadata.json", dict(
        sizes=list(sizes), split_seeds=list(seeds), coordinates=["a", "v1", "v2", "v3"], time_step=TAU,
        horizons=list(range(HORIZONS)), training_horizons=[0, 1],
        formula="closed-form time-log(3) map of the vector field",
        sampling="uniform a, r and direction; one eighth near a = 0, one eighth near r = 0, one quarter near "
                 "the four equilibrium orbits, starts on the axes, and the four equilibrium orbits last",
        sha256=training.digest(path),
        array_sha256={key: _array_digest(value) for key, value in values.items()}))
    return values


def _array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_data(folder):
    """The three splits of `folder/trajectories.npz`, checked, as a dict of float64 arrays."""
    path = Path(folder)
    path = path/"trajectories.npz" if path.is_dir() else path
    with np.load(path, allow_pickle=False) as saved:
        data = {f"{name}_x": saved[f"{name}_x"] for name in SPLITS}
    for name, values in data.items():
        if values.ndim != 3 or values.shape[1:] != (HORIZONS, 4) or len(values) < 8:
            raise ValueError(f"{name} must have shape (count, 11, 4), with at least eight starts.")
        if values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("Use finite float64 states.")
        if np.max(np.abs(values[..., 0])) > 1.5 + 1e-12 or np.max(np.linalg.norm(values[..., 1:], axis=-1)) > 1.5 + 1e-12:
            raise ValueError("States must lie in X.")
    return data


# Networks and objective

class InvariantDynamics(objax.Module):
    """Gain corrections, by O(3) EMLPs, around the orbit coordinates.

    E(x) = (a, |v|) * exp(0.2 tanh e(x/1.5)),
    d(u) = clip(u * exp(0.35 tanh A(u/2))) to [-1.5, 1.5] x [0, 1.5],
    f(u) = 2 tanh(u * exp(3 tanh B(u/2)) / 2),
    with coordinatewise products. E is invariant because e is, and f and d preserve
    the sign of each coordinate and the axes a = 0 and r = 0.
    """

    def __init__(self, width=32, layers=2, recipe=None):
        if width < 2 or width % 2:
            raise ValueError("The width must be a positive even integer.")
        self.recipe = dict(DEFAULTS["recipe"] if recipe is None else recipe)
        group = O(3)
        hidden = (width//2)*T(0) + (width//2)*V
        self.encoder = EMLP(T(0) + V, 2*T(0), group, ch=hidden, num_layers=layers)
        self.decoder = EMLP(2*T(0), 2*T(0), group, ch=width*T(0), num_layers=layers)
        self.dynamics = EMLP(2*T(0), 2*T(0), group, ch=width*T(0), num_layers=layers)

    def encode(self, x):
        ar = jnp.stack((x[..., 0], jnp.linalg.norm(x[..., 1:], axis=-1)), axis=-1)
        return ar*jnp.exp(self.recipe["encoder_gain_log_bound"]*jnp.tanh(self.encoder(x/1.5)))

    def decode(self, z):
        values = z*jnp.exp(self.recipe["decoder_gain_log_bound"]*jnp.tanh(self.decoder(z/2.)))
        return jnp.clip(values, jnp.array([-1.5, 0.]), jnp.array([1.5, 1.5]))

    def advance(self, z):
        values = z*jnp.exp(self.recipe["time_gain_log_bound"]*jnp.tanh(self.dynamics(z/2.)))
        return 2*jnp.tanh(values/2.)


def _orbit(x):
    return jnp.stack((x[..., 0], jnp.linalg.norm(x[..., 1:], axis=-1)), axis=-1)


def loss_parts(model, x, y):
    """Reconstruction, prediction and consistency for pairs y = F(x), as three numbers."""
    z, z_next = model.encode(x), model.encode(y)
    predicted = model.advance(z)
    reconstruction = jnp.mean(jnp.sum((model.decode(z) - _orbit(x))**2, axis=-1))
    prediction = jnp.mean(jnp.sum((model.decode(predicted) - _orbit(y))**2, axis=-1))
    consistency = jnp.mean(jnp.sum((predicted - z_next)**2, axis=-1))
    return jnp.stack((reconstruction, prediction, consistency))


def evaluator(model, method):
    return objax.Jit(objax.Function.with_vars(model.vars())(method))


def batched(function, values, size=8192):
    values = np.asarray(values)
    return np.concatenate([np.asarray(function(jnp.asarray(values[i:i + size]))) for i in range(0, len(values), size)])


_COMPILED = weakref.WeakKeyDictionary()


def functions(model):
    """Compiled E, d and f, each applied in batches of 8192 to arrays."""
    if model not in _COMPILED:
        encode, decode, advance = [evaluator(model, getattr(model, name)) for name in ("encode", "decode", "advance")]
        _COMPILED[model] = ((lambda x: batched(encode, x)), (lambda z: batched(decode, z)),
                            (lambda z: batched(advance, z)))
    return _COMPILED[model]


def validation_objective(model, data):
    parts = evaluator(model, lambda x, y: loss_parts(model, x, y))
    values = data["validation_x"]
    return float(np.asarray(parts(values[:, 0], values[:, 1])).sum())


# Training

def train(folder, data, seed, settings=DEFAULTS, progress=None):
    """Fit E, d and f jointly from the pairs at times 0 and tau, keeping the weights of
    the lowest validation objective."""
    folder = Path(folder)
    if (folder / "weights.npz").exists():
        raise FileExistsError(folder / "weights.npz")
    folder.mkdir(parents=True, exist_ok=True)  # a progress record may already be there
    options, recipe = settings["training"], settings["recipe"]
    steps, batch, every = options["steps"], options["batch"], options["validate_every"]
    np.random.seed(seed)
    objax.random.DEFAULT_GENERATOR.seed(seed)
    model = InvariantDynamics(options["width"], options["layers"], recipe)
    initial = {name: [np.asarray(v.value).copy() for v in getattr(model, name).vars().values()] for name in NETWORKS}
    optimizer = objax.optimizer.Adam(model.vars())
    gradient = objax.GradValues(lambda x, y: jnp.sum(loss_parts(model, x, y)), model.vars())

    @objax.Function.with_vars(model.vars() + optimizer.vars())
    def update(x, y, rate):
        gradients, value = gradient(x, y)
        optimizer(rate, gradients)
        return value[0]

    update = objax.Jit(update)
    validate = evaluator(model, lambda x, y: loss_parts(model, x, y))
    inputs = {name: [jnp.asarray(data[f"{name}_x"][:, i]) for i in (0, 1)] for name in ("train", "validation")}
    parts = np.asarray(validate(*inputs["validation"]))
    best, best_step = float(parts.sum()), 0
    training.save_weights(folder/"weights.npz", model.vars())
    history = [dict(step=0, validation=best, parts=parts.tolist())]
    start = time.perf_counter()
    rng = np.random.default_rng(seed + 200)
    if progress is not None:
        progress.update(status="training", step=0, validation=best, seconds=0.)
    for step in range(1, steps + 1):
        ids = rng.integers(0, len(inputs["train"][0]), batch)
        rate = training.cosine_rate(step, steps, options["learning_rate_start"], options["learning_rate_end"])
        value = float(update(*(x[ids] for x in inputs["train"]), rate))
        if not np.isfinite(value):
            if progress is not None:
                progress.update(status="failed", step=step)
            raise FloatingPointError("The training objective is not finite.")
        if step == 1 or step % every == 0 or step == steps:
            parts = np.asarray(validate(*inputs["validation"]))
            score = float(parts.sum())
            history.append(dict(step=step, training=value, validation=score, parts=parts.tolist()))
            if score < best:
                best, best_step = score, step
                training.save_weights(folder/"weights.npz", model.vars())
            if progress is not None:
                progress.update(status="training", step=step, validation=score, seconds=time.perf_counter() - start)
    seconds = time.perf_counter() - start
    training.load_weights(folder/"weights.npz", model.vars())
    changes = {name: float(np.sqrt(sum(np.sum((np.asarray(v.value) - old)**2)
                                       for v, old in zip(getattr(model, name).vars().values(), initial[name], strict=True))))
               for name in NETWORKS}
    training.write(folder/"config.json", dict(
        seed=seed, best_step=best_step, best_validation=best, seconds=seconds, settings=settings,
        versions={name: version(name) for name in ("emlp", "objax", "jax", "numpy", "scipy", "cmgdb")},
        parameters={name: int(sum(v.value.size for v in getattr(model, name).vars().values())) for name in NETWORKS},
        parameter_changes=changes, time_step=TAU, weights_format="numeric-ordered-v1",
        weights_sha256=training.digest(folder/"weights.npz")))
    training.write(folder/"history.json", history)
    if progress is not None:
        progress.update(status="trained", step=steps, validation=best, seconds=seconds)
    return model


def load(folder):
    """The model saved in `folder` by `train`, or by the earlier training code."""
    folder = Path(folder)
    config = training.read(folder/"config.json")
    if "settings" in config:
        width, layers = config["settings"]["training"]["width"], config["settings"]["training"]["layers"]
        recipe = config["settings"]["recipe"]
    else:
        width, layers, recipe = config["width"], config["layers"], config.get("recipe")
    if training.digest(folder/"weights.npz") != config["weights_sha256"]:
        raise ValueError("The weights differ from those recorded in config.json.")
    np.random.seed(config["seed"])
    objax.random.DEFAULT_GENERATOR.seed(config["seed"])
    model = InvariantDynamics(width, layers, recipe)
    training.load_weights(folder/"weights.npz", model.vars())
    return model


# Diagnostics

def encoded_anchors(model):
    """E at the four equilibrium orbits O, C, A_-, A_+, as a dict of latent points."""
    encode, _, _ = functions(model)
    values = encode(section(CONTROLS))
    return {name: value.tolist() for name, value in zip(("O", "C", "A-", "A+"), values, strict=True)}


def diagnostics(model, data):
    """Orbit errors of 0 to 10 learned steps on the test set, the sampled invariance,
    the one-step residual, sampled Jacobians of E on the section, encoded equilibria,
    fixed points of f on the axes, and the validation objective."""
    encode, decode, advance = functions(model)
    x = data["test_x"]
    codes = encode(x[:, 0])
    predicted = codes.copy()
    forecasts = {}
    for horizon in range(HORIZONS):
        error = np.linalg.norm(decode(predicted) - orbit(x[:, horizon]), axis=1)
        forecasts[str(horizon)] = dict(orbit_rms=float(np.sqrt(np.mean(error**2))), orbit_maximum=float(error.max()))
        if horizon < HORIZONS - 1:
            predicted = advance(predicted)
    residual = advance(codes) - encode(x[:, 1])
    rng = np.random.default_rng(1929)
    invariance = []
    for reflection in (False, True):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix[:, 0] *= np.linalg.det(matrix)*(-1 if reflection else 1)
        transformed = x[:, 0].copy()
        transformed[:, 1:] = transformed[:, 1:]@matrix.T
        invariance.append(float(np.max(np.abs(encode(transformed) - codes))))
    a, r = np.linspace(-1.5, 1.5, 257), np.linspace(0, 1.5, 257)
    aa, rr = np.meshgrid(a, r, indexing="ij")
    ar = np.column_stack((aa.ravel(), rr.ravel()))
    encoded_grid = encode(section(ar))
    # The Jacobian on the section, away from r = 0, where the norm is not differentiable.
    derivative = evaluator(model, jax.vmap(jax.jacfwd(lambda t: model.encode(jnp.array([[t[0], t[1], 0., 0.]]))[0])))
    check_grid = ar.reshape(257, 257, 2)[::8, 1::8].reshape(-1, 2)
    jacobian = batched(derivative, check_grid)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    determinant = np.linalg.det(jacobian)
    support_codes = np.concatenate([encode(data[f"{name}_x"].reshape(-1, 4)) for name in ("train", "validation")])
    anchors = encode(section(np.array([[0., 0.], [0., 1.], [-1., 0.], [1., 0.]])))
    # Fixed points of f on the axes, searched over the ranges of the encoded axes.
    scalar_axis, radial_axis = encoded_grid[ar[:, 1] == 0, 0], encoded_grid[ar[:, 0] == 0, 1]
    roots = [dict(kind="O", latent=[0., 0.])]
    for coordinate, intervals, kind in ((0, [(float(scalar_axis.min()), -1e-7), (1e-7, float(scalar_axis.max()))], "scalar"),
                                        (1, [(1e-7, float(radial_axis.max()))], "radial")):
        for left, right in intervals:
            sweep = np.linspace(left, right, 4001)
            points = np.zeros((len(sweep), 2))
            points[:, coordinate] = sweep
            displacement = advance(points)[:, coordinate] - sweep
            for index in np.flatnonzero(displacement[:-1]*displacement[1:] < 0):
                def scalar(value):
                    z = np.zeros((1, 2))
                    z[0, coordinate] = value
                    return float(advance(z)[0, coordinate] - value)
                root = brentq(scalar, sweep[index], sweep[index + 1], xtol=1e-13)
                z = np.zeros(2)
                z[coordinate] = root
                roots.append(dict(kind=kind, latent=z.tolist()))
    for row, decoded in zip(roots, decode(np.array([row["latent"] for row in roots])), strict=True):
        row["decoded_orbit"] = decoded.tolist()
    return dict(
        forecasts=forecasts, sampled_encoder_invariance_error=max(invariance),
        invariance_by_rotation_and_reflection=invariance,
        residual_rms=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        residual_maximum=float(np.linalg.norm(residual, axis=1).max()),
        minimum_sampled_encoder_singular_value=float(singular_values.min()),
        minimum_sampled_encoder_jacobian_determinant=float(determinant.min()),
        maximum_sampled_encoder_jacobian_determinant=float(determinant.max()),
        physical_anchor_names=["O", "C", "A-", "A+"], encoded_anchors=anchors.tolist(),
        anchor_time_residuals=np.linalg.norm(advance(anchors) - anchors, axis=1).tolist(),
        observed_axis_fixed_points=roots,
        support_code_range=[support_codes.min(axis=0).tolist(), support_codes.max(axis=0).tolist()],
        support_source="encoded training and validation trajectories, horizons 0 through 10",
        support_samples=len(support_codes), test_code_covariance=np.cov(codes.T).tolist(),
        validation_objective=validation_objective(model, data),
        evidence_status="sampled numerical diagnostics; no global injectivity or transfer certificate")


# Morse graphs of the learned map

def label_roles(graph, positions, radius=4):
    """Give each Morse set the reference invariant set whose region it lies in.

    `positions` gives the points of O, C, A_- and A_+ in the grid coordinates. The region
    of one of them is the set of points nearer to it than to the other three, with
    distances in units of the cell side along each axis, and within `radius` cells of
    it. A Morse set lies in a region when the centers of all its cells do. Each node
    row gets a key "role", the name or None.
    """
    k = graph.boxes.shape[1]//2
    side = graph.boxes[0, k:] - graph.boxes[0, :k]
    names = list(positions)
    points = np.asarray([positions[name] for name in names], float)
    for row in graph.nodes:
        ids = row["cell_ids"]
        centers = (graph.boxes[ids, :k] + graph.boxes[ids, k:])/2
        distance = np.linalg.norm((centers[:, None, :] - points[None])/side, axis=-1)
        nearest = distance.argmin(axis=1)
        inside = len(set(nearest.tolist())) == 1 and distance[np.arange(len(ids)), nearest].max() < radius
        row["role"] = names[nearest[0]] if inside else None
        row["distance_cells"] = float(distance[np.arange(len(ids)), nearest].max())
    return graph


def groups(graph):
    """The Morse sets of each reference invariant set, after `label_roles`.

    Raises ValueError when a Morse set lies in no region or a reference set has no
    Morse set, since then the structure of the example does not determine a grouping.
    """
    found = {name: [row["node"] for row in graph.nodes if row["role"] == name] for name in ANCHORS}
    loose = [row["node"] for row in graph.nodes if row["role"] is None]
    if loose:
        raise ValueError(f"The Morse sets {loose} lie in the region of no reference invariant set.")
    missing = [name for name, nodes in found.items() if not nodes]
    if missing:
        raise ValueError(f"No Morse set lies in the region of {missing}.")
    return found


def merge(graph):
    """Merge the Morse sets of each reference invariant set with all connecting cells.

    The index of each merged set is recomputed by CMGDB on its cells (grids.merge),
    which raises ValueError if a group is not order convex or does not give an index
    pair. Node rows of the result carry their role.
    """
    found = groups(graph)
    merged = grids.merge(graph, list(found.values()))
    role_of = {node: name for name, nodes in found.items() for node in nodes}
    for row in merged.nodes:
        row["role"] = role_of[row["original_nodes"][0]]
    return merged


def source_pairs(model, size):
    """The pairs (E(x), f(E(x))) for x on the section (a, r, 0, 0), a uniform grid of
    size x size points of X/G, followed by the four equilibrium orbits."""
    encode, _, advance = functions(model)
    a, r = np.linspace(-1.5, 1.5, size), np.linspace(0., 1.5, size)
    aa, rr = np.meshgrid(a, r, indexing="ij")
    codes = encode(section(np.column_stack((aa.ravel(), rr.ravel()))))
    controls = encode(section(CONTROLS))
    sources = np.concatenate((codes, controls))
    images = np.concatenate((advance(codes), advance(controls)))
    if not np.all((sources >= LOWER) & (sources <= UPPER)) or not np.isfinite(images).all():
        raise ValueError("An encoded source lies outside Z.")
    return sources, images


def data_map_graph(model, settings=DEFAULTS):
    """The Morse graph of the data map from (E(x), f(E(x))) on the paper's grid, with
    the roles of its Morse sets; cells without sources map outside Z."""
    analysis = settings["analysis"]
    sources, images = source_pairs(model, analysis["data_map"]["primary_source_size"])
    boxes = grids.cells(2**analysis["primary_depth"], LOWER, UPPER)
    graph = grids.morse_graph(boxes, data_map=grids.data_map(sources, images, LOWER, UPPER))
    return label_roles(graph, encoded_anchors(model), analysis["sampled_hulls"]["group_radius_cells"])


def learned_graph(model, settings=DEFAULTS):
    """The data-map Morse graph of Section 5.2 and, when a reference invariant set has
    several Morse sets, the graph merged by `merge`; otherwise the merged graph is None."""
    graph = data_map_graph(model, settings)
    if all(sum(row["role"] == name for row in graph.nodes) == 1 for name in ANCHORS) and len(graph.nodes) == 4:
        return graph, None
    return graph, merge(graph)


def hull_graph(model, settings=DEFAULTS):
    """The Morse graph of the sampled image hulls of f on 2^6 x 2^6 cells of Z.

    Each hull contains the images of 5 x 5 points of its cell, corners included, is
    padded by 1e-9 and rounded outward, and is cut at r = 0.
    """
    options = settings["analysis"]["sampled_hulls"]
    _, _, advance = functions(model)
    boxes = grids.cells(2**options["depth"], LOWER, UPPER)
    images = grids.sampled_hulls(advance, boxes, options["samples_per_axis"], options["padding"],
                                 lower=[-np.inf, 0.])
    graph = grids.morse_graph(boxes, images)
    return label_roles(graph, encoded_anchors(model), options["group_radius_cells"])


def learned_hull_graph(model, settings=DEFAULTS):
    """The sampled-hull Morse graph of f and its merged graph, as in the quotient graph
    of the figure of Section 5.2: each Morse set grouped with the reference invariant set
    whose region it lies in, with all connecting cells and indices recomputed by CMGDB."""
    graph = hull_graph(model, settings)
    return graph, merge(graph)


def compare_with_reference(graph, reference):
    """Whether two graphs with roles have the same order and annotations of O, C, A_±."""
    def summary(g):
        role = {row["node"]: row["role"] for row in g.nodes}
        if None in role.values() or len(set(role.values())) != len(role):
            return None
        return ({name: row["annotations"] for row in g.nodes for name in [role[row["node"]]]},
                sorted((role[a], role[b]) for a, b in g.edges))
    ours, theirs = summary(graph), summary(reference)
    return dict(comparable=ours is not None and theirs is not None,
                same_order=ours is not None and theirs is not None and ours[1] == theirs[1],
                same_annotations=ours is not None and theirs is not None and ours[0] == theirs[0],
                annotations=None if ours is None else ours[0], edges=None if ours is None else ours[1])


def supported(graph):
    """The cells whose image lies in the closure of Z; a data-map cell without sources maps beyond it."""
    return ~np.all(graph.images[:, :2] > UPPER, axis=1)


def basins(graph):
    """The attracting regions of the minimal Morse sets and the cells on paths from O to C.

    Returns the region label of each cell (`regions.attracting_regions`, -1 for none)
    and a mask of the cells on a path of the outer approximation from a Morse set of O
    to one of C.
    """
    labels = regions.attracting_regions(graph, FACES, available=supported(graph))
    source = [node for row in graph.nodes for node in row.get("original_nodes", [row["node"]]) if row["role"] == "O"]
    target = [node for row in graph.nodes for node in row.get("original_nodes", [row["node"]]) if row["role"] == "C"]
    path = np.zeros(len(graph.boxes), bool)
    if source and target:
        native = np.asarray(CMGDB.MorseDirectedPathCells(graph.phase, graph.morse, source, target), dtype=np.int64)
        path[np.argsort(graph.order)[native]] = True
    return dict(labels=labels, path=path)


def physical_preimages(model, graph, labels=None, points=None, settings=DEFAULTS):
    """The latent cell of E(a, r, 0, 0) on a uniform grid of X/G, with its Morse set and region.

    A point is assigned to the cell found by flooring its coordinates, as in the panels.
    """
    points = points or settings["analysis"]["physical_section_points_per_axis"]
    encode, _, _ = functions(model)
    a, r = np.linspace(-1.5, 1.5, points), np.linspace(0., 1.5, points)
    aa, rr = np.meshgrid(a, r, indexing="ij")
    codes = encode(section(np.column_stack((aa.ravel(), rr.ravel()))))
    n = round(np.sqrt(len(graph.boxes)))
    index = np.clip(np.floor((codes - LOWER)/(UPPER - LOWER)*n).astype(int), 0, n - 1)
    cell = index[:, 0]*n + index[:, 1]
    shape = (points, points)
    result = dict(a=a, r=r, codes=codes.reshape(*shape, 2), cell=cell.reshape(shape),
                  node=graph.membership[cell].reshape(shape))
    if labels is not None:
        result["region"] = np.asarray(labels)[cell].reshape(shape)
    return result


# Attracting regions

def residual_samples(count=1024):
    """Section points of X/G for the residual: count x count cell centers, points near
    a = 0, the axis r = 0, and the four equilibrium orbits."""
    a, r = -1.5 + 3*(np.arange(count) + .5)/count, 1.5*(np.arange(count) + .5)/count
    aa, rr = np.meshgrid(a, r, indexing="ij")
    grid = np.column_stack((aa.reshape(-1), rr.reshape(-1)))
    near = np.geomspace(1e-10, .1, min(193, max(3, count//4)))
    axis_a = np.r_[-near[::-1], 0., near]
    aa, rr = np.meshgrid(axis_a, np.linspace(0., 1.5, min(257, count + 1)), indexing="ij")
    edge = min(4097, 4*count + 1)
    return np.concatenate((grid, np.column_stack((aa.reshape(-1), rr.reshape(-1))),
                           np.column_stack((np.linspace(-1.5, 1.5, edge), np.zeros(edge))),
                           np.array([[-1., 0.], [1., 0.], [0., 1.], [0., 0.]])))


def attracting_regions(model, graph, settings=DEFAULTS):
    """For each minimal Morse set of the graph, its region, tolerance and residual maximum.

    The residual |f(E(x)) - E(F(x))| is sampled on `residual_samples` and the metric is
    the Euclidean metric of Z, at map time log(3). Returns the rows of
    `regions.compare`, the region labels and the residuals.
    """
    encode, _, advance = functions(model)
    physical = section(residual_samples(settings["analysis"]["tolerances"]["samples_per_axis"]))
    codes = encode(physical)
    predicted, encoded_image = advance(codes), encode(physical_step(physical))
    labels = regions.attracting_regions(graph, FACES, available=supported(graph))
    rows, residual = regions.compare(graph, labels, codes, predicted, encoded_image, regions.Euclidean(), FACES)
    return rows, labels, residual


def settings_with(**changes):
    """A copy of DEFAULTS with nested changes named by dotted paths, such as
    settings_with(**{"analysis.data_map.primary_source_size": 513})."""
    settings = copy.deepcopy(DEFAULTS)
    for path, value in changes.items():
        *parents, last = path.split(".")
        target = settings
        for key in parents:
            target = target[key]
        target[last] = value
    return settings


def attracting_regions_of(model, graphs, settings=DEFAULTS):
    """`attracting_regions` for several graphs of `model`, with the residual sampled once.

    Returns {name: dict(rows, labels)} and the number of samples.
    """
    encode, _, advance = functions(model)
    physical = section(residual_samples(settings["analysis"]["tolerances"]["samples_per_axis"]))
    codes = encode(physical)
    predicted, encoded_image = advance(codes), encode(physical_step(physical))
    result = {}
    for name, graph in graphs.items():
        labels = regions.attracting_regions(graph, FACES, available=supported(graph))
        rows, _ = regions.compare(graph, labels, codes, predicted, encoded_image, regions.Euclidean(), FACES)
        result[name] = dict(rows=rows, labels=labels)
    return result, len(codes)


# Roles, comparison with the reference, and indices in X.

ORDER = ("O", "C", "A-", "A+")
TEX = {"O": "$O$", "C": "$C$", "A-": "$A_-$", "A+": "$A_+$"}
FIXED = [(1, "lower")]  # P/G = {r = 0}, the face of the rectangle where v = 0
ORBIT_SPHERE = 2        # the orbits G/H = O(3)/O(2) of the points with v != 0 are spheres S^2
# The indices in X proved on explicit index pairs (Supplement supp:o3-pairs), as CMGDB shift classes
# in degrees 0 to 4: H_4 = F at O, H_1 = H_3 = F at C and H_0 = F at A_-+, with identity index maps.
PROVED = {"O": ["0", "0", "0", "0", "x-1"], "C": ["0", "x-1", "0", "x-1", "0"],
          "A-": ["x-1", "0", "0", "0", "0"], "A+": ["x-1", "0", "0", "0", "0"]}


def node_roles(graph):
    """{node: role}, the roles given by `label_roles` or carried by a merged graph."""
    return {row["node"]: row["role"] for row in graph.nodes}


def _unique_roles(graph):
    roles = node_roles(graph)
    if None in roles.values() or len(set(roles.values())) != len(roles):
        raise ValueError("Each Morse set needs its own role; use the merged graph.")
    return roles


def role_edges(graph):
    """The edges of a Morse graph as pairs of roles, in the order of ORDER."""
    roles = _unique_roles(graph)
    return sorted(([roles[a], roles[b]] for a, b in graph.edges), key=lambda e: (ORDER.index(e[0]), ORDER.index(e[1])))


def compare_graphs(reference, learned):
    """Compare a learned Morse graph with the reference graph role by role.

    The graphs agree when they have the same roles, the same Conley index annotation at
    each role and the same edges between roles; the order alone is compared by
    `edges_agree`.
    """
    ref_roles, learned_roles = _unique_roles(reference), _unique_roles(learned)
    ref = {ref_roles[row["node"]]: row for row in reference.nodes}
    new = {learned_roles[row["node"]]: row for row in learned.nodes}
    rows = []
    for role in ORDER:
        a, b = ref.get(role), new.get(role)
        rows.append(dict(role=role, reference_cells=a["cell_ids"] if a else None,
                         reference_annotations=a["annotations"] if a else None,
                         learned_cells=b["cell_ids"] if b else None,
                         learned_annotations=b["annotations"] if b else None,
                         learned_morse_sets=(b.get("original_nodes", [b["node"]]) if b else None),
                         agree=bool(a and b and a["annotations"] == b["annotations"])))
    edges_agree = role_edges(reference) == role_edges(learned)
    return dict(nodes=rows, reference_edges=role_edges(reference), learned_edges=role_edges(learned),
                edges_agree=edges_agree, agree=bool(edges_agree and all(row["agree"] for row in rows)))


def reconstructed_indices(graphs):
    """The Conley indices in X of the Morse sets of each graph, by Theorem thm:sphere-reconstruction.

    `graphs` maps a name to a MorseGraph on a uniform grid of a rectangle, the reference
    graph on X/G or a learned graph on Z, whose cells are product cells with the axis of r
    (or of u_2) last. For each Morse set, `homology.reconstructed_index` evaluates the
    theorem on the index pair (|M u F(M)|, |F(M) minus M|) of the outer approximation F,
    with P/G the face r = 0, G/H = S^2 (m = 2) and delta the identity (degree 1). The
    quotient index it computes is checked against CMGDB's annotation. Returns
    {name: {role: record}} with the shift classes of the quotient index, of the index
    relative to N_P/G and of the index in X, in degrees 0 to 4.
    """
    result = {}
    for name, graph in graphs.items():
        roles = _unique_roles(graph)
        side = round(np.sqrt(len(graph.boxes)))
        grid = homology.Grid((side, side))
        outer = homology.BoxMap.from_relation(grid, graph.offsets, graph.targets)
        result[name] = {}
        for row in sorted(graph.nodes, key=lambda row: ORDER.index(roles[row["node"]])):
            answer = homology.reconstructed_index(outer, row["cell_ids"], fixed=FIXED, m=ORBIT_SPHERE)
            quotient = answer.quotient.shift_class()
            if quotient != list(row["annotations"]):
                raise RuntimeError(f"{name} {roles[row['node']]}: the quotient index {quotient} differs from "
                                   f"CMGDB's annotation {row['annotations']}.")
            result[name][roles[row["node"]]] = dict(
                cells=list(row["cell_ids"]), index_pair_cells=grid.ids(answer.N), exit_cells=grid.ids(answer.L),
                fixed_point_cells=len(answer.N_P), quotient=quotient, relative=answer.relative.shift_class(),
                in_X=answer.lifted.shift_class(), proved=PROVED[roles[row["node"]]],
                agree=answer.lifted.shift_class() == PROVED[roles[row["node"]]])
    return result


def homology_degrees(index):
    """The degrees in which a shift class is nonzero, as in 'H_1 = H_3 = F'."""
    degrees = [str(k) for k, value in enumerate(index) if value != "0"]
    return " = ".join(f"H_{k}" for k in degrees) + " = F" if degrees else "trivial"


# Figures. Each panel is bare, saved as PDF and SVG by `plotting`, under the manuscript's name.

GRAPH_SCALE = .52        # the graphs of fig:o3 are included at scale .52
PANEL = (2.6, 1.75)      # the first row of fig:o3, printed at a height of about 1.3 in
OUTSIDE = plotting.mix(plotting.GRAY, .3)  # latent cells that contain no encoded section point
# Orbits of (a0, r0 e1) for the learned dynamics: on r = 0 from O to A_-+, on a = 0 from O to C, next
# to a = 0 up to C and then to A_-+, and on rays out of O.
ORBIT_STARTS = ((.02, 0.), (-.02, 0.), (0., .02), (3e-4, .02), (-3e-4, .02),
                (.02*np.cos(np.pi/6), .02*np.sin(np.pi/6)), (-.02*np.cos(np.pi/6), .02*np.sin(np.pi/6)),
                (.02*np.cos(np.pi/3), .02*np.sin(np.pi/3)), (-.02*np.cos(np.pi/3), .02*np.sin(np.pi/3)))
OFFSETS = {"O": (0, -3, "top"), "C": (0, 3, "bottom"), "A-": (0, -3, "top"), "A+": (0, -3, "top")}  # points


def quotient_flow(a, r, t):
    """The quotient flow from (a, r) at the times t, in closed form: a^2 and a^2 + r^2 follow
    y' = 2 y (1 - y), so with s = exp(2t), a(t) = sqrt(s) a / sqrt(D) and
    r(t) = sqrt(s) r / sqrt(D (D + (s - 1) r^2)) for D = 1 + (s - 1) a^2."""
    s = np.exp(2*np.asarray(t, float))
    d = 1 + (s - 1)*a*a
    return np.sqrt(s)*a/np.sqrt(d), np.sqrt(s)*r/np.sqrt(d*(d + (s - 1)*r*r))


def _axes(ax, xlabel, ylabel):
    ax.set_xticks([-1, 0, 1], ["$-1$", "$0$", "$1$"])
    ax.set_yticks([0, 1], ["$0$", "$1$"])
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.tick_params(width=.6, length=2.5, pad=2)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.6)


def _arrow_at(ax, points, fraction):
    """An arrowhead at the given fraction of the arc length of a curve."""
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s = fraction*length[-1]
    tail, head = ([np.interp(t, length, points[:, k]) for k in (0, 1)] for t in (s - .045, s))
    ax.annotate("", xy=head, xytext=tail, zorder=2, arrowprops=dict(
        arrowstyle="-|>", lw=.8, color=plotting.GRAY, mutation_scale=8, shrinkA=0, shrinkB=0))


def _equilibrium(ax, role, point, filled):
    ink = plotting.color(role)
    ax.plot(*point, "o", ms=4.2, mec=ink, mew=1., mfc=ink if filled else plotting.WHITE, zorder=5, clip_on=False)
    dx, dy, va = OFFSETS[role]
    ax.annotate(TEX[role], point, xytext=(dx, dy), textcoords="offset points", ha="center", va=va, color=ink)


def quotient_flow_panel(path):
    """The quotient flow in (a, r): orbits from four starts in both time directions, the arcs
    of a^2 + r^2 = 1 from C to A_-+, the segments from O to A_-+ on r = 0 and from O to C on
    a = 0, each with one arrowhead, and the equilibria, filled where they attract."""
    plt = plotting.style()
    fig, ax = plt.subplots(figsize=PANEL)
    times = np.linspace(-5., 6., 700)

    def curve(a, r, fraction, lw=.85):
        ax.plot(a, r, color=plotting.GRAY, lw=lw, zorder=1)
        _arrow_at(ax, np.column_stack((a, r)), fraction)

    for sign in (-1, 1):
        for a0, r0 in ((.39, .22), (.34, .59)):
            curve(*quotient_flow(sign*a0, r0, times), .6)
        angle = np.linspace(np.pi/2, 0 if sign > 0 else np.pi, 300)
        curve(np.cos(angle), np.sin(angle), .48, 1.)
        curve(np.linspace(0, sign, 200), np.zeros(200), .55)
    curve(np.zeros(200), np.linspace(0, 1, 200), .58, 1.)
    for role in ORDER:
        _equilibrium(ax, role, ANCHORS[role], role in ("A-", "A+"))
    ax.set(aspect="equal", xlim=(-1.16, 1.16), ylim=(-.2, 1.14))
    _axes(ax, "$a$", "$r$")
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def learned_orbits(model, steps=40):
    """The latent orbits f^k(E(x0)) and the decoded orbits d(f^k(E(x0))), k = 0, ..., steps, of
    the representatives x0 = (a0, r0, 0, 0) of ORBIT_STARTS."""
    encode, decode, advance = functions(model)
    latent = [encode(section(np.array(ORBIT_STARTS)))]
    for _ in range(steps):
        latent.append(advance(latent[-1]))
    latent = np.stack(latent, axis=1)
    return latent, decode(latent.reshape(-1, 2)).reshape(latent.shape)


def fixed_point_roles(metrics, graph):
    """The fixed points of f on the latent axes (`diagnostics`), each with the role of the
    Morse set of `graph` whose cells contain it, or None."""
    index = grids.CellIndex(graph.boxes[:, :2], graph.boxes[:, 2:])
    roles = node_roles(graph)
    result = []
    for row in metrics["observed_axis_fixed_points"]:
        cells = [c for c in index.containing(np.array([row["latent"]]), slack=0)[0] if c >= 0]
        nodes = {int(graph.membership[c]) for c in cells} - {-1}
        result.append(dict(row, role=roles[nodes.pop()] if len(nodes) == 1 else None))
    return result


def _trim(points, step=2e-3):
    """An orbit up to its last step longer than `step`."""
    long = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > step)
    return points[:long[-1] + 2] if long.size else points[:2]


def _longest_arrow(ax, points, after=None):
    """An arrowhead on the longest step of an orbit, counted from the point nearest to `after`."""
    first = 0 if after is None else int(np.argmin(np.linalg.norm(points - after, axis=1)))
    steps = np.diff(points, axis=0)
    length = np.linalg.norm(steps, axis=1)
    i = first + int(np.argmax(length[first:]))
    middle = points[i] + .55*steps[i]
    ax.annotate("", xy=middle, xytext=middle - 1e-3*steps[i]/length[i], arrowprops=dict(
        arrowstyle="-|>,head_length=0.5,head_width=0.22", color=plotting.GRAPHITE, lw=0, shrinkA=0, shrinkB=0))


def orbits_panel(model, graph, metrics, path, steps=40):
    """Decoded orbits d(f^k(E(x0))) of the learned map in (a, r), with the decoded fixed points
    of f on the latent axes in the colors of the Morse sets that contain them, filled at the
    minimal Morse sets."""
    plt = plotting.style()
    _, decoded = learned_orbits(model, steps)
    fixed = fixed_point_roles(metrics, graph)
    roles, minimal = node_roles(graph), set(graph.minimal)
    attracting = {roles[node] for node in minimal}
    saddle = next((np.asarray(row["decoded_orbit"]) for row in fixed if row["role"] == "C"), None)
    fig, ax = plt.subplots(figsize=PANEL)
    for (a0, _), points in zip(ORBIT_STARTS, decoded, strict=True):
        points = _trim(points)
        ax.plot(points[:, 0], points[:, 1], color=plotting.GRAY, lw=.8, zorder=1, solid_capstyle="round")
        ax.plot(points[:, 0], points[:, 1], ls="none", marker="o", ms=1.4, mew=0, color=plotting.GRAPHITE, zorder=2)
        _longest_arrow(ax, points, saddle if 0 < abs(a0) < 1e-3 else None)
    for row in fixed:
        if row["role"] is None:
            ax.plot(*row["decoded_orbit"], "o", ms=4.2, color=plotting.UNLABELED, zorder=5)
        else:
            _equilibrium(ax, row["role"], row["decoded_orbit"], row["role"] in attracting)
    ax.set(xlim=(-1.12, 1.12), ylim=(-.17, 1.1))
    _axes(ax, "$a$", "$r$")
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def cell_colors(graph, basins_, image=None):
    """The color of each cell: the ink of the role of its Morse set, OUTSIDE where `image` (a
    mask) is False, the tint of the role of the minimal Morse set whose attracting region
    contains it, the tint of C on a path from O to C, and None (white) otherwise."""
    roles = node_roles(graph)
    labels, path = basins_["labels"], basins_["path"]
    colors = []
    for cell in range(len(graph.boxes)):
        node = graph.membership[cell]
        if node >= 0:
            colors.append(plotting.color(roles[node]))
        elif image is not None and not image[cell]:
            colors.append(OUTSIDE)
        elif labels[cell] >= 0:
            colors.append(plotting.tint(roles[labels[cell]]))
        elif path[cell]:
            colors.append(plotting.tint("C"))
        else:
            colors.append(None)
    return colors


def _cells(ax, graph, colors, xlabel, ylabel):
    """The cells in their colors, with the cells of the Morse sets outlined and labeled by role."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    plotting.rectangle_cells(ax, graph.boxes, colors, xlabel, ylabel)
    morse = np.flatnonzero(graph.membership >= 0)
    ax.add_collection(PatchCollection(
        [Rectangle(box[:2], box[2] - box[0], box[3] - box[1]) for box in graph.boxes[morse]],
        facecolors=[colors[c] for c in morse], edgecolors=plotting.GRAPHITE, linewidths=.45, zorder=3))
    roles = node_roles(graph)
    for row in graph.nodes:
        if roles[row["node"]] in OFFSETS:
            lo, hi = np.asarray(row["hull"][:2]), np.asarray(row["hull"][2:])
            # Above the cells, since the Morse sets of O and A_-+ lie on the axis u_2 = 0.
            ax.annotate(TEX[roles[row["node"]]], ((lo[0] + hi[0])/2, hi[1]), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color=plotting.color(roles[row["node"]]), zorder=4)


def cells_panel(graph, basins_, xlabel, ylabel, path):
    """The Morse sets of a graph, the attracting regions of its minimal Morse sets and the cells on
    paths from O to C, on the whole grid."""
    plt = plotting.style()
    fig, ax = plt.subplots(figsize=PANEL)
    _cells(ax, graph, cell_colors(graph, basins_), xlabel, ylabel)
    lower, upper = graph.boxes[:, :2].min(axis=0), graph.boxes[:, 2:].max(axis=0)
    margin = .03*(upper - lower)
    ax.set(xlim=(lower[0] - margin[0], upper[0] + margin[0]), ylim=(lower[1] - margin[1], upper[1] + margin[1]))
    _axes(ax, xlabel, ylabel)
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def graph_panel(graph, annotations, path, scale=GRAPH_SCALE):
    """The Morse graph with a label `role: (index)` at each node."""
    roles = _unique_roles(graph)
    nodes = {row["node"]: (plotting.label(roles[row["node"]], annotations[roles[row["node"]]]), roles[row["node"]])
             for row in sorted(graph.nodes, key=lambda row: ORDER.index(roles[row["node"]]))}
    return plotting.graph(nodes, graph.edges, path, scale)


def preimage_panel(model, graph, basins_, path, points=None):
    """Left, the cells of Z in the colors of `cell_colors`, gray where no encoded section point
    lies outside the Morse sets; right, the color of the cell that contains E(a, r, 0, 0) on a uniform grid of X/G.

    Every point (a, r) stands for its orbit {(a, v) : |v| = r}, so the right side shows the
    preimages under E of the latent regions, not decoded sets.
    """
    from matplotlib.colors import to_rgb
    plt = plotting.style()
    preimages = physical_preimages(model, graph, basins_["labels"], points)
    image = np.zeros(len(graph.boxes), bool)
    image[np.unique(preimages["cell"])] = True
    colors = cell_colors(graph, basins_, image)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.2, 1.85), gridspec_kw=dict(width_ratios=[2, 2], wspace=.3))
    _cells(left, graph, colors, "$u_1$", "$u_2$")
    _axes(left, "$u_1$", "$u_2$")
    rgb = np.array([to_rgb(value or plotting.WHITE) for value in colors])[preimages["cell"]]
    right.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", interpolation="nearest", aspect="equal",
                 extent=(PHYSICAL_LOWER[0], PHYSICAL_UPPER[0], PHYSICAL_LOWER[1], PHYSICAL_UPPER[1]))
    _axes(right, "$a$", "$r$")
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def grouping_panel(raw, path, scale=GRAPH_SCALE):
    """The raw Morse graph, each Morse set labeled by its number and the role of its group."""
    nodes = {row["node"]: (f"{row['node']}: {plotting.label(row['role'], row['annotations'])}", row["role"])
             for row in raw.nodes}
    return plotting.graph(nodes, raw.edges, path, scale)


def _panels(paths, folder):
    """{manuscript name: PDF path} for panels saved under `folder`."""
    return {path.relative_to(folder).with_suffix("").as_posix(): path for path in paths}


def figure_o3(model, metrics, merged, basins_, indices, folder):
    """fig:o3: (a) the exact quotient flow, (b) the learned dynamics on X/G, (c) the Morse sets
    and attracting regions in Z, (d) the Morse graph in Z and (e) the graph with the indices in
    X reconstructed from it. Panels (c) to (e) use the sampled-hull graph `merged` of the fit.

    Returns {panel name: PDF path}, the names those of the manuscript's panel files.
    """
    folder = Path(folder)
    annotations = {row["role"]: row["annotations"] for row in merged.nodes}
    in_x = {role: row["in_X"] for role, row in indices["learned"].items()}
    return _panels([quotient_flow_panel(folder/"dynamics/o3-quotient-flow"),
                    orbits_panel(model, merged, metrics, folder/"dynamics/learned-quotient-orbits"),
                    cells_panel(merged, basins_, "$u_1$", "$u_2$", folder/"o3/learned-morse-sets"),
                    graph_panel(merged, annotations, folder/"o3/learned-quotient"),
                    graph_panel(merged, in_x, folder/"o3/learned-reconstructed")], folder)


def figure_o3_flow(folder):
    """fig:appendix-o3-flow: the exact quotient flow in (a, r), the panel (a) of fig:o3."""
    folder = Path(folder)
    return _panels([quotient_flow_panel(folder/"dynamics/o3-quotient-flow")], folder)


def figure_o3_comparison(reference, reference_basins, merged, basins_, indices, folder):
    """fig:appendix-o3-comparison: the reference olF on X/G (left) and the learned f on Z (right).

    The rows are the Morse sets with the attracting regions and the cells on paths from O to C,
    the Morse graphs with their Conley indices in the rectangle, and the Morse graphs with the
    indices in X reconstructed from them.
    """
    folder = Path(folder)
    paths = [cells_panel(reference, reference_basins, "$a$", "$r$", folder/"o3/analytical-morse-sets"),
             cells_panel(merged, basins_, "$u_1$", "$u_2$", folder/"o3/learned-morse-sets")]
    for kind, key in (("quotient", "quotient"), ("reconstructed", "in_X")):
        for name, graph, source in (("analytical", reference, "reference"), ("learned", merged, "learned")):
            labels = {role: row[key] for role, row in indices[source].items()}
            paths.append(graph_panel(graph, labels, folder/f"o3/{name}-{kind}"))
    return _panels(paths, folder)


def figure_preimages(model, raw, merged, basins_, folder):
    """The O(3) panels of fig:appendix-notebook-preimages: the regions in Z and their preimages in
    X/G, and the raw Morse graph with its grouping when some reference set has several Morse sets."""
    folder = Path(folder)
    paths = [preimage_panel(model, merged, basins_, folder/"notebook-preimages/o3-preimage-panels")]
    if any(len(row["original_nodes"]) > 1 for row in merged.nodes):
        paths.append(grouping_panel(raw, folder/"notebook-preimages/o3-grouping-graphs"))
    return _panels(paths, folder)


# The record of the results checked in replay.

EXAMPLE = "ex2_o3_flow"
ROOT = Path(__file__).resolve().parents[1]
EXACT = ("fit", "graphs", "comparison", "basins", "indices")  # compared exactly; other numbers to rtol 1e-9
VERSIONS = ("emlp", "objax", "jax", "numpy", "scipy", "cmgdb")


def _basins_record(basins_):
    return dict(labels=[int(v) for v in basins_["labels"]], path=[int(v) for v in np.flatnonzero(basins_["path"])])


def record_of(fit, metrics, reference, data_map, hull, merged, region_results, samples, basins_, indices):
    """The JSON-ready record of every result the notebook checks, from computed parts.

    `metrics` is `diagnostics`; the graphs come from `reference_graph`, `learned_graph` and
    `learned_hull_graph`; `region_results` and `samples` from `attracting_regions_of`;
    `basins_` maps "reference" and "hull" to `basins`; `indices` is `reconstructed_indices`.
    """
    return dict(
        example=EXAMPLE, fit=dict(fit), diagnostics=metrics,
        graphs=dict(reference=reference.record(), data_map=data_map.record(), hull=hull.record(),
                    merged=merged.record()),
        comparison=dict(data_map=compare_graphs(reference, data_map), hull=compare_graphs(reference, merged)),
        regions=dict(samples=int(samples), **{name: dict(rows=value["rows"], labels=[int(v) for v in value["labels"]])
                                               for name, value in region_results.items()}),
        basins={name: _basins_record(value) for name, value in basins_.items()},
        indices=indices)


def selected_run(folder):
    """The name of the selected fit in `<folder>/selection.json`."""
    return training.read(Path(folder)/"selection.json")["all"]["selected"]


def analyze(model, settings):
    """The graphs, regions, basins and indices of a fit, as the notebook computes them."""
    reference = reference_graph(settings["analysis"]["primary_depth"])
    data_map, _ = learned_graph(model, settings)
    hull, merged = learned_hull_graph(model, settings)
    region_results, samples = attracting_regions_of(model, dict(data_map=data_map, hull=merged), settings)
    basins_ = dict(reference=basins(reference), hull=basins(merged))
    indices = reconstructed_indices(dict(reference=reference, learned=merged))
    return reference, data_map, hull, merged, region_results, samples, basins_, indices


def reference_record(folder):
    """The record of the selected fit of the data folder `folder`, for `scripts/record_reference.py`."""
    folder = Path(folder)
    settings = training.read(folder/"config.json")
    run = selected_run(folder)
    model, data = load(folder/"models"/run), load_data(folder)
    fit = dict(run=run, weights_sha256=weights_digest(folder/"models"/run/"weights.npz"))
    return record_of(fit, diagnostics(model, data), *analyze(model, settings))


# Steps of the notebook ex2_o3_flow.ipynb.

def _fmt(value, digits=4):
    return "none" if value is None else f"{value:.{digits}g}"


def _relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def _tuple(annotations):
    return "(" + ", ".join(annotations) + ")" if annotations else "none"


def _box(hull):
    return f"[{hull[0]:.4g}, {hull[2]:.4g}] x [{hull[1]:.4g}, {hull[3]:.4g}]"


@dataclass
class Run:
    """The mode, the data folder, the selected fit, the settings and the output folder of a notebook run."""
    mode: str
    data: Path
    selected: str
    settings: dict
    out: Path

    @property
    def figures(self):
        return self.out/"figures"

    @property
    def fit(self):
        """The fit used: its name and the digest of its weights, stored (replay) or retrained (train)."""
        folder = self.out if self.mode == "train" else self.data/"models"
        return dict(run=self.selected, weights_sha256=weights_digest(folder/self.selected/"weights.npz"))

    def table(self):
        analysis = self.settings["analysis"]
        depth, hulls = analysis["primary_depth"], analysis["sampled_hulls"]
        size = analysis["data_map"]["primary_source_size"]
        return Table(["Setting", "Value"], [
            ["Mode", "replay of the selected fit" if self.mode == "replay"
             else "retraining of the selected seed with config.json"],
            ["Selected fit", f"{self.selected}, lowest validation objective ({_relative(self.data/'selection.json')})"],
            ["Grid of the learned maps", f"2^{depth} x 2^{depth} cells of Z = [-2, 2] x [0, 2]"],
            ["Data map", f"pairs (E(x), f(E(x))) for {size}^2 + 4 section points; cells without samples map outside Z"],
            ["Sampled image hulls", f"{hulls['samples_per_axis']} x {hulls['samples_per_axis']} samples per cell, "
                                    f"padding {hulls['padding']:g}, cut at u_2 = 0"],
            ["Reference grid", f"2^{depth} x 2^{depth} cells of X/G = [-1.5, 1.5] x [0, 1.5]"],
            ["Output folder", _relative(self.out)]])

    def _repr_html_(self):
        return self.table()._repr_html_()

    def __repr__(self):
        return repr(self.table())


def start(mode="replay", out=None):
    """Check the mode, read the selection and settings, and make a new output folder.

    `out` None makes `outputs/notebooks/ex2_o3_flow-<mode>-<time>`; a relative `out` is taken
    from the repository root. The folder must not exist yet.
    """
    if mode not in ("replay", "train"):
        raise ValueError('MODE is "replay" or "train".')
    data = ROOT/"data"/EXAMPLE
    if out is None:
        out = ROOT/"outputs"/"notebooks"/f"{EXAMPLE}-{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = Path(out) if Path(out).is_absolute() else ROOT/out
    if out.exists():
        raise FileExistsError(f"{_relative(out)} exists; choose a new OUT.")
    (out/"figures").mkdir(parents=True)
    return Run(mode, data, selected_run(data), training.read(data/"config.json"), out)


def system_table():
    """Sampled checks of the map, its quotient map, and the explicit enclosures of the reference."""
    x = sample(256, 11)
    rng = np.random.default_rng(3)
    equivariance = 0.
    for reflection in (False, True):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix[:, 0] *= np.linalg.det(matrix)*(-1 if reflection else 1)
        moved = x.copy()
        moved[:, 1:] = moved[:, 1:]@matrix.T
        image = physical_step(x)
        image[:, 1:] = image[:, 1:]@matrix.T
        equivariance = max(equivariance, float(np.max(np.abs(physical_step(moved) - image))))
    quotient = float(np.max(np.abs(orbit(physical_step(x)) - quotient_step(orbit(x)))))
    physics = verify_physics()
    boxes = grids.cells(64, PHYSICAL_LOWER, PHYSICAL_UPPER)
    images = reference_images(boxes)
    t = np.array(list(np.ndindex(5, 5)))/4.
    points = boxes[:, None, :2] + (boxes[:, None, 2:] - boxes[:, None, :2])*t[None]
    values = quotient_step(points)
    inside = bool(np.all(values >= images[:, None, :2] - 1e-15) and np.all(values <= images[:, None, 2:] + 1e-15))
    return Table(["Check", "Value"], [
        ["Equivariance of F: max |F(a, Qv) - (F_1, Q F_2)(a, v)|, 256 states, a rotation and a reflection",
         _fmt(equivariance, 3)],
        ["Quotient map: max |(a, |v|) o F - olF o (a, |v|)| on the same states", _fmt(quotient, 3)],
        ["Closed form against DOP853 on the vector field, 64 states: one step, ten steps",
         f"{physics['maximum_state_error_one_step']:.3g}, {physics['maximum_state_error_ten_steps']:.3g}"],
        ["olF fixes O, C, A- and A+", "yes" if np.allclose(quotient_step(CONTROLS), CONTROLS, atol=1e-15) else "no"],
        ["Enclosures on 2^6 x 2^6 cells of X/G contain olF of 5 x 5 points of each cell (to 1e-15)",
         "yes" if inside else "no"],
        ["Widest enclosure on that grid", _fmt(float(np.max(images[:, 2:] - images[:, :2])), 4)]],
        "The image of a cell under olF is enclosed by binary64 neighbors of exact square roots, found by "
        "comparing rational squares; olF_1 increases with a, and olF_2 increases with r and decreases with |a|.")


def graph_table(graph, coordinates, caption=None):
    """The Morse sets of a graph: role, cells, box, Conley index annotation and edges."""
    roles = node_roles(graph)
    successors = {}
    for a, b in graph.edges:
        successors.setdefault(a, []).append(f"{b} ({roles[b]})")
    order = {role: i for i, role in enumerate(ORDER)}
    rows = []
    for row in sorted(graph.nodes, key=lambda row: (order.get(row["role"], 9), row["node"])):
        rows.append([row["node"], row["role"] or "none", ", ".join(map(str, row.get("original_nodes", []))),
                     ", ".join(map(str, row["cell_ids"])), _box(row["hull"]), _tuple(row["annotations"]),
                     ", ".join(sorted(successors.get(row["node"], []))) or "none"])
    header = ["Morse set", "Role", "Merged Morse sets", "Cells", f"Box in {coordinates}", "Conley index", "Edges to"]
    if not any(r[2] for r in rows):
        header.pop(2)
        rows = [r[:2] + r[3:] for r in rows]
    return Table(header, rows, caption)


def data_table(data, folder):
    """The splits of the data, checked against the checksums recorded with them."""
    metadata = training.read(Path(folder)/"metadata.json")
    rows = []
    for split in SPLITS:
        values = data[f"{split}_x"]
        if _array_digest(values) != metadata["array_sha256"][f"{split}_x"]:
            raise ValueError(f"{split}_x differs from its checksum in metadata.json.")
        ar = orbit(values[:, 0])
        rows.append([split, len(values), f"{np.mean(np.abs(ar[:, 0]) < .1):.3f}", f"{np.mean(ar[:, 1] < .1):.3f}",
                     f"0, tau, ..., {values.shape[1] - 1} tau"])
    return Table(["Split", "Starts", "Share with |a| < 0.1", "Share with |v| < 0.1", "Times"], rows,
                 f"tau = log 3. Sampling: {metadata['sampling']}. Every array matches its SHA-256 checksum in "
                 f"{_relative(Path(folder)/'metadata.json')}.")


def seeds_table(run):
    """All fits of the sweep, ranked by validation objective, with their stored held-out errors."""
    selection = training.read(run.data/"selection.json")["all"]
    rows = []
    for rank, entry in enumerate(selection["ranking"], 1):
        folder = run.data/"models"/entry["run"]
        config, metrics = training.read(folder/"config.json"), training.read(folder/"metrics.json")
        rows.append([rank, entry["run"] + (" (selected)" if entry["run"] == selection["selected"] else ""),
                     _fmt(entry["validation"]), config["best_step"], _fmt(metrics["forecasts"]["1"]["orbit_rms"]),
                     _fmt(metrics["forecasts"]["10"]["orbit_rms"]), _fmt(metrics["residual_maximum"]),
                     _fmt(metrics["minimum_sampled_encoder_singular_value"], 3),
                     len(metrics["observed_axis_fixed_points"])])
    return Table(["Rank", "Fit", "Validation objective", "Selected step", "One-step orbit RMS", "Ten-step orbit RMS",
                  "Largest one-step residual", "Least singular value of DE", "Fixed points of f on the axes"], rows,
                 f"Selection by {selection['criterion']}. Held-out errors on the test starts, from the stored "
                 "metrics of each fit.")


def selected_fit(run, data):
    """The selected fit: loaded from `data/ex2_o3_flow/models` in replay, retrained in train mode.

    Retraining uses the seed of the selected fit and `config.json`, on the CPU, and writes
    into `<out>/<fit>`. Returns the model and a line that reports the fit.
    """
    stored = run.data/"models"/run.selected
    config = training.read(stored/"config.json")
    if run.mode == "replay":
        model = load(stored)
        return model, Line(f"Loaded {run.selected} from {_relative(stored)}: selected at step {config['best_step']:,} "
                           f"with validation objective {config['best_validation']:.6g}; weights digest "
                           f"{weights_digest(stored/'weights.npz')[:16]}.")
    folder = run.out/run.selected
    model = train(folder, data, config["seed"], run.settings)
    new = training.read(folder/"config.json")
    same = weights_digest(folder/"weights.npz") == weights_digest(stored/"weights.npz")
    with np.load(folder/"weights.npz") as a, np.load(stored/"weights.npz") as b:
        gap = max(float(np.max(np.abs(a[k] - b[k]), initial=0.)) for k in a.files)
    return model, Line(f"Trained seed {config['seed']} for {run.settings['training']['steps']:,} updates in "
                       f"{new['seconds']:.0f} s; selected at step {new['best_step']:,} with validation objective "
                       f"{new['best_validation']:.6g}. The weights " +
                       ("equal those of the stored fit bitwise." if same else
                        f"differ from those of the stored fit by at most {gap:.3g}."), passed=same)


def diagnostics_table(metrics):
    """The held-out orbit errors and the sampled checks of the encoder and the latent map."""
    forecasts = metrics["forecasts"]
    rows = [[f"Orbit error |d(f^{h}(E(x))) - (a, |v|)(F^{h}(x))|", _fmt(forecasts[h]["orbit_rms"]),
             _fmt(forecasts[h]["orbit_maximum"])] for h in ("0", "1", "10")]
    rows += [["Invariance of E: max |E(a, Qv) - E(a, v)|, a rotation and a reflection",
              _fmt(metrics["sampled_encoder_invariance_error"], 3), ""],
             ["One-step residual |f(E(x)) - E(F(x))|", _fmt(metrics["residual_rms"]), _fmt(metrics["residual_maximum"])],
             ["Least singular value of DE on the section; range of det DE",
              _fmt(metrics["minimum_sampled_encoder_singular_value"], 3),
              f"{metrics['minimum_sampled_encoder_jacobian_determinant']:.4f} to "
              f"{metrics['maximum_sampled_encoder_jacobian_determinant']:.4f}"]]
    for name, code, residual in zip(metrics["physical_anchor_names"], metrics["encoded_anchors"],
                                    metrics["anchor_time_residuals"], strict=True):
        rows.append([f"E({name}) and |f(E({name})) - E({name})|", f"({code[0]:.4f}, {code[1]:.4f})", _fmt(residual, 3)])
    for row in metrics["observed_axis_fixed_points"]:
        rows.append([f"Fixed point of f on the axes ({row['kind']}) and its decoded orbit",
                     f"({row['latent'][0]:.4f}, {row['latent'][1]:.4f})",
                     f"({row['decoded_orbit'][0]:.4f}, {row['decoded_orbit'][1]:.4f})"])
    return Table(["Quantity", "RMS or value", "Maximum or second value"], rows,
                 "Test split, 4,096 starts, at times h log 3. Section 5.2 quotes the one-step orbit RMS.")


def comparison_table(comparisons):
    """The learned Morse graphs against the reference, role by role, and the order."""
    data_map, hull = comparisons["data_map"], comparisons["hull"]
    order = lambda edges: ", ".join(f"{a} -> {b}" for a, b in edges)  # noqa: E731
    rows = []
    for a, b in zip(data_map["nodes"], hull["nodes"], strict=True):
        rows.append([a["role"], _tuple(a["reference_annotations"]), _tuple(a["learned_annotations"]),
                     "yes" if a["agree"] else "no", _tuple(b["learned_annotations"]), "yes" if b["agree"] else "no"])
    rows.append(["Order", order(data_map["reference_edges"]), order(data_map["learned_edges"]),
                 "yes" if data_map["edges_agree"] else "no", order(hull["learned_edges"]),
                 "yes" if hull["edges_agree"] else "no"])
    return Table(["Role", "Reference", "Data map", "Agree", "Sampled hulls", "Agree"], rows,
                 "Conley indices over F_5 in degrees 0, 1, 2, and the order between roles. 'none': CMGDB returns "
                 "no index for that Morse set.")


def regions_table(region_results, samples, graphs):
    """Tolerances of the attracting regions and the sampled residuals modulo G, for both learned graphs."""
    names = {"data_map": "data map", "hull": "sampled hulls"}
    body = []
    for name, result in region_results.items():
        roles = node_roles(graphs[name])
        for row in result["rows"]:
            body.append([names[name], roles[row["node"]], row["cells"], _box(row["hull"]), f"{row['samples']:,}",
                         _fmt(row["max_residual"]), _fmt(row["tolerance"]), "yes" if row["passed"] else "no"])
    return Table(["Map", "Minimal Morse set", "Cells of A", "Box of |A| in u", "Samples in E^-1(N)",
                  "Largest residual", "Tolerance", "Residual below tolerance"], body,
                 f"Euclidean metric of Z, map time log 3, {samples:,} section points. The data-map rows are the "
                 "values of Section 5.2; the sampled-hull rows belong to the regions of fig:o3(c). Sampled residuals "
                 "and hulls, binary64, not validated.")


def indices_table(indices):
    """The indices in X reconstructed for the reference and the sampled-hull graph, against the proved ones."""
    rows = []
    for role in ORDER:
        ref, new = indices["reference"][role], indices["learned"][role]
        rows.append([role, _tuple(PROVED[role]), homology_degrees(PROVED[role]), _tuple(ref["quotient"]),
                     _tuple(ref["in_X"]), _tuple(new["quotient"]), _tuple(new["in_X"]),
                     "yes" if ref["agree"] and new["agree"] else "no"])
    return Table(["Role", "Proved in X", "Nonzero groups", "Reference in X/G", "Reference in X", "Learned in Z",
                  "Learned in X", "Agree"], rows,
                 "Shift classes over F_5, in degrees 0, 1, 2 in the rectangle and 0 to 4 in X. The quotient indices "
                 "computed here equal CMGDB's annotations. The reference reconstruction uses the enclosures of olF; "
                 "the learned one uses sampled hulls, so its indices in X are observed.")


def check_record(run, record):
    """Compare the record with `data/ex2_o3_flow/reference/results.json`.

    The record is written to `<out>/record.json`. In replay any difference raises
    AssertionError; in train mode the differences are reported.
    """
    path = run.data/"reference"/"results.json"
    training.write(run.out/"record.json", record)
    count, differences = compare_records(record, training.read(path), exact=EXACT)
    text = (f"The record matches {_relative(path)}: {count:,} values, graphs, basins and indices exactly, "
            f"other numbers to relative 1e-9." if not differences else
            f"The record differs from {_relative(path)} in {len(differences)} of {count:,} values, first at "
            f"{differences[0][0]}.")
    if differences and run.mode == "replay":
        raise AssertionError(text + " " + "; ".join(f"{p}: {a!r} != {b!r}" for p, a, b in differences[:10]))
    return Line(text, passed=not differences, compared=count, differences=differences)


def summary(run, record, check):
    """Every quantity of Section 5.2 and its appendix, computed and as the manuscript states it.

    The manuscript values describe the fits of its current text; the computed values come from
    the selected fit of this repository. Writes `<out>/verification.json`, which also lists the
    saved panels, and returns the table.
    """
    graphs, comparison, indices = record["graphs"], record["comparison"], record["indices"]
    roles = {name: {row["node"]: row["role"] for row in graphs[name]["nodes"]} for name in graphs}
    region = {name: {roles[name if name == "data_map" else "merged"][row["node"]]: row
                     for row in record["regions"][name]["rows"]} for name in ("data_map", "hull")}
    order = lambda edges: ", ".join(f"{a} -> {b}" for a, b in edges)  # noqa: E731
    hull_nodes, merged_nodes = len(graphs["hull"]["nodes"]), len(graphs["merged"]["nodes"])
    c_index = next(row["learned_annotations"] for row in comparison["hull"]["nodes"] if row["role"] == "C")
    data_c = next(row["learned_annotations"] for row in comparison["data_map"]["nodes"] if row["role"] == "C")
    others_agree = all(row["agree"] for row in comparison["data_map"]["nodes"] if row["role"] != "C")
    rows = [
        ["Section 5.2", "One-step orbit RMS on held-out data", "0.0053",
         _fmt(record["diagnostics"]["forecasts"]["1"]["orbit_rms"], 2)],
        ["Section 5.2", "Data-map and reference Morse graphs on 2^6 x 2^6 cells agree, order O -> C -> A-+",
         "agree, O -> C -> A-+",
         ("same order" if comparison["data_map"]["edges_agree"] else "different order") +
         f" ({order(comparison['data_map']['learned_edges'])}); indices agree at O and A-+: "
         f"{'yes' if others_agree else 'no'}; index at C: {_tuple(data_c)}"],
        ["Section 5.2", "Quotient graph of fig:o3 from sampled image hulls: native Morse sets and nodes",
         "6 grouped into 4 (a separate fit)",
         f"{hull_nodes} Morse sets, {merged_nodes} nodes, of the selected fit" +
         (", no grouping" if hull_nodes == merged_nodes else "")],
        ["Section 5.2", "Sampled-hull indices agree with the reference, (0, x-1, 0) at C", "yes",
         ("yes" if comparison["hull"]["agree"] else "no") + f", {_tuple(c_index)} at C"]]
    for role, residual, tolerance in (("A-", "0.020", "0.123"), ("A+", "0.018", "0.122")):
        row = region["data_map"][role]
        relation = "<" if row["passed"] else ">="
        rows.append(["Section 5.2", f"{role}: largest sampled residual of the data-map fit against the tolerance",
                     f"{residual} < {tolerance}", f"{_fmt(row['max_residual'], 3)} {relation} {_fmt(row['tolerance'], 3)}"])
    agree = all(indices[name][role]["agree"] for name in indices for role in indices[name])
    rows += [
        ["Section 5.2", "Indices in X: H_4 = F at O, H_1 = H_3 = F at C, H_0 = F at A-+", "proved",
         "reconstruction agrees for the reference and the learned graph" if agree else "reconstruction differs"],
        ["supp:o3-computation", "Cells of each attracting region of the data map", "1,160",
         " and ".join(f"{region['data_map'][r]['cells']:,}" for r in ("A-", "A+"))],
        ["supp:o3-computation", "Encoded physical sources of the data map", "1025^2 + 4",
         f"{run.settings['analysis']['data_map']['primary_source_size']}^2 + 4"]]
    for role, value in (("A-", "(0.123087, 0.0195666)"), ("A+", "(0.122487, 0.0180491)")):
        row = region["data_map"][role]
        rows.append(["supp:o3-computation", f"{role}: (tolerance, residual)", value,
                     f"({row['tolerance']:.6g}, {row['max_residual']:.6g})"])
    training_settings, sizes = run.settings["training"], run.settings["data"]["sizes"]
    rows += [
        ["supp:o3-computation", "Physical-section samples for the residual", "1,152,136", f"{record['regions']['samples']:,}"],
        ["Saved notebook configurations", "Starts (train, validation, test), updates, batch",
         "16384, 4096, 4096; 20000; 256",
         f"{', '.join(map(str, sizes))}; {training_settings['steps']}; {training_settings['batch']}"],
        ["Saved notebook configurations", "Learning rate", "1e-3 to 2e-5",
         f"{training_settings['learning_rate_start']:g} to {training_settings['learning_rate_end']:g}"],
        ["fig:o3 caption", "Fits of the panels", "(b) the data-map fit, (c)-(e) a separate fit",
         f"all panels from {record['fit']['run']}; (c)-(e) use its sampled-hull graph"],
        ["Replay", "Record against the stored reference", "", "matches" if check.passed else "differs"]]
    verification = dict(example=EXAMPLE, mode=run.mode, fit=record["fit"], record_matches_reference=check.passed,
                        compared_values=check.values.get("compared"),
                        differences=[list(map(str, d)) for d in check.values.get("differences", [])],
                        summary=[dict(zip(("where", "statement", "manuscript", "computed"), row)) for row in rows],
                        figures=sorted(str(p.relative_to(run.figures)) for p in run.figures.rglob("*.pdf")),
                        versions={p: version(p) for p in VERSIONS})
    training.write(run.out/"verification.json", verification)
    return Table(["Where", "Statement", "Manuscript", "Computed"], rows,
                 f"Written to {_relative(run.out/'verification.json')}. The manuscript values describe the fits "
                 "of its current text.")
