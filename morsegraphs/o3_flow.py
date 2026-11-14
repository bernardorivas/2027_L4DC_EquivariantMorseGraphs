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

Import this module after `device.configure`, if that is called at all: it imports
JAX and enables float64.
"""
import copy
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.metadata import version
from math import inf, nextafter, sqrt
from pathlib import Path
import time
import weakref

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

from . import grids, regions, training

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
    folder.mkdir(parents=True, exist_ok=False)
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
