"""Section 5.1: an SO(2)-equivariant flow on the sphere.

The vector field V(p) = 0.6 e3 x p + z (0.4^2 - z^2)(e3 - z p) on S^2, with z = p . e3,
is equivariant under the rotations about e3, and F is its time-ten map. The height z
identifies S^2/SO(2) with [-1, 1] and the quotient map with the time-ten map h_10 of
the flow of z' = z (0.4^2 - z^2)(1 - z^2): the poles and the equator repel and the
latitudes z = -0.4 and z = 0.4 attract.

The learned model is an invariant encoder E: S^2 -> [-1, 1], an orbit decoder d and a
time-one map f of [-1, 1], fitted jointly from pairs (p, F_1(p)). CMGDB computes the
Morse graph of f^10 on 2^7 cells of [-1, 1] from sampled image hulls, and that of h_10
from explicit enclosures. Both ends of [-1, 1] are faces of the orbit space (the
poles). Everything here runs in float64 on the CPU unless `device.configure` chose
otherwise before JAX started.

The last sections hold the panels of the figures of Section 5.1 and its appendix, the
record of every result that the notebook `ex1_so2_flow.ipynb` checks, and the steps of
that notebook, each of which returns one table, one line or the saved panels of a figure.
"""
import copy
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
import hashlib
from html import escape
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import time
import warnings
import zipfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # no effect once JAX has started
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# tqdm, which EMLP imports, warns in a notebook that progress bars need ipywidgets; none is shown.
warnings.filterwarnings("ignore", message="IProgress not found")
# fontTools logs a warning about font timestamps when Matplotlib embeds fonts in a PDF.
logging.getLogger("fontTools").setLevel(logging.ERROR)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import objax
from emlp.groups import SO2eR3
from emlp.nn.objax import EMLP
from emlp.reps import T, V
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import grids, homology, plotting, regions, training

LOWER, UPPER = [-1.], [1.]
FACES = np.array([[True, True]])  # z = -1 and z = 1 are the poles, faces of the orbit space
SPLITS = ("train", "validation", "test")
VERSIONS = ("emlp", "objax", "jax", "numpy", "scipy", "cmgdb")

DEFAULTS = {
    "experiment": "joint-interval-v1",
    "seeds": [0, 1, 2],
    "data": {
        "generator_seed": 20261112,
        "sizes": {"train": 16384, "validation": 4096, "test": 4096},
        "times": [0, 1, 10],
        "polar_fraction": 0.25,
        "polar_distance_log10_range": [-10, -2],
        "integration": {"variable": "w=arctanh(z)", "method": "DOP853", "rtol": 2e-13, "atol": 2e-14},
    },
    "training": {
        "width": 32,
        "layers": 2,
        "hidden": "scalar",
        "steps": 20000,
        "batch": 512,
        "learning_rate_start": 0.002,
        "learning_rate_end": 0.0001,
        "validate_every": 100,
    },
    "recipe": {
        "encoder": "calibrated",
        "encoder_shift_bound": 0.35,
        "decoder_shift_bound": 0.35,
        "time_shift_bound": 1.4,
        "orbit_log_weight": 0.03,
        "consistency_log_weight": 0.08,
        "ordering_weight": 0.1,
        "ordering_minimum_slope": 0.2,
        "ordering_points": 65,
        "ordering_domain": [-0.999, 0.999],
    },
    "analysis": {
        "cells": 128,
        "image_samples": 65,
        "time_map_iterates": 10,
        "image_padding": 1e-10,
        "region_samples": 65537,
        "pole_samples_per_end": 2000,
        "pole_angle_range": [1e-9, 0.01],
    },
}


# The system and its group action.

def vector_field(p):
    """V(p) = 0.6 e3 x p + z (0.4^2 - z^2)(e3 - z p) at points p of S^2, rows (x, y, z)."""
    p = np.asarray(p, float)
    z = p[..., 2:3]
    turn = np.stack((-p[..., 1], p[..., 0], np.zeros_like(p[..., 0])), axis=-1)
    return .6 * turn + z * (.16 - z**2) * (np.array([0., 0., 1.]) - z * p)


def rotate(p, angle):
    """The rotation by `angle` about e3."""
    p = np.array(p, float)
    c, s = np.cos(angle), np.sin(angle)
    p[..., :2] = p[..., :2] @ np.array([[c, -s], [s, c]]).T
    return p


def _log_latitude_rate(_, w):
    """z' = z (0.4^2 - z^2)(1 - z^2) in the log latitude w = arctanh(z)."""
    return np.tanh(w) * (.16 - np.tanh(w)**2)


def _flow(w, duration, t_eval=None, integration=DEFAULTS["data"]["integration"]):
    answer = solve_ivp(_log_latitude_rate, (0., duration), w, t_eval=t_eval, method=integration["method"],
                       rtol=integration["rtol"], atol=integration["atol"])
    if not answer.success:
        raise RuntimeError(answer.message)
    return answer.y


# Explicit enclosures of the quotient map h_10.

@lru_cache(maxsize=1)
def _exp_interval():
    """Rational lower and upper bounds for exp(336/5)."""
    x = Fraction(336, 5)
    term = total = Fraction(1)
    for k in range(1, 201):
        term *= x / k
        total += term
    remainder = term * (x / 201) / (1 - x / 202)
    return total, total + remainder


def _first_integral(z):
    """Q(z) = w^21 (1 - w)^4 / |w - 4/25|^25 with w = z^2, so that Q(h_t(z)) = exp(168 t/25) Q(z)."""
    w = z * z
    return w**21 * (1 - w)**4 / abs(w - Fraction(4, 25))**25


@lru_cache(maxsize=None)
def time_ten(height):
    """An enclosure of h_10(height) in binary64, treating the binary64 height as an exact rational."""
    height = float(height)
    if not -1 <= height <= 1:
        raise ValueError("Sphere heights must be in [-1, 1].")
    if height < 0:
        low, high = time_ten(-height)
        return -high, -low
    z = Fraction(height)
    tropic = Fraction(2, 5)
    if z in (0, tropic, 1):
        return height, height
    q = _first_integral(z)
    exp_low, exp_high = _exp_interval()
    lower, upper = q * exp_low, q * exp_high
    lo, hi = sorted((z, tropic))
    for _ in range(44):
        mid = (lo + hi) / 2
        q_mid = _first_integral(mid)
        if lower <= q_mid <= upper:
            break
        if (q_mid < lower) == (z < tropic):
            lo = mid
        else:
            hi = mid
    return max(-1., float(np.nextafter(float(lo), -np.inf))), min(1., float(np.nextafter(float(hi), np.inf)))


def time_ten_images(boxes):
    """Enclosures of h_10 on intervals of heights, rows (lower, upper), from its monotonicity."""
    boxes = np.asarray(boxes, float)
    if boxes.ndim != 2 or boxes.shape[1] != 2 or not np.isfinite(boxes).all() or np.any(boxes[:, 0] > boxes[:, 1]):
        raise ValueError("Give intervals of heights as rows (lower, upper).")
    if np.any(boxes < -1) or np.any(boxes > 1):
        raise ValueError("Sphere heights must be in [-1, 1].")
    return np.asarray([(time_ten(lo)[0], time_ten(hi)[1]) for lo, hi in boxes], float).reshape(-1, 2)


def reference_graph(settings=DEFAULTS):
    """The Morse graph of h_10 on the grid of the paper, from the explicit enclosures."""
    boxes = grids.cells(settings["analysis"]["cells"], LOWER, UPPER)
    return grids.morse_graph(boxes, time_ten_images(boxes))


# Data.

def trajectories(settings=DEFAULTS["data"]):
    """Starts on S^2 and their states at the recorded times, split by start.

    In each split, three quarters of the heights are uniform in [-1, 1] and a quarter lie
    near the poles, with log10(1 - |z|) uniform; the longitudes are uniform. The heights
    are integrated in the log latitude w = arctanh(z), and the longitude turns at rate 0.6.
    """
    rng = np.random.default_rng(settings["generator_seed"])
    times = [float(t) for t in settings["times"]]
    if times[0] != 0:
        raise ValueError("The first time is the start.")
    low, high = settings["polar_distance_log10_range"]
    arrays = {}
    for split in SPLITS:
        count = settings["sizes"][split]
        z = rng.uniform(-1, 1, count)
        polar = int(count * settings["polar_fraction"])
        z[:polar] = rng.choice([-1., 1.], polar) * (1 - 10.**rng.uniform(low, high, polar))
        rng.shuffle(z)
        angle = rng.uniform(0, 2 * np.pi, count)
        w = np.arctanh(z)
        later = _flow(w, times[-1], t_eval=times[1:], integration=settings["integration"])
        for column, duration in enumerate(times):
            latitude = w if column == 0 else later[:, column - 1]
            radius = 1 / np.cosh(latitude)
            phi = angle + .6 * duration
            tag = f"{duration:g}"
            arrays[f"{split}_p{tag}"] = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.tanh(latitude)))
            arrays[f"{split}_w{tag}"] = latitude
    return arrays


def _array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _save_npz(path, arrays):
    """A compressed .npz whose bytes depend only on the arrays (fixed member timestamps)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)


def generate_data(folder, settings=DEFAULTS):
    """Write `trajectories.npz` and `metadata.json` into `folder`; return the metadata.

    For each split the archive holds `<split>_p<t>`, the states (x, y, z) at time t, of
    shape (count, 3), and `<split>_w<t>`, their log latitudes arctanh(z). The training
    objective uses times 0 and 1; time 10 serves the held-out rollouts.
    """
    folder = Path(folder)
    path = folder / "trajectories.npz"
    if path.exists():
        raise FileExistsError(path)
    folder.mkdir(parents=True, exist_ok=True)
    data = settings["data"]
    arrays = trajectories(data)
    _save_npz(path, arrays)
    metadata = dict(experiment=settings["experiment"], generator=f"numpy.random.default_rng({data['generator_seed']})",
                    generator_seed=data["generator_seed"], sizes=dict(data["sizes"]), times=list(data["times"]),
                    state_order=["x", "y", "z"],
                    fields={"p": "states on S^2 at time t, shape (count, 3)",
                            "w": "log latitude arctanh(z) at time t, shape (count,)"},
                    initial_distribution=dict(uniform_height_fraction=1 - data["polar_fraction"],
                                              polar_fraction=data["polar_fraction"],
                                              polar_distance_log10_range=data["polar_distance_log10_range"],
                                              longitude="uniform in [0, 2 pi)"),
                    flow=dict(height_rate="z*(0.16-z*z)*(1-z*z)", longitude_rate=0.6,
                              log_latitude_rate="tanh(w)*(0.16-tanh(w)**2)"),
                    integration=dict(data["integration"]),
                    training_pairs=["train_p0", "train_p1"], auxiliary_fields=["train_w0", "train_w1"],
                    arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=_array_digest(value))
                            for key, value in arrays.items()},
                    file="trajectories.npz", sha256=training.digest(path))
    training.write(folder / "metadata.json", metadata)
    return metadata


def load_data(folder):
    """The arrays of `trajectories.npz`, checked against the checksums of `metadata.json`."""
    folder = Path(folder)
    with np.load(folder / "trajectories.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    if (folder / "metadata.json").exists():
        expected = training.read(folder / "metadata.json").get("arrays", {})
        for key, row in expected.items():
            if key not in data or _array_digest(data[key]) != row["sha256"]:
                raise ValueError(f"The array {key} differs from its recorded checksum.")
    return data


def meridian_samples(settings=DEFAULTS["analysis"]):
    """Points of one meridian of S^2 and their images under the flow at the map time.

    The map time is `time_map_iterates` steps of the time-one map, ten in the paper. The
    polar angles are equally spaced, with geometric refinement near both poles. Every
    orbit of SO(2) meets the meridian, so these points sample E^{-1}(N) for every
    invariant N. Returns the angles, the starts and the images.
    """
    count, poles = settings["region_samples"], settings["pole_samples_per_end"]
    duration = float(settings["time_map_iterates"])
    near = np.geomspace(*settings["pole_angle_range"], poles)
    theta = np.unique(np.r_[np.linspace(0., np.pi, count), near, np.pi - near])
    starts = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
    inside = (theta > 0) & (theta < np.pi)
    w = -np.log(np.tan(theta[inside] / 2))
    later = _flow(w, duration)
    radius, z = np.zeros(len(theta)), np.cos(theta)
    radius[inside], z[inside] = 1 / np.cosh(later[:, -1]), np.tanh(later[:, -1])
    phi = .6 * duration
    images = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    starts[0], starts[-1] = [0., 0., 1.], [0., 0., -1.]
    return theta, starts, images


# The networks.

def meridian(z):
    """The point of height z on the meridian through e1."""
    z = jnp.asarray(z)
    return jnp.stack((jnp.sqrt(jnp.maximum(0., (1 - z) * (1 + z))), jnp.zeros_like(z), z), axis=-1)


def move(u, shift):
    """A bounded change of log latitude, with both endpoints of [-1, 1] fixed."""
    gain = jnp.tanh(shift)
    return (u + gain) / (1 + u * gain)


class SphereInterval(objax.Module):
    """An invariant encoder, an orbit decoder and an interval time-one map.

    For M(u, s) = (u + tanh s)/(1 + u tanh s) and EMLP networks e, a, b with scalar
    hidden channels, E(p) = M(z, c_E tanh e(p)), d(u) = M(u, c_d tanh a(u)) and
    f(u) = M(u, c_f tanh b(u)). The heads fix the interval endpoints. The decoder
    returns a height, which stands for the circle of latitude of that height.
    """

    def __init__(self, width=32, layers=2, hidden="scalar", recipe=None):
        if hidden != "scalar":
            raise ValueError("The hidden channels are scalars.")
        self.recipe = dict(DEFAULTS["recipe"] if recipe is None else recipe)
        group = SO2eR3()
        channels = width * T(0)
        self.encoder = EMLP(V, T(0), group=group, ch=channels, num_layers=layers)
        self.decoder = EMLP(T(0), T(0), group=group, ch=channels, num_layers=layers)
        self.dynamics = EMLP(T(0), T(0), group=group, ch=channels, num_layers=layers)

    def encoder_shift(self, p):
        return self.recipe["encoder_shift_bound"] * jnp.tanh(self.encoder(p)[..., 0])

    def calibrated(self):
        return self.recipe.get("encoder", "height_shift") == "calibrated"

    def _eta(self, p):
        """The invariant network with values in (-1, 1), and its values at the poles."""
        eta = jnp.tanh(self.encoder(p)[..., 0])
        top, bottom = jnp.tanh(self.encoder(jnp.array([[0., 0., 1.], [0., 0., -1.]]))[..., 0])
        return eta, top, bottom

    def encode(self, p):
        if self.calibrated():
            # E = (eta - c)/b sends the poles, the points fixed by the group, to -1 and 1.
            eta, top, bottom = self._eta(p)
            return 2 * (eta - bottom) / (top - bottom) - 1
        return move(p[..., 2], self.encoder_shift(p))

    def encoded_log_latitude(self, p, w):
        """arctanh(E(p)), for a state p of log latitude w = arctanh(z)."""
        if self.calibrated():
            # Before training the network need not increase with the height, so E can leave
            # (-1, 1); the floor keeps the objective finite until the ordering penalty acts.
            eta, top, bottom = self._eta(p)
            floor, sign = 1e-12, jnp.sign(top - bottom)  # (1 + E)/(1 - E) = (eta - bottom)/(top - eta)
            return .5 * (jnp.log(jnp.maximum(sign * (eta - bottom), floor))
                         - jnp.log(jnp.maximum(sign * (top - eta), floor)))
        return w + self.encoder_shift(p)

    def decoder_shift(self, u):
        raw = self.decoder(u[..., None])[..., 0]
        bound = self.recipe["decoder_shift_bound"]
        return raw if bound is None else bound * jnp.tanh(raw)  # None: no bound

    def decode(self, u):
        return move(u, self.decoder_shift(u))

    def time_shift(self, u):
        raw = self.dynamics(u[..., None])[..., 0]
        bound = self.recipe["time_shift_bound"]
        return raw if bound is None else bound * jnp.tanh(raw)  # None: no bound

    def advance(self, u):
        return move(u, self.time_shift(u))


def objective(model, p0, p1, w0, w1):
    """Orbit reconstruction, orbit prediction and consistency, with an ordering penalty.

    The orbit distance is that of the circles of latitude, (z - z*)^2 + (r - r*)^2,
    plus a multiple of the squared error in log latitude; the losses are evaluated in
    log latitude before returning to heights. The recipe's `consistency_metric` measures
    consistency between f(E(x)) and E(F(x)) in the latent coordinate ("latent") or,
    after decoding, in the orbit metric ("orbit").
    """
    e0 = model.encoded_log_latitude(p0, w0)
    u0 = jnp.tanh(e0)
    reconstructed = e0 + model.decoder_shift(u0)
    predicted = e0 + model.time_shift(u0)
    decoded_prediction = predicted + model.decoder_shift(jnp.tanh(predicted))
    e1 = model.encoded_log_latitude(p1, w1)

    def orbit_error(a, b):
        return jnp.mean((jnp.tanh(a) - jnp.tanh(b))**2 +
                        (1 / jnp.cosh(a) - 1 / jnp.cosh(b))**2 + model.recipe["orbit_log_weight"] * (a - b)**2)

    reconstruction = orbit_error(reconstructed, w0)
    prediction = orbit_error(decoded_prediction, w1)
    if model.recipe.get("consistency_metric", "latent") == "orbit":
        # Compare f(E(x)) with E(F(x)) after decoding, in the orbit metric: the objective
        # then does not depend on how E parametrizes the interval.
        consistency = orbit_error(decoded_prediction, e1 + model.decoder_shift(jnp.tanh(e1)))
    else:
        consistency = jnp.mean((jnp.tanh(predicted) - jnp.tanh(e1))**2
                               + model.recipe["consistency_log_weight"] * (predicted - e1)**2)
    # An increasing encoder is a condition for using the learned interval as orbit coordinates.
    z = jnp.linspace(*model.recipe["ordering_domain"], model.recipe["ordering_points"])
    differences = jnp.diff(model.encode(meridian(z))) / jnp.diff(z)
    ordering = jnp.mean(jax.nn.relu(model.recipe["ordering_minimum_slope"] - differences)**2)
    return reconstruction + prediction + consistency + model.recipe["ordering_weight"] * ordering


def _inputs(data, split):
    return [jnp.asarray(data[f"{split}_{key}"]) for key in ("p0", "p1", "w0", "w1")]


def objective_value(model, data, split="validation"):
    """The objective on a whole split, as used for selection."""
    return float(objective(model, *_inputs(data, split)))


def train(folder, data, seed, settings=DEFAULTS, progress=None):
    """Train one seed and keep the weights with the lowest validation objective.

    Writes `weights.npz`, `config.json` and `history.json` into `folder`. Adam takes
    minibatches drawn with replacement by `default_rng(seed + 100)`, with a cosine
    learning rate; NumPy and the Objax generator are seeded with `seed` before the
    networks are built. Validation runs at the start, at step one, every
    `validate_every` steps and at the last step.
    """
    folder = Path(folder)
    if (folder / "weights.npz").exists():
        raise FileExistsError(folder / "weights.npz")
    folder.mkdir(parents=True, exist_ok=True)
    options = settings["training"]
    steps, batch, every = options["steps"], options["batch"], options["validate_every"]
    np.random.seed(seed)
    objax.random.DEFAULT_GENERATOR.seed(seed)
    model = SphereInterval(options["width"], options["layers"], options["hidden"], settings["recipe"])
    initial = {name: [np.asarray(v.value).copy() for v in getattr(model, name).vars().values()]
               for name in ("encoder", "decoder", "dynamics")}
    optimizer = objax.optimizer.Adam(model.vars())
    loss = lambda *values: objective(model, *values)
    gradient = objax.GradValues(loss, model.vars())

    @objax.Function.with_vars(model.vars() + optimizer.vars())
    def update(p0, p1, w0, w1, rate):
        gradients, values = gradient(p0, p1, w0, w1)
        optimizer(rate, gradients)
        return values[0]

    update = objax.Jit(update)
    validate = objax.Jit(objax.Function.with_vars(model.vars())(loss))
    inputs = {split: _inputs(data, split) for split in ("train", "validation")}
    rng = np.random.default_rng(seed + 100)
    start = time.perf_counter()
    best, best_step = float(validate(*inputs["validation"])), 0
    training.save_weights(folder / "weights.npz", model.vars())
    history = [dict(step=0, validation=best)]
    if progress is not None:
        progress.update(status="training", step=0, validation=best, seconds=time.perf_counter() - start)
    for step in range(1, steps + 1):
        ids = rng.integers(0, len(inputs["train"][0]), batch)
        rate = training.cosine_rate(step, steps, options["learning_rate_start"], options["learning_rate_end"])
        training_loss = float(update(*(values[ids] for values in inputs["train"]), rate))
        if not np.isfinite(training_loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        if step == 1 or step % every == 0 or step == steps:
            valid = float(validate(*inputs["validation"]))
            history.append(dict(step=step, training=training_loss, validation=valid))
            if valid < best:
                best, best_step = valid, step
                training.save_weights(folder / "weights.npz", model.vars())
            if progress is not None:
                progress.update(status="training", step=step, validation=valid, seconds=time.perf_counter() - start)
    seconds = time.perf_counter() - start
    training.load_weights(folder / "weights.npz", model.vars())
    changes = {name: float(np.sqrt(sum(np.sum((np.asarray(v.value) - a)**2)
                                       for v, a in zip(getattr(model, name).vars().values(), initial[name], strict=True))))
               for name in initial}
    config = dict(seed=seed, best_step=best_step, best_validation=best, seconds=seconds,
                  training="joint orbit reconstruction, orbit prediction, and latent consistency",
                  weights_format="numeric-ordered-v1", parameter_changes=changes,
                  parameters={name: int(sum(v.value.size for v in getattr(model, name).vars().values()))
                              for name in initial},
                  settings=copy.deepcopy(settings), versions={p: version(p) for p in VERSIONS})
    training.write(folder / "config.json", config)
    training.write(folder / "history.json", history)
    if progress is not None:
        progress.update(status="trained", step=steps, validation=best, seconds=seconds, best_step=best_step)
    return model


def load(folder):
    """The model saved by `train` in `folder`."""
    folder = Path(folder)
    config = training.read(folder / "config.json")
    options = config["settings"]["training"]
    np.random.seed(config["seed"])
    objax.random.DEFAULT_GENERATOR.seed(config["seed"])
    model = SphereInterval(options["width"], options["layers"], options["hidden"], config["settings"]["recipe"])
    training.load_weights(folder / "weights.npz", model.vars())
    return model


def evaluator(model, method):
    """A compiled function of JAX arrays that evaluates `method` with the weights of `model`."""
    return objax.Jit(objax.Function.with_vars(model.vars())(method))


def functions(model, iterates=DEFAULTS["analysis"]["time_map_iterates"]):
    """Compiled maps of a model, from NumPy arrays to NumPy arrays.

    `encode` takes points of S^2, `height_map` heights z (the encoder on a meridian),
    `decode`, `advance` (f) and `time_ten` (f^iterates) points of [-1, 1].
    """
    def iterate(u):
        for _ in range(iterates):
            u = model.advance(u)
        return u
    compiled = dict(encode=evaluator(model, model.encode), decode=evaluator(model, model.decode),
                    advance=evaluator(model, model.advance), time_ten=evaluator(model, iterate),
                    height_map=evaluator(model, lambda z: model.encode(meridian(z))))
    return {name: _numpy(f) for name, f in compiled.items()}


def _numpy(function):
    return lambda values: np.asarray(function(jnp.asarray(values)))


def _batched(function, values, size=8192):
    return np.concatenate([function(values[start:start + size]) for start in range(0, len(values), size)])


def _invert(height_map, u):
    """The height z with E(meridian(z)) = u, for an increasing encoder."""
    if abs(u) == 1:
        return float(u)
    return float(brentq(lambda z: float(height_map(np.array([z]))[0]) - u, -1., 1., xtol=1e-13))


# Diagnostics.

def profile(model):
    """The encoder along a meridian and the derivatives of E and f on a sweep of heights.

    The sweep has 4,001 equally spaced heights and 200 heights on each side with
    log10(1 - |z|) equally spaced in [-10, -2].
    """
    maps = functions(model)
    derivative_e = evaluator(model, jax.vmap(jax.grad(lambda z: model.encode(meridian(z[None]))[0])))
    derivative_f = evaluator(model, jax.vmap(jax.grad(lambda u: model.advance(u[None])[0])))
    sweep = np.unique(np.r_[np.linspace(-1, 1, 4001),
                            1 - 10.**np.linspace(-10, -2, 200), -1 + 10.**np.linspace(-10, -2, 200)])
    inside = sweep[(sweep > -1) & (sweep < 1)]
    return dict(physical_height=sweep, encoded_height=maps["height_map"](sweep),
                interior_height=inside, encoder_derivative=np.asarray(derivative_e(jnp.asarray(inside))),
                time_map_derivative=np.asarray(derivative_f(jnp.asarray(sweep))),
                displacement=maps["advance"](sweep) - sweep)


def diagnostics(model, data):
    """Sampled errors of the fit on the test split and sampled checks of E and f.

    `forecasts[h]` compares the decoded h-step prediction d(f^h(E(p))) with the orbit of
    the state at time h: the root-mean-square height error, the orbit (chordal) error
    and the largest height error. The manuscript quotes `forecasts["10"]["orbit_chordal_rms"]`.
    """
    maps = functions(model)
    encode, decode, advance, height_map = maps["encode"], maps["decode"], maps["advance"], maps["height_map"]
    encoded = encode(data["test_p0"])
    values = encoded
    forecasts = {}
    for horizon in range(11):
        if horizon in (0, 1, 10):
            decoded = decode(values)
            target = data[f"test_p{horizon}"][:, 2]
            radius = np.sqrt(np.maximum(0., (1 - decoded) * (1 + decoded)))
            target_radius = np.linalg.norm(data[f"test_p{horizon}"][:, :2], axis=1)
            forecasts[str(horizon)] = dict(
                height_rms=float(np.sqrt(np.mean((decoded - target)**2))),
                orbit_chordal_rms=float(np.sqrt(np.mean((decoded - target)**2 + (radius - target_radius)**2))),
                height_maximum=float(np.max(np.abs(decoded - target))))
        if horizon < 10:
            values = advance(values)
    residual = advance(encoded) - encode(data["test_p1"])
    residual_ten = values - encode(data["test_p10"])
    errors = [float(np.max(np.abs(encode(rotate(data["test_p0"], angle)) - encoded))) for angle in (.13, .731, 1.7)]
    sweep = profile(model)
    heights, displacement = sweep["physical_height"], sweep["displacement"]
    roots = [-1., 1.]
    for i in np.flatnonzero(displacement[:-1] * displacement[1:] < 0):
        roots.append(float(brentq(lambda u: float(advance(np.array([u]))[0]) - u,
                                  heights[i], heights[i + 1], xtol=1e-13)))
    roots.extend(map(float, heights[(displacement == 0) & (np.abs(heights) < 1 - 1e-7)]))
    fixed_points = [dict(latent=u, physical_height=_invert(height_map, u), decoded_height=float(decode(np.array([u]))[0]))
                    for u in sorted(set(roots))]
    return dict(forecasts=forecasts, sampled_encoder_invariance_error=max(errors),
                sampled_residual_rms=float(np.sqrt(np.mean(residual**2))),
                sampled_residual_maximum=float(np.max(np.abs(residual))),
                sampled_time_ten_residual_rms=float(np.sqrt(np.mean(residual_ten**2))),
                sampled_time_ten_residual_maximum=float(np.max(np.abs(residual_ten))),
                minimum_sampled_encoder_derivative=float(sweep["encoder_derivative"].min()),
                minimum_sampled_time_map_derivative=float(sweep["time_map_derivative"].min()),
                encoder_endpoints=sweep["encoded_height"][[0, -1]].tolist(),
                time_map_endpoints=advance(np.array([-1., 1.])).tolist(),
                sampled_encoder_increasing=bool(np.all(np.diff(sweep["encoded_height"]) > 0)),
                sampled_time_map_increasing=bool(np.all(sweep["time_map_derivative"] > 0)),
                observed_sign_change_fixed_points=fixed_points,
                evidence_status="sampled diagnostics; no uniform residual certificate")


# Morse graphs and attracting regions.

REFERENCE_HEIGHTS = {"S": -1., "A-": -.4, "C": 0., "A+": .4, "N": 1.}


def roles(model, graph, radius=4):
    """Give each Morse set the reference invariant set it lies next to.

    The reference sets are the poles S and N, the attracting latitudes A- and A+ and the
    equator C, at the heights of REFERENCE_HEIGHTS, whose codes are E of those latitudes. A
    Morse set takes the role of the nearest code when all its cells lie within `radius`
    cells of it, and None otherwise. Returns {node: role}.
    """
    height_map = functions(model)["height_map"]
    codes = {name: float(np.asarray(height_map(np.array([height]))).ravel()[0])
             for name, height in REFERENCE_HEIGHTS.items()}
    side = float(graph.boxes[0, 1] - graph.boxes[0, 0])
    result = {}
    for row in graph.nodes:
        centers = graph.boxes[row["cell_ids"]].mean(axis=1)
        distance = {name: np.abs(centers - code).max() / side for name, code in codes.items()}
        name = min(distance, key=distance.get)
        result[row["node"]] = name if distance[name] < radius else None
    return result


def learned_graph(model, settings=DEFAULTS):
    """The Morse graph of f^10 on the grid of the paper, from sampled image hulls, and its
    merged graph.

    Each hull contains the images of `image_samples` equally spaced points of the cell,
    its endpoints included, padded by `image_padding` and cut at the poles. The Morse sets
    that lie next to the same reference invariant set (`roles`) are merged with all cells
    on paths between them, and CMGDB recomputes the index of the merged set; the merged
    graph is None when a Morse set lies next to no reference set or a reference set has
    no Morse set. Each node of the merged graph carries its role.
    """
    analysis = settings["analysis"]
    time_map = functions(model, analysis["time_map_iterates"])["time_ten"]
    boxes = grids.cells(analysis["cells"], LOWER, UPPER)
    images = grids.sampled_hulls(lambda u: _batched(time_map, u[:, 0])[:, None], boxes,
                                 analysis["image_samples"], analysis["image_padding"], LOWER, UPPER)
    raw = grids.morse_graph(boxes, images)
    role = roles(model, raw)
    for row in raw.nodes:
        row["role"] = role[row["node"]]
    if None in role.values() or set(role.values()) != set(REFERENCE_HEIGHTS):
        return raw, None
    groups = [[node for node, name in role.items() if name == reference] for reference in REFERENCE_HEIGHTS]
    merged = grids.merge(raw, groups)
    for row in merged.nodes:
        row["role"] = role[row["original_nodes"][0]]
    return raw, merged


def lifted_grid(model, settings=DEFAULTS):
    """The encoder preimages of the cell boundaries on a meridian.

    For an increasing encoder, the cell [u_i, u_{i+1}] pulls back to the band of S^2
    between the heights z_i and z_{i+1} with E(meridian(z_i)) = u_i. Returns the
    latent boundaries, the physical heights and the residual E(meridian(z_i)) - u_i.
    """
    height_map = functions(model)["height_map"]
    boundaries = np.linspace(-1, 1, settings["analysis"]["cells"] + 1)
    physical = np.asarray([_invert(height_map, u) for u in boundaries])
    return dict(latent_boundaries=boundaries, physical_boundaries=physical,
                encoder_inverse_residual=height_map(physical) - boundaries)


def attracting_regions(model, graph, settings=DEFAULTS):
    """Tolerances of the attracting regions of the minimal Morse sets and the sampled residuals.

    `graph` is the learned graph of `model`. The residual modulo G of f^10 at a start p
    is the distance between f^10(E(p)) and E(F(p)); it is maximized over the meridian
    samples whose code lies in the region. Returns the rows of `regions.compare` in the
    Euclidean metric of the learned interval (the values quoted in the manuscript),
    followed by the rows in the chordal metric of the sphere-height lift, and a dict of
    the region labels, samples and residuals.
    """
    analysis = settings["analysis"]
    maps = functions(model, analysis["time_map_iterates"])
    labels = regions.attracting_regions(graph, FACES)
    theta, starts, physical_images = meridian_samples(analysis)
    codes = _batched(maps["encode"], starts)
    predicted = _batched(maps["time_ten"], codes)
    encoded_image = _batched(maps["encode"], physical_images)
    rows, residuals = [], {}
    for metric in (regions.Euclidean(), regions.SphereHeight()):
        part, residuals[metric.name] = regions.compare(graph, labels, codes[:, None], predicted[:, None],
                                                       encoded_image[:, None], metric, FACES)
        rows += [dict(row, map_time=float(analysis["time_map_iterates"])) for row in part]
    return rows, dict(labels=labels, theta=theta, codes=codes, predicted=predicted,
                      encoded_image=encoded_image, residuals=residuals)


# Roles, comparison with the reference, and indices in S^2.

ORDER = ("S", "A-", "C", "A+", "N")
NAMES = {"S": "-e3", "A-": "A-", "C": "C", "A+": "A+", "N": "e3"}  # as the text of Section 5.1 names them
TEX = {"S": "$S$", "A-": "$A_-$", "C": "$C$", "A+": "$A_+$", "N": "$N$"}
POLES = [(0, "lower"), (0, "upper")]  # P/G = {-1, 1}, the facets of [-1, 1] that the poles form
# The indices in S^2 proved on explicit index pairs (Appendix supp:sphere-pairs), as CMGDB shift
# classes in degrees 0, 1, 2: x-1 is the field F_5 with the identity map, 0 the zero group.
PROVED = {"S": ["0", "0", "x-1"], "A-": ["x-1", "x-1", "0"], "C": ["0", "x-1", "x-1"],
          "A+": ["x-1", "x-1", "0"], "N": ["0", "0", "x-1"]}


def reference_roles(graph):
    """The reference invariant set whose height lies in the cells of each Morse set.

    For the reference graph each Morse set contains exactly one of the heights of
    REFERENCE_HEIGHTS; ValueError otherwise. Returns {node: role}.
    """
    result = {}
    for row in graph.nodes:
        lo, hi = row["hull"]
        inside = [name for name, height in REFERENCE_HEIGHTS.items() if lo <= height <= hi]
        if len(inside) != 1:
            raise ValueError(f"The Morse set {row['node']} contains the reference sets {inside}.")
        result[row["node"]] = inside[0]
    if sorted(result.values()) != sorted(REFERENCE_HEIGHTS):
        raise ValueError("The reference graph does not have one Morse set for each reference set.")
    return result


def node_roles(graph):
    """{node: role}: the roles carried by a learned graph, or those of `reference_roles`."""
    if all("role" in row for row in graph.nodes):
        return {row["node"]: row["role"] for row in graph.nodes}
    return reference_roles(graph)


def _unique_roles(graph):
    roles = node_roles(graph)
    if None in roles.values() or len(set(roles.values())) != len(roles):
        raise ValueError("Each Morse set needs its own role; use the merged graph.")
    return roles


def role_edges(graph):
    """The edges of a Morse graph as pairs of roles, in the order of ORDER."""
    roles = node_roles(graph)
    return sorted(([roles[a], roles[b]] for a, b in graph.edges), key=lambda e: (ORDER.index(e[0]), ORDER.index(e[1])))


def compare_graphs(reference, learned):
    """Compare the learned Morse graph with the reference graph role by role.

    The graphs agree when they have the same roles, the same annotations at each role and
    the same edges between roles.
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
    """The Conley indices in S^2 of the Morse sets of each graph, by Theorem thm:sphere-reconstruction.

    `graphs` maps a name to a MorseGraph on the grid of [-1, 1]. For each Morse set,
    `homology.reconstructed_index` evaluates the theorem on the index pair (|M u F(M)|,
    |F(M) minus M|) of the outer approximation F, with P/G = {-1, 1}, m = 1 (the orbits are
    circles) and degree 1. The quotient index it computes is checked against CMGDB's
    annotation. Returns {name: {role: record}} with the shift classes of the quotient index,
    of the index relative to N_P/G, and of the index in S^2.
    """
    result = {}
    for name, graph in graphs.items():
        roles = _unique_roles(graph)
        grid = homology.Grid((len(graph.boxes),))
        outer = homology.BoxMap.from_relation(grid, graph.offsets, graph.targets)
        result[name] = {}
        for row in sorted(graph.nodes, key=lambda row: ORDER.index(roles[row["node"]])):
            answer = homology.reconstructed_index(outer, row["cell_ids"], fixed=POLES, m=1)
            quotient = answer.quotient.shift_class()
            if quotient != list(row["annotations"]):
                raise RuntimeError(f"{name} {roles[row['node']]}: the quotient index {quotient} differs from "
                                   f"CMGDB's annotation {row['annotations']}.")
            result[name][roles[row["node"]]] = dict(
                cells=list(row["cell_ids"]), index_pair_cells=grid.ids(answer.N),
                exit_cells=grid.ids(answer.L), fixed_point_cells=len(answer.N_P), quotient=quotient,
                relative=answer.relative.shift_class(), in_X=answer.lifted.shift_class(),
                proved=PROVED[roles[row["node"]]],
                agree=answer.lifted.shift_class() == PROVED[roles[row["node"]]])
    return result


def homology_degrees(index):
    """The degrees in which a shift class is nonzero, as in 'H_1 = H_2 = F'."""
    degrees = [str(k) for k, value in enumerate(index) if value != "0"]
    return " = ".join(f"H_{k}" for k in degrees) + " = F" if degrees else "trivial"


# Figures.

def _view(elevation=8., azimuth=-60.):
    """The view direction and the screen axes of `plotting.sphere_bands`."""
    elevation, azimuth = np.deg2rad(elevation), np.deg2rad(azimuth)
    direction = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                          np.sin(elevation)])
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    return direction, horizontal, np.cross(direction, horizontal)


def _limb(z):
    """The screen position of the right end of the circle of latitude of height z."""
    direction, horizontal, vertical = _view()
    phi = np.arctan2(horizontal[1], horizontal[0])
    point = np.array([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z])
    return float(point @ horizontal), float(point @ vertical)


def morse_bands(graph, boundaries):
    """Bands of latitude (z_lower, z_upper, role) covered by the Morse sets of a graph.

    `boundaries` are the heights of the cell boundaries: the grid itself for the reference
    graph, the encoder preimages of `lifted_grid` for a learned graph. Adjacent cells of one
    Morse set form one band.
    """
    roles, boundaries, bands = node_roles(graph), np.asarray(boundaries, float), []
    for row in graph.nodes:
        ids = sorted(row["cell_ids"])
        start = ids[0]
        for a, b in zip(ids, ids[1:] + [None]):
            if b != a + 1:
                bands.append((float(boundaries[start]), float(boundaries[a + 1]), roles[row["node"]]))
                start = b
    return bands


def sphere_panel(bands, path):
    """The bands on the sphere in the colors of their roles, with the latitudes labeled."""
    plt = plotting.style()
    fig, ax = plt.subplots(figsize=(2.3, 2.5))
    plotting.sphere_bands(ax, [(lo, hi, plotting.color(role)) for lo, hi, role in bands])
    for role in ("A-", "C", "A+"):
        heights = [(lo + hi) / 2 for lo, hi, name in bands if name == role]
        if heights:
            x, y = _limb(float(np.mean(heights)))
            ax.text(x + .07, y, TEX[role], color=plotting.color(role), ha="left", va="center")
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def graph_panel(graph, annotations, path):
    """The Morse graph with a label `role: (index)` at each node, the nodes in order of height."""
    roles = node_roles(graph)
    ordered = sorted(graph.nodes, key=lambda row: row["hull"][0])
    nodes = {row["node"]: (plotting.label(roles[row["node"]], annotations[roles[row["node"]]]), roles[row["node"]])
             for row in ordered}
    return plotting.graph(nodes, graph.edges, path)


def interval_panel(graph, xlabel, path):
    """The cells of the Morse sets in the interval, labeled by role."""
    plt = plotting.style()
    roles = node_roles(graph)
    fig, ax = plt.subplots(figsize=(4.6, .8))
    plotting.interval_strip(ax, graph.boxes, roles, xlabel, graph.membership)
    for row in graph.nodes:
        role = roles[row["node"]]
        ax.text(float(np.mean(row["hull"])), .24, TEX[role], color=plotting.color(role), ha="center", va="bottom")
    ax.set_xticks([-1, 0, 1])
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def learned_height_curve(model, graph):
    """The curve z -> d(f^10(E(p_z))), p_z the point of height z on a meridian, and its fixed points.

    The curve is sampled at 1,601 equally spaced heights and 200 heights toward each pole,
    1 - |z| = 10^-s with s equally spaced in [1, 12]. Its fixed points are the poles and the
    roots of the curve minus the identity at its sign changes (Brent, to 1e-14). Each fixed
    point carries its slope (central difference 1e-6, one-sided at the poles) and the role of
    the Morse set of `graph` whose cells contain its code E(p_z), or None.
    """
    maps = functions(model)
    curve = lambda z: maps["decode"](maps["time_ten"](maps["height_map"](np.asarray(z, float))))  # noqa: E731
    near = 1 - 10.**-np.linspace(1, 12, 200)
    z = np.unique(np.r_[np.linspace(-1, 1, 1601), near, -near])
    values = curve(z)
    excess = values - z
    roots = [-1., 1.]
    for i in np.flatnonzero(excess[:-1] * excess[1:] < 0):
        roots.append(float(brentq(lambda x: float(curve(np.array([x]))[0]) - x, z[i], z[i + 1], xtol=1e-14)))
    index = grids.CellIndex(graph.boxes[:, :1], graph.boxes[:, 1:])
    roles, h = node_roles(graph), 1e-6
    fixed = []
    for root in sorted(roots):
        lo, hi = max(-1., root - h), min(1., root + h)
        slope = float((curve(np.array([hi]))[0] - curve(np.array([lo]))[0]) / (hi - lo))
        code = float(maps["height_map"](np.array([root]))[0])
        cells = [c for c in index.containing(np.array([[code]]), slack=0)[0] if c >= 0]
        nodes = {int(graph.membership[c]) for c in cells} - {-1}
        fixed.append(dict(height=root, code=code, slope=slope,
                          role=roles[nodes.pop()] if len(nodes) == 1 else None))
    return dict(height=z, value=values, fixed_points=fixed)


def _fixed_point_marks(ax, points):
    for point in points:
        ink = plotting.color(point["role"])
        ax.plot(point["height"], point["height"], "o", ms=4.2, mec=ink, mew=1.,
                mfc=ink if point["slope"] < 1 else plotting.WHITE, zorder=3, clip_on=False)


def height_map_panel(model, graph, path):
    """The learned height map d(f^10(E(p_z))) against z, with its fixed points.

    A fixed point is drawn in the color of the Morse set whose cells contain its code,
    filled where the slope is below one (attracting) and open otherwise.
    """
    plt = plotting.style()
    curve = learned_height_curve(model, graph)
    fig, ax = plt.subplots(figsize=(2.35, 2.2))
    ax.plot([-1, 1], [-1, 1], color=plotting.LIGHT, lw=1)
    ax.plot(curve["height"], curve["value"], color=plotting.INK, lw=1)
    _fixed_point_marks(ax, curve["fixed_points"])
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), xticks=[-1, 0, 1], yticks=[-1, 0, 1], aspect="equal",
           xlabel="$z$", ylabel="$d(f^{10}(E(p_z)))$")
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def height_flow(z, duration):
    """h_t(z), the flow of z' = z (0.4^2 - z^2)(1 - z^2) at time `duration`, integrated in arctanh(z)."""
    z = np.asarray(z, float)
    out = z.copy()
    inside = np.abs(z) < 1
    out[inside] = np.tanh(_flow(np.arctanh(z[inside]), float(duration))[:, -1])
    return out


def physical_height_panel(path, duration=3.):
    """The physical height map h_3 against z, with the fixed heights of the reference sets."""
    plt = plotting.style()
    near = 1 - 10.**-np.linspace(1, 12, 100)
    z = np.unique(np.r_[np.linspace(-1, 1, 1401), near, -near])
    fig, ax = plt.subplots(figsize=(2.35, 2.2))
    ax.plot([-1, 1], [-1, 1], color=plotting.LIGHT, lw=1)
    ax.plot(z, height_flow(z, duration), color=plotting.INK, lw=1)
    _fixed_point_marks(ax, [dict(height=height, role=role, slope=.5 if role in ("A-", "A+") else 2.)
                            for role, height in REFERENCE_HEIGHTS.items()])
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), xticks=[-1, -.4, 0, .4, 1], yticks=[-1, 0, 1],
           xticklabels=["$-1$", "$-0.4$", "$0$", "$0.4$", "$1$"], aspect="equal", xlabel="$z$",
           ylabel=f"$h_{{{duration:g}}}(z)$")
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


TRAJECTORY_STARTS = ((-.93, .5), (-.72, 2.5), (-.12, 4.), (.12, 1.), (.72, 3.), (.93, 5.))


def trajectories_panel(path, duration=18., samples=1201):
    """Six trajectories of V on the visible hemisphere, with the latitudes z = -0.4, 0, 0.4.

    Each trajectory starts at a height z0 and a longitude phi0 of TRAJECTORY_STARTS; its
    height follows h_t and its longitude turns at rate 0.6. A dot marks the start and an
    arrow the direction of motion at t = 3.
    """
    plt = plotting.style()
    direction, horizontal, vertical = _view()
    fig, ax = plt.subplots(figsize=(2.3, 2.5))
    plotting.sphere_bands(ax, [])
    times = np.linspace(0., duration, samples)
    starts = np.array(TRAJECTORY_STARTS)
    w = _flow(np.arctanh(starts[:, 0]), duration, t_eval=times)
    for latitude, phi0 in zip(w, starts[:, 1], strict=True):
        phi = phi0 + .6 * times
        points = np.column_stack((np.cos(phi) / np.cosh(latitude), np.sin(phi) / np.cosh(latitude), np.tanh(latitude)))
        visible = points @ direction > 0
        x, y = points @ horizontal, points @ vertical
        ax.plot(np.where(visible, x, np.nan), np.where(visible, y, np.nan), color=plotting.GRAY, lw=.7)
        if visible[0]:
            ax.plot(x[0], y[0], "o", color=plotting.INK, ms=2.5)
        arrow = next((i for i in range(samples // 6, samples - 1) if visible[i] and visible[i + 1]), None)
        if arrow is not None:
            ax.annotate("", xy=(x[arrow + 1], y[arrow + 1]), xytext=(x[arrow], y[arrow]),
                        arrowprops=dict(arrowstyle="-|>", color=plotting.GRAY, lw=.7, mutation_scale=7))
    phi = np.linspace(0, 2 * np.pi, 721)
    for role in ("A-", "C", "A+"):
        z = REFERENCE_HEIGHTS[role]
        circle = np.column_stack((np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), np.full_like(phi, z)))
        visible = circle @ direction > 0
        ax.plot(np.where(visible, circle @ horizontal, np.nan), np.where(visible, circle @ vertical, np.nan),
                color=plotting.color(role), lw=1.3, ls="--" if role == "C" else "-")
        x, y = _limb(z)
        ax.text(x + .07, y, TEX[role], color=plotting.color(role), ha="left", va="center")
    path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def _panels(paths, folder):
    """{manuscript name: PDF path} for panels saved under `folder`."""
    return {path.relative_to(folder).with_suffix("").as_posix(): path for path in paths}


def figure_sphere_computation(model, merged, folder):
    """fig:sphere-computation: the learned height map and the Morse graph of f^10 in [-1, 1].

    Returns {panel name: PDF path}, the names those of the manuscript's panel files.
    """
    folder = Path(folder)
    annotations = {row["role"]: row["annotations"] for row in merged.nodes}
    return _panels([height_map_panel(model, merged, folder / "dynamics/learned-height-map"),
                    graph_panel(merged, annotations, folder / "sphere/learned-conley-morse-interval")], folder)


def figure_sphere_reconstruction(model, merged, indices, folder):
    """fig:sphere-reconstruction: E^{-1}(|M|) in S^2 and the reconstructed Conley-Morse graph in S^2.

    The bands are the preimages of the cells of the merged Morse sets under the encoder, by
    inversion along a meridian (`lifted_grid`); the indices are those of `reconstructed_indices`.
    """
    folder = Path(folder)
    boundaries = lifted_grid(model)["physical_boundaries"]
    in_x = {role: row["in_X"] for role, row in indices["learned"].items()}
    return _panels([sphere_panel(morse_bands(merged, boundaries), folder / "sphere/learned-sphere"),
                    graph_panel(merged, in_x, folder / "sphere/learned-conley-morse-sphere")], folder)


def figure_sphere_height(folder):
    """fig:appendix-sphere-height: trajectories on the sphere and the physical height map h_3."""
    folder = Path(folder)
    return _panels([trajectories_panel(folder / "dynamics/sphere-trajectories"),
                    physical_height_panel(folder / "dynamics/sphere-height-map")], folder)


def figure_sphere_comparison(model, reference, merged, indices, folder):
    """fig:appendix-sphere-comparison: the reference (left) and the learned map (right).

    The rows are the preimages of the Morse sets on the sphere, their cells in the interval,
    the Morse graphs with their indices in [-1, 1], and the Morse graphs with the indices in
    S^2 reconstructed from them.
    """
    folder = Path(folder)
    grid_heights = np.linspace(-1, 1, len(reference.boxes) + 1)
    learned_heights = lifted_grid(model)["physical_boundaries"]
    paths = [sphere_panel(morse_bands(reference, grid_heights), folder / "sphere/analytical-sphere"),
             sphere_panel(morse_bands(merged, learned_heights), folder / "sphere/learned-sphere"),
             interval_panel(reference, "$z$", folder / "sphere/analytical-interval"),
             interval_panel(merged, "$u$", folder / "sphere/learned-interval")]
    for kind, key in (("quotient", "quotient"), ("reconstructed", "in_X")):
        for name, graph in (("analytical", reference), ("learned", merged)):
            source = "reference" if name == "analytical" else "learned"
            labels = {role: row[key] for role, row in indices[source].items()}
            paths.append(graph_panel(graph, labels, folder / f"sphere/{name}-{kind}"))
    return _panels(paths, folder)


# The record of the results checked in replay.

def weights_digest(path):
    """SHA-256 of the arrays of a weights file in their order, independent of the archive bytes."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as saved:
        for key in sorted(saved.files, key=int):
            value = np.ascontiguousarray(saved[key])
            digest.update(f"{key}:{value.dtype.str}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _graph_record(graph):
    roles = node_roles(graph)
    record = graph.record()
    for row in record["nodes"]:
        row["role"] = roles[row["node"]]
    return record


def record_of(fit, metrics, reference, raw, merged, rows, details, indices):
    """The JSON-ready record of every result the notebook checks, from computed parts.

    `fit` names the fit and the digest of its weights, `metrics` is `diagnostics`, the
    graphs come from `reference_graph` and `learned_graph`, `rows` and `details` from
    `attracting_regions` and `indices` from `reconstructed_indices`.
    """
    return dict(
        example="ex1_so2_flow", fit=dict(fit), diagnostics=metrics,
        graphs=dict(reference=_graph_record(reference), raw=_graph_record(raw), merged=_graph_record(merged)),
        comparison=compare_graphs(reference, merged),
        regions=dict(rows=rows, meridian_samples=int(len(details["theta"])),
                     labels=[int(v) for v in details["labels"]]),
        indices=indices)


EXACT = ("fit", "graphs", "comparison", "indices")  # compared exactly; other numbers to a relative tolerance


def compare_records(new, saved, rtol=1e-9, exact=EXACT):
    """Compare two records: the parts named in `exact` exactly, the other numbers to `rtol`.

    Returns the number of compared values and a list of differences, each a path and both values.
    """
    differences, count = [], 0

    def walk(a, b, path, strict):
        nonlocal count
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append((f"{path}/{key}", a.get(key, "missing"), b.get(key, "missing")))
                else:
                    walk(a[key], b[key], f"{path}/{key}", strict or (path == "" and key in exact))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                differences.append((path, f"{len(a)} entries", f"{len(b)} entries"))
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{path}/{i}", strict)
        else:
            count += 1
            numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
            if numbers and not strict and (isinstance(a, float) or isinstance(b, float)):
                same = a == b or abs(a - b) <= rtol * max(abs(a), abs(b))
            else:
                same = type(a) is type(b) and a == b
            if not same:
                differences.append((path, a, b))

    walk(json.loads(json.dumps(new)), json.loads(json.dumps(saved)), "", False)
    return count, differences


def selected_run(folder):
    """The name of the selected fit in `<folder>/selection.json`."""
    return training.read(Path(folder) / "selection.json")["all"]["selected"]


def reference_record(folder):
    """The record of the selected fit of the data folder `folder`, for `scripts/record_reference.py`."""
    folder = Path(folder)
    settings = training.read(folder / "config.json")
    run = selected_run(folder)
    model, data = load(folder / "models" / run), load_data(folder)
    reference = reference_graph(settings)
    raw, merged = learned_graph(model, settings)
    rows, details = attracting_regions(model, raw, settings)
    indices = reconstructed_indices(dict(reference=reference, learned=merged))
    fit = dict(run=run, weights_sha256=weights_digest(folder / "models" / run / "weights.npz"))
    return record_of(fit, diagnostics(model, data), reference, raw, merged, rows, details, indices)


# Steps of the notebook ex1_so2_flow.ipynb.

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "ex1_so2_flow"


class Table:
    """A table shown by a notebook cell: HTML in Jupyter, aligned text otherwise."""

    def __init__(self, header, rows, caption=None):
        self.header, self.rows, self.caption = [str(h) for h in header], [[str(c) for c in r] for r in rows], caption

    def _repr_html_(self):
        head = "".join(f"<th style='text-align:left'>{escape(h)}</th>" for h in self.header)
        body = "".join("<tr>" + "".join(f"<td style='text-align:left'>{escape(c)}</td>" for c in row) + "</tr>"
                       for row in self.rows)
        title = f"<caption style='text-align:left'>{escape(self.caption)}</caption>" if self.caption else ""
        return f"<table>{title}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def __repr__(self):
        widths = [max(len(r[i]) for r in [self.header] + self.rows) for i in range(len(self.header))]
        lines = ([self.caption] if self.caption else []) + [
            "  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in [self.header] + self.rows]
        return "\n".join(lines)


class Line:
    """A one-line result shown by a notebook cell."""

    def __init__(self, text, passed=True, **values):
        self.text, self.passed, self.values = text, passed, values

    def _repr_html_(self):
        return f"<p>{escape(self.text)}</p>"

    def __repr__(self):
        return self.text


def _fmt(value, digits=4):
    return "none" if value is None else f"{value:.{digits}g}"


def _relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def _tuple(annotations):
    return "(" + ", ".join(annotations) + ")"


def _role_name(role):
    return "none" if role is None else (f"{role} ({NAMES[role]})" if NAMES[role] != role else role)


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
        return self.out / "figures"

    @property
    def fit(self):
        """The fit used: its name and the digest of its weights, stored (replay) or retrained (train)."""
        folder = self.out if self.mode == "train" else self.data / "models"
        return dict(run=self.selected, weights_sha256=weights_digest(folder / self.selected / "weights.npz"))

    def table(self):
        analysis = self.settings["analysis"]
        return Table(["Setting", "Value"], [
            ["Mode", "replay of the selected fit" if self.mode == "replay"
             else "retraining of the selected seed with config.json"],
            ["Selected fit", f"{self.selected}, lowest validation objective ({_relative(self.data / 'selection.json')})"],
            ["Grid", f"2^{int(np.log2(analysis['cells']))} cells of [-1, 1]"],
            ["Map of the Morse graph", f"f^{analysis['time_map_iterates']}, for the learned time-one map f"],
            ["Image hulls", f"{analysis['image_samples']} samples per cell, padding {analysis['image_padding']:g}"],
            ["Output folder", _relative(self.out)]])

    def _repr_html_(self):
        return self.table()._repr_html_()

    def __repr__(self):
        return repr(self.table())


def start(mode="replay", out=None):
    """Check the mode, read the selection and settings, and make a new output folder.

    `out` None makes `outputs/notebooks/ex1_so2_flow-<mode>-<time>`; a relative `out` is taken
    from the repository root. The folder must not exist yet.
    """
    if mode not in ("replay", "train"):
        raise ValueError('MODE is "replay" or "train".')
    data = ROOT / "data" / EXAMPLE
    if out is None:
        out = ROOT / "outputs" / "notebooks" / f"{EXAMPLE}-{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = Path(out) if Path(out).is_absolute() else ROOT / out
    if out.exists():
        raise FileExistsError(f"{_relative(out)} exists; choose a new OUT.")
    (out / "figures").mkdir(parents=True)
    return Run(mode, data, selected_run(data), training.read(data / "config.json"), out)


def system_table():
    """Sampled checks of the system and the explicit enclosures of the reference quotient map."""
    rng = np.random.default_rng(0)
    p = rng.normal(size=(1000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    equivariance = max(float(np.max(np.abs(vector_field(rotate(p, a)) - rotate(vector_field(p), a))))
                       for a in (.3, 1.7, 4.))
    rate = float(vector_field(np.array([np.sqrt(.75), 0., .5]))[2])
    boundaries = np.linspace(-1, 1, DEFAULTS["analysis"]["cells"] + 1)
    widths = [hi - lo for lo, hi in map(time_ten, boundaries)]
    exact = all(time_ten(z) == (z, z) for z in (-1., 0., 1.))
    return Table(["Check", "Value"], [
        ["Equivariance of V: max |V(R p) - R V(p)| over 1,000 points and 3 rotations", _fmt(equivariance, 3)],
        ["Height rate z(0.4^2 - z^2)(1 - z^2) at z = 1/2", _fmt(rate, 6)],
        ["Widest enclosure of h_10 at the 129 cell boundaries of the grid", _fmt(max(widths), 3)],
        ["h_10 fixes -1, 0 and 1 exactly", "yes" if exact else "no"]],
        "The reference quotient map h_10 is enclosed by bisection, in exact rational arithmetic, on "
        "Q(z) = w^21 (1 - w)^4 / |w - 4/25|^25, w = z^2, which satisfies Q(h_t(z)) = exp(168 t/25) Q(z).")


def graph_table(graph, coordinate, caption=None):
    """The Morse sets of a graph: cells, interval, Conley index annotation, role and edges."""
    roles = node_roles(graph)
    successors = {}
    for a, b in graph.edges:
        successors.setdefault(a, []).append(f"{b} ({roles[b]})")
    rows = []
    for row in sorted(graph.nodes, key=lambda row: row["hull"][0]):
        cells = row["cell_ids"]
        rows.append([row["node"], _role_name(roles[row["node"]]),
                     ", ".join(map(str, row.get("original_nodes", []))) or "",
                     ", ".join(map(str, cells)), f"[{row['hull'][0]:.6g}, {row['hull'][1]:.6g}]",
                     _tuple(row["annotations"]), ", ".join(sorted(successors.get(row["node"], []))) or "none"])
    header = ["Morse set", "Role", "Merged Morse sets", "Cells", f"Interval in {coordinate}", "Conley index", "Edges to"]
    if not any(r[2] for r in rows):
        header.pop(2)
        rows = [r[:2] + r[3:] for r in rows]
    return Table(header, rows, caption)


def data_table(data, folder):
    """The splits of the data: starts, the share near the poles, and the times."""
    metadata = training.read(Path(folder) / "metadata.json")
    rows = []
    for split in SPLITS:
        z = data[f"{split}_p0"][:, 2]
        rows.append([split, len(z), f"{np.mean(1 - np.abs(z) <= 1e-2):.3f}",
                     ", ".join(f"{t:g}" for t in metadata["times"])])
    return Table(["Split", "Starts", "Share within 0.01 of a pole", "Times"], rows,
                 f"Generated by {metadata['generator']}; every array matches its SHA-256 checksum in "
                 f"{_relative(Path(folder) / 'metadata.json')}.")


def seeds_table(run):
    """All fits of the sweep, ranked by validation objective, with their stored held-out errors."""
    selection = training.read(run.data / "selection.json")["all"]
    rows = []
    for rank, entry in enumerate(selection["ranking"], 1):
        folder = run.data / "models" / entry["run"]
        config, metrics = training.read(folder / "config.json"), training.read(folder / "metrics.json")
        rows.append([rank, entry["run"] + (" (selected)" if entry["run"] == selection["selected"] else ""),
                     _fmt(entry["validation"]), config["best_step"],
                     _fmt(metrics["forecasts"]["1"]["orbit_chordal_rms"]),
                     _fmt(metrics["forecasts"]["10"]["orbit_chordal_rms"]),
                     "yes" if metrics["sampled_encoder_increasing"] else "no",
                     _fmt(metrics["minimum_sampled_encoder_derivative"], 3),
                     len(metrics["observed_sign_change_fixed_points"])])
    return Table(["Rank", "Fit", "Validation objective", "Selected step", "One-step orbit RMS",
                  "Ten-step orbit RMS", "E increasing", "Least slope of E", "Fixed points of f"], rows,
                 f"Selection by {selection['criterion']}. Held-out errors on the test starts, from the stored "
                 "metrics of each fit.")


def selected_fit(run, data):
    """The selected fit: loaded from `data/ex1_so2_flow/models` in replay, retrained in train mode.

    Retraining uses the seed of the selected fit and `config.json`, on the CPU, and writes
    into `<out>/<fit>`. Returns the model and a line that reports the fit.
    """
    stored = run.data / "models" / run.selected
    config = training.read(stored / "config.json")
    if run.mode == "replay":
        model = load(stored)
        return model, Line(f"Loaded {run.selected} from {_relative(stored)}: selected at step {config['best_step']:,} "
                           f"with validation objective {config['best_validation']:.6g}; weights digest "
                           f"{weights_digest(stored / 'weights.npz')[:16]}.")
    folder = run.out / run.selected
    model = train(folder, data, config["seed"], run.settings)
    new = training.read(folder / "config.json")
    same = weights_digest(folder / "weights.npz") == weights_digest(stored / "weights.npz")
    with np.load(folder / "weights.npz") as a, np.load(stored / "weights.npz") as b:
        gap = max(float(np.max(np.abs(a[k] - b[k]), initial=0.)) for k in a.files)
    return model, Line(f"Trained seed {config['seed']} for {run.settings['training']['steps']:,} updates in "
                       f"{new['seconds']:.0f} s; selected at step {new['best_step']:,} with validation objective "
                       f"{new['best_validation']:.6g}. The weights " +
                       ("equal those of the stored fit bitwise." if same else
                        f"differ from those of the stored fit by at most {gap:.3g}."), passed=same)


def diagnostics_table(metrics):
    """The held-out orbit errors and the sampled checks of the encoder and the latent map."""
    forecasts = metrics["forecasts"]
    rows = [[f"Orbit RMS of d(f^{h}(E(p))) against the orbit at time {h}", _fmt(forecasts[h]["orbit_chordal_rms"]),
             _fmt(forecasts[h]["height_rms"]), _fmt(forecasts[h]["height_maximum"])] for h in ("0", "1", "10")]
    rows += [["Invariance of E: max |E(R p) - E(p)|, 3 rotations", _fmt(metrics["sampled_encoder_invariance_error"], 3), "", ""],
             ["E increasing along a meridian; least slope", "yes" if metrics["sampled_encoder_increasing"] else "no",
              _fmt(metrics["minimum_sampled_encoder_derivative"], 3), ""],
             ["f increasing; least slope", "yes" if metrics["sampled_time_map_increasing"] else "no",
              _fmt(metrics["minimum_sampled_time_map_derivative"], 3), ""],
             ["Residual |f^10(E(p)) - E(F(p))|: RMS and maximum on the test starts",
              _fmt(metrics["sampled_time_ten_residual_rms"]), _fmt(metrics["sampled_time_ten_residual_maximum"]), ""]]
    fixed = ", ".join(f"{row['physical_height']:.4f}" for row in metrics["observed_sign_change_fixed_points"])
    rows.append(["Heights z with E(p_z) a fixed point of f", fixed, "", ""])
    return Table(["Quantity", "Orbit or value", "Height RMS", "Height maximum"], rows,
                 "Test split, 4,096 starts. The manuscript quotes the ten-step orbit RMS.")


def comparison_table(comparison):
    """The learned Morse graph against the reference, role by role, and the order."""
    rows = []
    for row in comparison["nodes"]:
        rows.append([_role_name(row["role"]), ", ".join(map(str, row["reference_cells"] or [])),
                     _tuple(row["reference_annotations"] or []),
                     ", ".join(map(str, row["learned_cells"] or [])),
                     ", ".join(map(str, row["learned_morse_sets"] or [])),
                     _tuple(row["learned_annotations"] or []), "yes" if row["agree"] else "no"])
    order = lambda edges: ", ".join(f"{NAMES[a]} -> {NAMES[b]}" for a, b in edges)  # noqa: E731
    rows.append(["Order", order(comparison["reference_edges"]), "", order(comparison["learned_edges"]), "", "",
                 "yes" if comparison["edges_agree"] else "no"])
    return Table(["Role", "Reference cells", "Reference index", "Learned cells", "Raw Morse sets",
                  "Learned index", "Agree"], rows,
                 "The graphs agree" if comparison["agree"] else "The graphs differ")


def regions_table(rows, graph):
    """Tolerances of the attracting regions and the sampled residuals modulo G."""
    roles = node_roles(graph)
    body = []
    for row in rows:
        body.append([row["metric"], _role_name(roles[row["node"]]), row["cells"],
                     f"[{row['hull'][0]:.6g}, {row['hull'][1]:.6g}]", f"{row['samples']:,}",
                     _fmt(row["max_residual"]), _fmt(row["tolerance"]), "yes" if row["passed"] else "no"])
    return Table(["Metric", "Minimal Morse set", "Cells of A", "|A| in u", "Samples in E^-1(N)",
                  "Largest residual", "Tolerance", "Residual below tolerance"], body,
                 "Euclidean rows: the metric of the learned interval, the values of Section 5.1. Chordal rows: the "
                 "metric of the sphere heights. Sampled residuals and hulls, binary64, not validated.")


def indices_table(indices):
    """The indices in S^2 reconstructed for the reference and the learned graph, against the proved ones."""
    rows = []
    for role in ORDER:
        ref, new = indices["reference"][role], indices["learned"][role]
        rows.append([_role_name(role), _tuple(PROVED[role]), homology_degrees(PROVED[role]),
                     _tuple(ref["quotient"]), _tuple(ref["in_X"]), _tuple(new["quotient"]), _tuple(new["in_X"]),
                     "yes" if ref["agree"] and new["agree"] else "no"])
    return Table(["Role", "Proved in S^2", "Nonzero groups", "Reference in [-1, 1]", "Reference in S^2",
                  "Learned in [-1, 1]", "Learned in S^2", "Agree"], rows,
                 "Shift classes over F_5 in degrees 0, 1, 2. The quotient indices computed here equal CMGDB's "
                 "annotations. The reference reconstruction uses the enclosures of h_10; the learned one uses "
                 "sampled hulls, so its indices are observed.")


def check_record(run, record):
    """Compare the record with `data/ex1_so2_flow/reference/results.json`.

    The record is written to `<out>/record.json`. In replay any difference raises
    AssertionError; in train mode the differences are reported.
    """
    path = run.data / "reference" / "results.json"
    training.write(run.out / "record.json", record)
    count, differences = compare_records(record, training.read(path))
    text = (f"The record matches {_relative(path)}: {count:,} values, graphs and indices exactly, "
            f"other numbers to relative 1e-9." if not differences else
            f"The record differs from {_relative(path)} in {len(differences)} of {count:,} values, first at "
            f"{differences[0][0]}.")
    if differences and run.mode == "replay":
        raise AssertionError(text + " " + "; ".join(f"{p}: {a!r} != {b!r}" for p, a, b in differences[:10]))
    return Line(text, passed=not differences, compared=count, differences=differences)


def _pair(rows, role, metric):
    return next(row for row in rows if row["role"] == role and row["metric"] == metric)


def summary(run, record, check):
    """Every quantity of Section 5.1 and its appendix, computed and as the manuscript states it.

    The manuscript values describe the fit of its current text; the computed values come from
    the selected fit of this repository. Writes `<out>/verification.json`, which also lists the
    saved panels, and returns the table.
    """
    roles = {row["node"]: row["role"] for row in record["graphs"]["raw"]["nodes"]}
    regions_rows = [dict(row, role=roles[row["node"]]) for row in record["regions"]["rows"]]
    ten = record["diagnostics"]["forecasts"]["10"]["orbit_chordal_rms"]
    comparison, indices = record["comparison"], record["indices"]
    raw, merged = record["graphs"]["raw"], record["graphs"]["merged"]
    order = lambda edges: ", ".join(f"{NAMES[a]} -> {NAMES[b]}" for a, b in edges)  # noqa: E731
    settings = run.settings
    rows = [
        ["Section 5.1", "Ten-step orbit RMS on held-out data", "0.016", _fmt(ten, 3)],
        ["Section 5.1", "Morse graphs of f^10 and h_10 on 2^7 cells agree", "agree",
         ("agree" if comparison["agree"] else "differ") +
         f" after the prescribed merge ({len(raw['nodes'])} raw Morse sets, {len(merged['nodes'])} merged)"],
        ["Section 5.1", "Order", "-e3 -> A-, C -> A-, C -> A+, e3 -> A+", order(comparison["learned_edges"])],
        ["Section 5.1", "Conley indices in [-1, 1] match the reference", "yes",
         "yes" if all(row["agree"] for row in comparison["nodes"]) else "no"]]
    for role, residual, tolerance in (("A-", "0.026", "0.036"), ("A+", "0.033", "0.043")):
        row = _pair(regions_rows, role, "euclidean")
        rows.append(["Section 5.1", f"{NAMES[role]}: largest sampled residual of f^10 against the tolerance",
                     f"{residual} < {tolerance}", f"{_fmt(row['max_residual'], 3)} < {_fmt(row['tolerance'], 3)}"
                     if row["passed"] else f"{_fmt(row['max_residual'], 3)} >= {_fmt(row['tolerance'], 3)}"])
    reconstructed = all(indices[name][role]["agree"] for name in indices for role in indices[name])
    rows += [
        ["Section 5.1", "Indices in S^2: H_2 = F at the poles, H_1 = H_2 = F at C, H_0 = H_1 = F at A-+",
         "proved; reconstruction agrees", "reconstruction agrees for the reference and the learned graph"
         if reconstructed else "reconstruction differs"],
        ["supp:sphere-walkthrough", "Height derivative at z = 1/2", "-0.03375",
         _fmt(float(vector_field(np.array([np.sqrt(.75), 0., .5]))[2]), 6)],
        ["supp:sphere-computation", "Cells of the two attracting regions",
         "41 and 84", " and ".join(str(_pair(regions_rows, r, "euclidean")["cells"]) for r in ("A-", "A+"))],
        ["supp:sphere-computation", "Meridian samples", "69,537", f"{record['regions']['meridian_samples']:,}"]]
    for metric, values in (("euclidean", ("(0.0363731, 0.0255590)", "(0.0433473, 0.0333390)")),
                           ("chordal", ("(0.0390196, 0.0299066)", "(0.0455395, 0.0335467)"))):
        for role, value in zip(("A-", "A+"), values):
            row = _pair(regions_rows, role, metric)
            rows.append(["supp:sphere-computation", f"{NAMES[role]}, {metric} metric: (tolerance, residual)", value,
                         f"({row['tolerance']:.6g}, {row['max_residual']:.6g})"])
    training_settings, sizes = settings["training"], settings["data"]["sizes"]
    rows += [
        ["Saved notebook configurations", "Training starts, updates, batch",
         "4096, 8000, 256", f"{sizes['train']}, {training_settings['steps']}, {training_settings['batch']}"],
        ["Saved notebook configurations", "Learning rate",
         "1e-3 to 2e-5", f"{training_settings['learning_rate_start']:g} to {training_settings['learning_rate_end']:g}"],
        ["fig:appendix-sphere-height", "Trajectories: count, times, angular speed", "6, 0 <= t <= 18, 0.6",
         f"{len(TRAJECTORY_STARTS)}, 0 <= t <= 18, 0.6"],
        ["Replay", "Record against the stored reference", "", "matches" if check.passed else "differs"]]
    verification = dict(example=EXAMPLE, mode=run.mode, fit=record["fit"], record_matches_reference=check.passed,
                        compared_values=check.values.get("compared"),
                        differences=[list(map(str, d)) for d in check.values.get("differences", [])],
                        summary=[dict(zip(("where", "statement", "manuscript", "computed"), row)) for row in rows],
                        figures=sorted(str(p.relative_to(run.figures)) for p in run.figures.rglob("*.pdf")),
                        versions={p: version(p) for p in VERSIONS})
    training.write(run.out / "verification.json", verification)
    return Table(["Where", "Statement", "Manuscript", "Computed"], rows,
                 f"Written to {_relative(run.out / 'verification.json')}. The manuscript values describe the fit "
                 "of its current text.")
