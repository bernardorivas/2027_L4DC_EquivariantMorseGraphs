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
"""
import copy
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # no effect once JAX has started
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

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

from . import grids, regions, training

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

    def encode(self, p):
        return move(p[..., 2], self.encoder_shift(p))

    def decoder_shift(self, u):
        return self.recipe["decoder_shift_bound"] * jnp.tanh(self.decoder(u[..., None])[..., 0])

    def decode(self, u):
        return move(u, self.decoder_shift(u))

    def time_shift(self, u):
        return self.recipe["time_shift_bound"] * jnp.tanh(self.dynamics(u[..., None])[..., 0])

    def advance(self, u):
        return move(u, self.time_shift(u))


def objective(model, p0, p1, w0, w1):
    """Orbit reconstruction, orbit prediction and latent consistency, with an ordering penalty.

    The orbit distance is that of the circles of latitude, (z - z*)^2 + (r - r*)^2,
    plus a multiple of the squared error in log latitude; the losses are evaluated in
    log latitude before returning to heights.
    """
    e0 = w0 + model.encoder_shift(p0)
    u0 = jnp.tanh(e0)
    reconstructed = e0 + model.decoder_shift(u0)
    predicted = e0 + model.time_shift(u0)
    decoded_prediction = predicted + model.decoder_shift(jnp.tanh(predicted))
    e1 = w1 + model.encoder_shift(p1)

    def orbit_error(a, b):
        return jnp.mean((jnp.tanh(a) - jnp.tanh(b))**2 +
                        (1 / jnp.cosh(a) - 1 / jnp.cosh(b))**2 + model.recipe["orbit_log_weight"] * (a - b)**2)

    reconstruction = orbit_error(reconstructed, w0)
    prediction = orbit_error(decoded_prediction, w1)
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

def learned_graph(model, settings=DEFAULTS):
    """The Morse graph of f^10 on the grid of the paper, from sampled image hulls.

    Each hull contains the images of `image_samples` equally spaced points of the cell,
    its endpoints included, padded by `image_padding` and cut at the poles. Section 5.1
    prescribes no merging, so the second value is None.
    """
    analysis = settings["analysis"]
    time_map = functions(model, analysis["time_map_iterates"])["time_ten"]
    boxes = grids.cells(analysis["cells"], LOWER, UPPER)
    images = grids.sampled_hulls(lambda u: _batched(time_map, u[:, 0])[:, None], boxes,
                                 analysis["image_samples"], analysis["image_padding"], LOWER, UPPER)
    return grids.morse_graph(boxes, images), None


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
