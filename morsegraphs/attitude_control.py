"""Section 5.3: reduced attitude control on S^2.

The plant is the rigid-body model of a quadrotor with state (p, v, R, Omega), attitude
stored as a scalar-first unit quaternion, and thrust axis q = R e3 in S^2 with height
z = q . e3. For a gain a, let V_a(q) = a(z)(e3 - z q) + omega e3 x q, w = R Omega,
w_d(q) = q x V_a(q) and eta = w - w_d(q). The feedback has thrust mu g and moment
M = Omega x J Omega + J R^T (Dw_d(q)[w x q] - 2 eta), so the closed loop satisfies
q' = V_a(q) + eta x q and eta' = -2 eta, and on the invariant set {eta = 0} the axis
follows V_a, with height equation z' = a(z)(1 - z^2) and omega = pi / log 2. Rotations
Q about e3 act by (p, v, R, Omega) -> (Qp, Qv, QR, Omega) and q -> Qq, and the feedback
is equivariant. The nominal gain a = 1 makes -e3 repel and e3 attract; the coning gain
a(z) = 1/4 - z makes both poles repel and the latitude C = {z = 1/4} attract every
nonpolar axis. Position is not regulated.

The data are full-plant trajectories that start on {eta = 0}, observed every 0.1 time
units in 18 coordinates (p, v, the columns of R, Omega). The learned model has an
equivariant encoder E into a latent sphere Z, an axis decoder, an equivariant time-0.1
map f of Z that fixes the poles, an increasing coordinate psi of [-1, 1] identifying
Z/G with the box, and an orbit decoder. CMGDB analyzes the tenth iterate of
psi o fbar o psi^-1 on 2^7 cells, where fbar is the map of heights induced by f, and
the reference is the time-one map of the height equation, in z or in the coordinate psi
of a fit. For the coning gain, the Morse sets strictly between the poles are merged with
all cells on directed paths between them, as the structure of the example prescribes,
and CMGDB recomputes the Conley index of the merged set.

The networks run in float32, as when they were fitted: call
`morsegraphs.device.configure("float32")` before anything imports JAX. The plant, the
reference maps, CMGDB and the tolerances use binary64. Image hulls and residuals are
sampled, not validated, and no attracting block or index is claimed for the plant.

The last sections give the indices in X of each Morse set (`indices_in_X`), the record of
every checked result (`Analysis`, `reference_record`), the panels of the figures of Section
5.3 and its supplement, and the steps of the notebook `ex3_attitude_control.ipynb` (`Notebook`).
"""
from contextlib import contextmanager
from functools import cached_property, lru_cache
import hashlib
from html import escape
import importlib.metadata
import inspect
import json
import logging
import math
import os
from pathlib import Path
import time
import warnings
import zipfile

# The fits and their saved results were computed with single-threaded Eigen kernels on the CPU.
# In float32 the threaded kernels sum in another order and change the last digits, so the same
# flags are set here, before JAX starts its backend.
if "xla_cpu_multi_thread_eigen" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2").strip()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

import CMGDB
import jax
import jax.numpy as jnp
import objax
import optax
with warnings.catch_warnings():
    # EMLP imports tqdm.auto, which warns in a notebook without ipywidgets.
    warnings.simplefilter("ignore")
    from emlp.groups import SO2eR3
    from emlp.nn.objax import EMLP, gated
    from emlp.reps import T, V

from . import grids, homology, plotting, regions, training

POLICIES = ("nominal", "coning")
SPLITS = ("train", "validation", "test")

DEFAULTS = dict(
    # Data: complete full-plant trajectories starting on eta = 0, a quarter near the poles.
    policies=["nominal", "coning"],
    trajectories=dict(train=8192, validation=2048, test=4096),
    data_seeds=dict(train=2026111311, validation=2026111312, test=2026111323),
    intervals=32, spacing=0.1, integration_step=0.002,
    # Networks.
    width=64, layers=2, scalar_width=24, knots=33, boost_bound=0.5, angle_bound=1.0,
    coordinate_log_bound=0.35,
    # Training: calibration of the encoder and decoders, then the joint objective.
    steps=6000, warmup_steps=1000, batch=256, learning_rate_start=0.002, learning_rate_end=5e-05,
    validate_every=250, weight_decay=1e-06, gradient_clip=1.0, horizons=[0, 1, 4, 10],
    loss_weights=[1.0, 3.0, 2.0, 0.5, 0.5, 0.5, 0.1], warmup_loss_weights=[1.0, 0.0, 4.0, 0.0, 0.5, 0.0, 0.0],
    batch_seed=10900, validation_window_seed=8181, test_window_seed=9191,
    # Analysis: the tenth iterate on 2^7 cells, with sampled image hulls.
    cells=128, iterates=10, samples_per_cell=257, padding=2e-06,
    # Attracting regions: all one-second windows of the test split and fresh trajectories.
    fresh_trajectories=4096, fresh_seed=2026111341,
)

LOSS_NAMES = ["axis_reconstruction", "axis_prediction", "axis_calibration", "physical_closure",
              "orbit_reconstruction", "orbit_prediction", "quotient_consistency"]
NETWORK_NAMES = ("encoder", "axis_decoder", "time_heads", "coordinate", "orbit_decoder")
FACES = np.array([[True, True]])  # the poles z = -1 and z = 1 are faces of the orbit space

# The plant.

E3 = np.array([0.0, 0.0, 1.0])
GRAVITY = 9.81
INERTIA = np.array([0.02, 0.025, 0.04])
PRECESSION = np.pi / np.log(2)
CONE_HEIGHT = 0.25
DAMPING = 2.0
PERIOD = 2 * np.pi / PRECESSION


def _policy(policy):
    if policy not in POLICIES:
        raise ValueError("Choose the nominal or the coning gain.")


def gain(z, policy):
    """The gain a(z) and its derivative."""
    _policy(policy)
    if policy == "nominal":
        return np.ones_like(z), np.zeros_like(z)
    return CONE_HEIGHT - z, -np.ones_like(z)


def field(q, policy):
    """The vector field V_a of the thrust axis on {eta = 0}."""
    z = q[..., 2:3]
    a, _ = gain(z, policy)
    return a * (E3 - z * q) + PRECESSION * np.cross(E3, q)


def desired_rate(q, policy):
    return np.cross(q, field(q, policy))


def derivative_rate(q, qdot, policy):
    """Dw_d(q)[qdot]."""
    z, zdot = q[..., 2:3], qdot[..., 2:3]
    a, derivative = gain(z, policy)
    return (derivative * zdot * np.cross(q, E3) + a * np.cross(qdot, E3)
            - PRECESSION * (zdot * q + z * qdot))


def rotation(quaternion):
    """The rotation matrix of a scalar-first unit quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("A quaternion cannot have zero norm.")
    a, b, c, d = np.moveaxis(quaternion / norm, -1, 0)
    entries = (1 - 2 * (c*c + d*d), 2 * (b*c - a*d), 2 * (b*d + a*c),
               2 * (b*c + a*d), 1 - 2 * (b*b + d*d), 2 * (c*d - a*b),
               2 * (b*d - a*c), 2 * (c*d + a*b), 1 - 2 * (b*b + c*c))
    return np.stack(entries, axis=-1).reshape(quaternion.shape[:-1] + (3, 3))


def components(state, policy):
    """R, q, eta, the axis velocity q' and the feedback moment M of plant states."""
    r = rotation(state[..., 6:10])
    q, omega = r[..., :, 2], state[..., 10:13]
    w = np.einsum("...ij,...j->...i", r, omega)
    eta, qdot = w - desired_rate(q, policy), np.cross(w, q)
    derivative = derivative_rate(q, qdot, policy)
    moment = (np.cross(omega, INERTIA * omega)
              + INERTIA * np.einsum("...ji,...j->...i", r, derivative - DAMPING * eta))
    return r, q, eta, qdot, moment


def rhs(state, policy):
    """The closed-loop rigid body with constant thrust mu g, for states (p, v, quaternion, Omega)."""
    state = np.asarray(state, dtype=np.float64)
    _, q, _, _, moment = components(state, policy)
    omega = state[..., 10:13]
    scalar, vector = state[..., 6:7], state[..., 7:10]
    quaternion_dot = .5 * np.concatenate((-np.sum(vector * omega, axis=-1, keepdims=True),
                                          scalar * omega + np.cross(vector, omega)), axis=-1)
    omega_dot = (moment - np.cross(omega, INERTIA * omega)) / INERTIA
    return np.concatenate((state[..., 3:6], GRAVITY * (q - E3), quaternion_dot, omega_dot), axis=-1)


def step(state, duration=.1, policy="coning", dt=.002):
    """Advance plant states by fourth-order Runge-Kutta, normalizing the quaternion at each step."""
    x = np.array(state, dtype=np.float64, copy=True)
    count = max(1, int(np.ceil(duration / dt)))
    h = duration / count
    for _ in range(count):
        k1 = rhs(x, policy)
        k2 = rhs(x + h * k1 / 2, policy)
        k3 = rhs(x + h * k2 / 2, policy)
        k4 = rhs(x + h * k3, policy)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[..., 6:10] /= np.linalg.norm(x[..., 6:10], axis=-1, keepdims=True)
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite closed-loop trajectory.")
    return x


def observe(state):
    """The 18 observation coordinates: p, v, the three columns of R, and Omega."""
    x = np.asarray(state, dtype=np.float64)
    r = rotation(x[..., 6:10])
    return np.concatenate((x[..., :6], r.swapaxes(-1, -2).reshape(x.shape[:-1] + (9,)), x[..., 10:13]), axis=-1)


def _ball(rng, count, dimension, radius):
    directions = rng.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    uniform = rng.random((count, 1))
    radial = np.cbrt(uniform) if dimension == 3 else uniform ** (1 / dimension)
    return radius * radial * directions


def _plant_states(count, seed):
    """Independent positions, velocities, attitudes and body rates."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    position, velocity = _ball(rng, count, 3, 1), _ball(rng, count, 3, 0.5)
    quaternion = rng.normal(size=(count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    omega = _ball(rng, count, 3, 6)
    return np.column_stack((position, velocity, quaternion, omega))


def sample(count, seed, policy, eta_radius=0.):
    """Initial plant states with independent translation and heading and rate error in a ball.

    The heights are uniform on [-1, 1], except that a quarter lie within 10^-2 to 10^-7 of
    a pole and the last two states are the poles. With `eta_radius` 0 the states lie on
    the invariant set {eta = 0}.
    """
    rng = np.random.default_rng(seed)
    state = _plant_states(count, seed)
    z = rng.uniform(-1, 1, count)
    n = count // 4
    z[:n] = rng.choice([-1., 1.], n) * (1 - 10. ** rng.uniform(-7, -2, n))
    z[-2:] = [-1., 1.]
    angles = np.column_stack((rng.uniform(0, 2 * np.pi, count), np.arccos(z), rng.uniform(0, 2 * np.pi, count)))
    rot = Rotation.from_euler("ZYZ", angles)
    state[:, 6:10] = rot.as_quat(scalar_first=True)
    r = rot.as_matrix()
    q = r[:, :, 2]
    eta = _ball(rng, count, 3, eta_radius)
    state[:, 10:13] = np.einsum("...ji,...j->...i", r, desired_rate(q, policy) + eta)
    return state


def group_action(state, matrix):
    """A rotation about e3 acting on plant states; the body rate is fixed.

    Quaternion representatives may change sign under composition: q and -q describe the
    same attitude, and the action on observations is exact.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3))
            or not np.allclose(matrix @ E3, E3) or not np.isclose(np.linalg.det(matrix), 1)):
        raise ValueError("The group consists of rotations about e3.")
    result = np.array(state, dtype=np.float64, copy=True)
    result[..., :3] = result[..., :3] @ matrix.T
    result[..., 3:6] = result[..., 3:6] @ matrix.T
    angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    a, vector = result[..., 6:7].copy(), result[..., 7:10].copy()
    scalar, left = np.cos(angle / 2), np.array([0, 0, np.sin(angle / 2)])
    result[..., 6:7] = scalar * a - np.sum(left * vector, axis=-1, keepdims=True)
    result[..., 7:10] = scalar * vector + a * left + np.cross(left, vector)
    return result


def observation_action(observations, matrix):
    """The action on observations: the matrix acts on the five vectors, the rates are fixed."""
    result = np.array(observations, dtype=np.float64, copy=True)
    result[..., :15] = (result[..., :15].reshape(result.shape[:-1] + (5, 3))
                        @ np.asarray(matrix).T).reshape(result.shape[:-1] + (15,))
    return result


def height_flow(z, duration, policy):
    """The flow of z' = a(z)(1 - z^2), integrated by DOP853 for all heights at once."""
    z = np.asarray(z, dtype=np.float64)
    shape = z.shape
    if duration == 0:
        return z.copy()
    answer = solve_ivp(lambda _, y: gain(y, policy)[0] * (1 - y * y), (0., duration), z.reshape(-1),
                       method="DOP853", rtol=2e-13, atol=2e-14)
    if not answer.success:
        raise RuntimeError(answer.message)
    return answer.y[:, -1].reshape(shape)


def axis_flow(q, duration, policy):
    """The flow of V_a on the sphere of thrust axes."""
    z = height_flow(q[..., 2], duration, policy)
    phi = np.arctan2(q[..., 1], q[..., 0]) + PRECESSION * duration
    radius = np.sqrt(np.maximum(0., 1 - z * z))
    return np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=-1)


def _vertical(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])


def controller_checks():
    """Sampled checks of the closed loop and the Floquet multipliers of the coning orbit.

    For each gain: the identities eta' = -2 eta and q' = V_a(q) + eta x q, the equivariance
    of the axis, eta, q' and the body moment, the integration step, and the decay of eta and
    the final heights from rate errors of radius 0, 0.25, 2 and 6. Then the variational
    equation along C in (z, eta) over one period, whose multipliers are 1,
    exp(-(15/16) T) and exp(-2 T) three times. These are floating-point checks of
    identities and of a calculation that are proved analytically.
    """
    results, trajectories = {}, {}
    matrix = _vertical(.731)
    for policy in POLICIES:
        x = sample(258, 1850, policy, eta_radius=3.)[:-2]
        r, q, eta, qdot, _ = components(x, policy)
        dx = rhs(x, policy)
        wdot = np.einsum("...ij,...j->...i", r, dx[:, 10:13])
        eta_dot = wdot - derivative_rate(q, qdot, policy)
        errors = dict(rate_error_identity=float(np.max(np.abs(eta_dot + DAMPING * eta))),
                      axis_identity=float(np.max(np.abs(qdot - field(q, policy) - np.cross(eta, q)))))
        transformed = group_action(x, matrix)
        _, tq, teta, tqdot, tmoment = components(transformed, policy)
        _, _, _, _, moment = components(x, policy)
        errors.update(axis_action=float(np.max(np.abs(tq - q @ matrix.T))),
                      eta_action=float(np.max(np.abs(teta - eta @ matrix.T))),
                      axis_field_equivariance=float(np.max(np.abs(tqdot - qdot @ matrix.T))),
                      body_moment_equivariance=float(np.max(np.abs(tmoment - moment))))
        assert max(errors.values()) < 1e-11
        initial = sample(130, 1851, policy)[:-2]
        fine, coarse = step(initial, 1., policy, dt=.001), step(initial, 1., policy, dt=.002)
        fine_obs, coarse_obs = observe(fine), observe(coarse)
        exact_axis = axis_flow(observe(initial)[:, 12:15], 1., policy)
        convergence = dict(step_comparison_max=float(np.max(np.abs(fine_obs - coarse_obs))),
                           sphere_reference_max=float(np.max(np.abs(coarse_obs[:, 12:15] - exact_axis))),
                           maximum_rate_error=float(np.max(np.linalg.norm(components(coarse, policy)[2], axis=1))))
        assert convergence["step_comparison_max"] < 2e-7
        assert convergence["sphere_reference_max"] < 2e-7
        rows = []
        for radius in (0., .25, 2., 6.):
            initial = sample(34, 1900, policy, eta_radius=radius)[:-2]
            initial_eta = components(initial, policy)[2]
            x = initial.copy()
            axes, etas, torques = [], [], []
            for index in range(201):
                _, q, eta, _, moment = components(x, policy)
                axes.append(q)
                etas.append(eta)
                torques.append(moment)
                if index < 200:
                    x = step(x, .1, policy)
            axes, etas, torques = np.stack(axes), np.stack(etas), np.stack(torques)
            times = np.arange(201) * .1
            expected = np.exp(-DAMPING * times)[:, None, None] * initial_eta[None]
            eta_error = float(np.max(np.abs(etas - expected)))
            target = 1. if policy == "nominal" else CONE_HEIGHT
            distance = np.abs(axes[-1, :, 2] - target)
            # Starts extremely close to a pole can have long transients.
            rows.append(dict(eta_radius=radius, trajectories=len(initial), maximum_rate_decay_error=eta_error,
                             final_height_error_median=float(np.median(distance)),
                             final_height_error_max=float(distance.max()),
                             maximum_moment_norm=float(np.linalg.norm(torques, axis=-1).max())))
            trajectories[f"{policy}-r{radius}-axis"] = axes
            trajectories[f"{policy}-r{radius}-eta"] = etas
            assert eta_error < 2e-7
        results[policy] = dict(identities=errors, integration=convergence, off_manifold=rows)
    c, period = CONE_HEIGHT, PERIOD
    radius = np.sqrt(1 - c * c)

    def variational(t, values):
        qx, qy = radius * np.cos(PRECESSION * t), radius * np.sin(PRECESSION * t)
        a = np.zeros((5, 5))
        a[0, 0] = -(1 - c * c)
        a[0, 2:5] = [qy, -qx, 0.]
        a[1, 2:5] = [-c * qx / (1 - c * c), -c * qy / (1 - c * c), 1.]
        a[2:, 2:] = -DAMPING * np.eye(3)
        return (a @ values.reshape(5, 5)).ravel()

    answer = solve_ivp(variational, (0., period), np.eye(5).ravel(), method="DOP853", rtol=1e-12, atol=1e-13)
    multipliers = np.sort(np.linalg.eigvals(answer.y[:, -1].reshape(5, 5)).real)
    expected = np.sort([1., np.exp(-(1 - c * c) * period), *([np.exp(-DAMPING * period)] * 3)])
    assert np.max(np.abs(multipliers - expected)) < 1e-11
    results["coning_orbit"] = dict(
        height=c, period=period, angular_rate=PRECESSION, radial_exponent=-(1 - c * c),
        rate_exponents=[-DAMPING] * 3, multipliers=multipliers.tolist(), expected_multipliers=expected.tolist(),
        pole_tangent_expansion_rates=dict(south=1 + c, north=1 - c),
        status="floating-point checks of the analytic identities and of the Floquet calculation")
    trajectories["times"] = np.arange(201) * .1
    return dict(results, trajectories=trajectories)


# Data.

def _array_digest(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(dtype=value.dtype.str, shape=list(value.shape))).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _save_npz(path, arrays):
    """A compressed .npz file whose bytes depend only on the arrays."""
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)


def generate_split(policy, split, settings=DEFAULTS):
    """The trajectories of one split: initial states and observations at spacing 0.1."""
    count, seed = settings["trajectories"][split], settings["data_seeds"][split]
    initial = sample(count, seed, policy)
    batches = []
    for start in range(0, count, 1024):
        state = initial[start:start + 1024].copy()
        observed = [observe(state)]
        for _ in range(settings["intervals"]):
            state = step(state, settings["spacing"], policy, settings["integration_step"])
            observed.append(observe(state))
        batches.append(np.stack(observed, axis=1))
    return dict(observations=np.concatenate(batches), initial_state=initial,
                times=np.arange(settings["intervals"] + 1) * settings["spacing"])


def generate_data(folder, settings=DEFAULTS):
    """Write `<policy>-<split>.npz` for both gains and a `metadata.json` with checksums."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = [f"{policy}-{split}.npz" for policy in settings["policies"] for split in SPLITS]
    if any((folder / name).exists() for name in names + ["metadata.json"]):
        raise FileExistsError(f"{folder} already holds data")
    files = {}
    for policy in settings["policies"]:
        for split in SPLITS:
            arrays = generate_split(policy, split, settings)
            path = folder / f"{policy}-{split}.npz"
            _save_npz(path, arrays)
            files[path.name] = dict(policy=policy, split=split, seed=settings["data_seeds"][split],
                                    trajectories=settings["trajectories"][split],
                                    shape=list(arrays["observations"].shape), bytes=path.stat().st_size,
                                    sha256=training.digest(path),
                                    arrays={name: _array_digest(value) for name, value in arrays.items()})
    metadata = dict(
        generator="morsegraphs.attitude_control.generate_data",
        settings={name: settings[name] for name in ("policies", "trajectories", "data_seeds", "intervals",
                                                    "spacing", "integration_step")},
        initialization="eta = 0 with independent translation and heading; a quarter of the heights near the poles",
        integration="fourth-order Runge-Kutta with quaternion normalization",
        observations="p, v, the columns of R, Omega; the thrust axis is columns 12:15",
        split_rule="independent complete trajectories", files=files)
    training.write(folder / "metadata.json", metadata)
    return metadata


def load_data(folder, policy):
    """The train, validation and test observations of one gain, checked against `metadata.json`."""
    _policy(policy)
    folder = Path(folder)
    metadata = training.read(folder / "metadata.json") if (folder / "metadata.json").exists() else None
    data = dict(policy=policy)
    for split in SPLITS:
        name = f"{policy}-{split}.npz"
        with np.load(folder / name, allow_pickle=False) as saved:
            observations = saved["observations"]
            if metadata is not None and _array_digest(observations) != metadata["files"][name]["arrays"]["observations"]:
                raise ValueError(f"{name} differs from its checksum.")
        data[split] = observations
    return data


def normalization(observations):
    """Per-block scales of the observations and the means of the vertical components and rates."""
    values = observations.reshape(-1, 18).astype(np.float64)
    mean, scale = np.zeros(18), np.ones(18)
    for i in range(5):
        block = slice(3 * i, 3 * i + 3)
        mean[3 * i + 2] = values[:, 3 * i + 2].mean()
        scale[block] = max(float(np.sqrt(np.mean((values[:, block] - mean[block]) ** 2))), 1e-8)
    mean[15:] = values[:, 15:].mean(axis=0)
    scale[15:] = np.maximum(values[:, 15:].std(axis=0), 1e-8)
    return mean.astype(np.float32), scale.astype(np.float32)


# Networks.

class _BlockBilinear(objax.Module):
    """EMLP's bilinear layer, contracted block by block without its dense matrix."""

    def __init__(self, native, rep):
        self.w = native.w
        function = native.weight_proj
        while hasattr(function, "__wrapped__"):
            function = function.__wrapped__
        meta = inspect.getclosurevars(function).nonlocals
        nin, nout = meta["mat_shape"][1], meta["mat_shape"][0]
        assert nin == nout == rep.size()
        groups = list(rep.as_dict(np.arange(rep.size())).items())
        offsets, co, po = {}, 0, 0
        for typ, mult in meta["W_multiplicities"].items():
            dim = typ.size()
            if typ in meta["x_multiplicities"]:
                ns = min(meta["x_multiplicities"][typ], dim)
                offsets[typ] = (co, po, ns)
                po += mult * ns
            co += mult * dim
        self.parts, order = [], []
        for a, oi in groups:
            da, pieces = oi.shape[1], []
            order.extend(oi.ravel().tolist())
            for b, ii in groups:
                db, typ = ii.shape[1], (a * b.T).canonicalize()[0]
                if typ not in offsets:
                    continue
                c0, p0, ns = offsets[typ]
                dim = da * db
                canonical = meta["inv_perm"][oi[:, None, :, None] * nin + ii[None, :, None, :]] - c0
                rr, entries = canonical // dim, canonical % dim
                assert np.all(rr == rr[:, :, 0, 0, None, None])
                assert np.all(entries == np.arange(dim).reshape(1, 1, da, db))
                theta = p0 + rr[:, :, 0, 0, None] * ns + np.arange(ns)
                bids = meta["reduced_indices_dict"][typ]
                pieces.append((jnp.array(ii), jnp.array(theta), jnp.array(bids), ns, da, db))
            self.parts.append((len(oi), da, pieces))
        self.inverse = jnp.array(np.argsort(order))

    def __call__(self, x, training=True):
        blocks = []
        for copies, da, pieces in self.parts:
            y = jnp.zeros((x.shape[0], copies, da), dtype=x.dtype)
            for ii, theta, bids, ns, da, db in pieces:
                selected = x[:, bids].reshape(x.shape[0], ns, da, db)
                products = jnp.einsum("bsoi,bni->bsno", selected, x[:, ii])
                y = y + .1 * jnp.einsum("ans,bsno->bao", self.w.value[theta], products)
            blocks.append(y.reshape(x.shape[0], -1))
        return jnp.concatenate(blocks, axis=-1)[:, self.inverse]


class _ProjectedLinear(objax.Module):
    """EMLP's linear layer applied to weights that are already projected."""

    def __init__(self, native):
        for name, value in native.__dict__.items():
            if isinstance(value, objax.TrainVar):
                setattr(self, name, value)

    def __call__(self, x):
        return x @ self.w.value.T + self.b.value


def _install_blocks(model):
    """Replace the layers of an EMLP by the block layers; return the projections of its weights."""
    variables = list(model.vars().values())
    locations = {id(v): i for i, v in enumerate(variables)}
    projectors = []
    for layer in model.network:
        if hasattr(layer, "bilinear"):
            layer.bilinear = _BlockBilinear(layer.bilinear, gated(layer.nonlinearity.rep))
            linear = layer.linear
            projectors.append((locations[id(linear.w)], locations[id(linear.b)], linear.Pw, linear.Pb))
            layer.linear = _ProjectedLinear(linear)
        else:
            linear = layer
            projectors.append((locations[id(linear.w)], locations[id(linear.b)], linear.Pw, linear.Pb))
    model.network[-1] = _ProjectedLinear(model.network[-1])
    assert [id(v) for v in model.vars().values()] == [id(v) for v in variables]
    return projectors


def _require_float32():
    if jax.config.jax_enable_x64:
        raise RuntimeError("Section 5.3 runs in float32: call morsegraphs.device.configure('float32') "
                           "before importing JAX, or disable jax_enable_x64.")


@lru_cache(maxsize=None)
def _group_and_observation():
    """SO(2) acting on R^3 by rotations about e3, and the observation representation 5V + 3T(0)."""
    return SO2eR3(), 5 * V + 3 * T(0)


class AttitudeModel:
    """The encoder E, the axis decoder, the time-0.1 map f of the latent sphere, psi and the orbit decoder.

    The five networks are EMLPs for SO(2) about e3: E from the observations to V and the
    decoder from V to V, both normalized to the sphere; a scalar network of the height that
    gives a boost b and an angle alpha, bounded by tanh, with which f(q) applies the Lorentz
    boost along e3 to the height and rotates the azimuth by alpha, so f fixes the poles and
    preserves the sphere; and two scalar networks whose bounded positive increments give the
    increasing piecewise linear coordinate psi of [-1, 1] on `knots` knots and the orbit
    decoder. The quotient map of heights is fbar(z) = (z cosh b + sinh b)/(cosh b + z sinh b).
    """

    def __init__(self, policy, seed=0, settings=DEFAULTS, backend="blocks"):
        _require_float32()
        _policy(policy)
        self.policy, self.seed = policy, seed
        self.settings = DEFAULTS | dict(settings)
        np.random.seed(seed)
        objax.random.DEFAULT_GENERATOR.seed(seed)
        s = self.settings
        group, observation = _group_and_observation()
        reps = [(observation, V, s["width"]), (V, V, s["width"]),
                (T(0), 2 * T(0), s["scalar_width"] * T(0)),
                (T(0), T(0), s["scalar_width"] * T(0)),
                (T(0), T(0), s["scalar_width"] * T(0))]
        self.models = [EMLP(a, b, group, ch=width, num_layers=s["layers"]) for a, b, width in reps]
        self.variables = [model.vars() for model in self.models]
        self.params = tuple(tuple(jnp.asarray(v.value, dtype=jnp.float32) for v in values.values())
                            for values in self.variables)
        self.projectors = [_install_blocks(model) for model in self.models] if backend == "blocks" else None
        self.parameters = sum(v.size for group_ in self.params for v in group_)
        self.knots = jnp.linspace(-1., 1., s["knots"], dtype=jnp.float32)
        self.midpoints = (self.knots[:-1] + self.knots[1:]) / 2
        self.mean = self.scale = None

    def prepare(self, params):
        """The weights projected onto the equivariant subspaces, as the block layers use them."""
        if self.projectors is None:
            return params
        result = []
        for values, projections in zip(params, self.projectors, strict=True):
            effective = list(values)
            for iw, ib, pw, pb in projections:
                effective[iw] = (pw @ values[iw].reshape(-1)).reshape(values[iw].shape)
                effective[ib] = pb @ values[ib]
            result.append(tuple(effective))
        return tuple(result)

    def network(self, i, p, x):
        variables = self.variables[i]
        old = variables.tensors()
        try:
            variables.assign(list(p))
            return self.models[i](x)
        finally:
            variables.assign(old)

    def encode(self, p, x):
        raw = self.network(0, p[0], x)
        return raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)

    def decode(self, p, q):
        raw = self.network(1, p[1], q)
        return raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)

    def heads(self, p, z):
        raw = self.network(2, p[2], z.reshape(-1, 1)).reshape(z.shape + (2,))
        return (self.settings["boost_bound"] * jnp.tanh(raw[..., 0]),
                self.settings["angle_bound"] * jnp.tanh(raw[..., 1]))

    def latitude(self, p, z):
        """fbar: the height of f(q) as a function of the height of q."""
        b, _ = self.heads(p, z)
        ch, sh = jnp.cosh(b), jnp.sinh(b)
        return (z * ch + sh) / (ch + z * sh)

    def advance(self, p, q):
        """The learned time-0.1 map f of the latent sphere."""
        z = q[..., 2]
        b, alpha = self.heads(p, z)
        ch, sh = jnp.cosh(b), jnp.sinh(b)
        den = ch + z * sh
        ca, sa = jnp.cos(alpha), jnp.sin(alpha)
        return jnp.stack(((ca * q[..., 0] - sa * q[..., 1]) / den,
                          (sa * q[..., 0] + ca * q[..., 1]) / den, (z * ch + sh) / den), axis=-1)

    def coordinate_values(self, p, decoder=False):
        i = 4 if decoder else 3
        raw = self.network(i, p[i], self.midpoints[:, None])[:, 0]
        increments = jnp.exp(self.settings["coordinate_log_bound"] * jnp.tanh(raw))
        cumulative = jnp.concatenate((jnp.zeros(1, dtype=raw.dtype), jnp.cumsum(increments)))
        return 2 * cumulative / cumulative[-1] - 1

    def coordinate(self, p, z):
        """psi, from heights to the box [-1, 1]."""
        return jnp.interp(z, self.knots, self.coordinate_values(p))

    def inverse(self, p, u):
        return jnp.interp(u, self.coordinate_values(p), self.knots)

    def orbit_decode(self, p, u):
        return jnp.interp(u, self.knots, self.coordinate_values(p, decoder=True))

    def quotient(self, p, u, steps=1):
        """psi o fbar^steps o psi^-1."""
        z = self.inverse(p, u)
        for _ in range(steps):
            z = self.latitude(p, z)
        return self.coordinate(p, z)

    def functions(self):
        """Compiled E, decoder, f, psi, psi^-1, orbit decoder and fbar at the current weights."""
        p = self.prepare(self.params)
        methods = (self.encode, self.decode, self.advance, self.coordinate, self.inverse,
                   self.orbit_decode, self.latitude)
        return tuple(jax.jit(lambda x, method=method: method(p, x)) for method in methods)

    def forecast(self, p, x):
        horizons = self.settings["horizons"]
        q = self.encode(p, x)

        def body(code, _):
            advanced = self.advance(p, code)
            return advanced, advanced

        _, sequence = jax.lax.scan(body, q, None, length=horizons[-1])
        codes = jnp.stack((q, *(sequence[h - 1] for h in horizons[1:])), axis=1)
        decoded = self.decode(p, codes.reshape(-1, 3)).reshape(len(x), len(horizons), 3)
        return codes, decoded


def _pieces(model, params, batch):
    """The seven terms of the objective on windows of normalized observations at the horizons."""
    p = model.prepare(params)
    codes, predicted = model.forecast(p, batch[:, 0])
    target = batch[..., 12:15]
    encoded = model.encode(p, batch.reshape(-1, 18)).reshape(len(batch), len(model.settings["horizons"]), 3)
    rec = jnp.mean((predicted[:, 0] - target[:, 0]) ** 2)
    pred = jnp.mean((predicted[:, 1:] - target[:, 1:]) ** 2)
    calibration = jnp.mean((encoded - target) ** 2)
    closure = jnp.mean((codes[:, 1:] - encoded[:, 1:]) ** 2)
    z = encoded[..., 2]
    h = model.coordinate(p, z)
    orbit_rec = jnp.mean((model.orbit_decode(p, h) - z) ** 2)
    hp = model.coordinate(p, codes[..., 2])
    orbit_pred = jnp.mean((model.orbit_decode(p, hp[:, 1:]) - z[:, 1:]) ** 2)
    quotient_consistency = jnp.mean((hp[:, 1:] - h[:, 1:]) ** 2)
    return jnp.stack((rec, pred, calibration, closure, orbit_rec, orbit_pred, quotient_consistency))


def evaluate(model, params, windows, size=256):
    """The mean objective terms over windows, in batches of `size` padded by repetition."""
    if not hasattr(model, "validation_function"):
        model.validation_function = jax.jit(lambda p, x: _pieces(model, p, x))
    total = np.zeros(7)
    for start in range(0, len(windows), size):
        batch = windows[start:start + size]
        n = len(batch)
        if n < size:
            batch = np.concatenate((batch, np.repeat(batch[-1:], size - n, axis=0)))
        value = np.asarray(model.validation_function(params, jnp.asarray(batch)))
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite validation objective.")
        total += value * n
    total /= len(windows)
    return dict(objective=float(total @ np.asarray(model.settings["loss_weights"], np.float32)),
                parts=total.tolist(), observable_prediction_rms=float(np.sqrt(total[1])),
                encoder_calibration_rms=float(np.sqrt(total[2])), physical_closure_rms=float(np.sqrt(total[3])))


def _windows(normalized, horizons, seed):
    starts = np.random.default_rng(seed).integers(normalized.shape[1] - horizons[-1], size=len(normalized))
    return normalized[np.arange(len(normalized))[:, None], starts[:, None] + np.asarray(horizons)]


def versions():
    names = ("emlp", "objax", "jax", "jaxlib", "optax", "numpy", "scipy", "cmgdb")
    return {name: importlib.metadata.version(name) for name in names}


def train(folder, data, seed, settings=DEFAULTS, progress=None):
    """Train one seed on the data of one gain (`load_data(folder, policy)`).

    The first `warmup_steps` updates calibrate the encoder and the decoders, and the joint
    objective follows. The weights with the lowest validation objective are kept. Writes
    `weights.npz`, `normalization.npz`, `config.json` and `history.json`.
    """
    folder = Path(folder)
    if (folder / "weights.npz").exists():
        raise FileExistsError(folder / "weights.npz")
    folder.mkdir(parents=True, exist_ok=True)  # a progress record may already be there
    settings = DEFAULTS | dict(settings)
    policy = data["policy"]
    model = AttitudeModel(policy, seed, settings)
    observations, validation = data["train"], data["validation"]
    horizons = np.asarray(settings["horizons"])
    mean, scale = normalization(observations)
    # The thrust axis is both an input and the target, so it keeps its scale.
    mean[12:15], scale[12:15] = 0., 1.
    train_data = ((observations - mean) / scale).astype(np.float32)
    valid = ((validation - mean) / scale).astype(np.float32)
    model.mean, model.scale = mean, scale
    np.savez_compressed(folder / "normalization.npz", mean=mean, scale=scale)
    windows = _windows(valid, horizons, settings["validation_window_seed"])
    params = model.params
    initial_params = [np.concatenate([np.asarray(v).ravel() for v in group]) for group in params]
    steps = settings["steps"]
    schedule = optax.cosine_decay_schedule(settings["learning_rate_start"] - settings["learning_rate_end"], steps)
    optimizer = optax.chain(optax.clip_by_global_norm(settings["gradient_clip"]),
                            optax.adamw(lambda t: schedule(t) + settings["learning_rate_end"],
                                        weight_decay=settings["weight_decay"]))
    state = optimizer.init(params)
    joint = jnp.asarray(np.asarray(settings["loss_weights"], np.float32))
    warmup = jnp.asarray(np.asarray(settings["warmup_loss_weights"], np.float32))

    @jax.jit
    def update(p, optimizer_state, batch, weights):
        def objective(values):
            parts = _pieces(model, values, batch)
            return parts @ weights, parts
        (loss, parts), gradient = jax.value_and_grad(objective, has_aux=True)(p)
        changes, optimizer_state = optimizer.update(gradient, optimizer_state, p)
        updated = optax.apply_updates(p, changes)
        finite = jnp.isfinite(loss)
        for value in jax.tree_util.tree_leaves((gradient, updated, optimizer_state)):
            finite = finite & jnp.all(jnp.isfinite(value))
        return updated, optimizer_state, loss, finite

    initial = evaluate(model, params, windows)
    best, best_step = initial["objective"], 0
    history = [dict(step=0, validation=initial)]
    training.save_arrays(folder / "weights.npz", params)
    rng, start_time = np.random.default_rng(settings["batch_seed"] + seed), time.perf_counter()
    span = train_data.shape[1] - int(horizons[-1])
    for step_ in range(1, steps + 1):
        ids = rng.integers(len(train_data), size=settings["batch"])
        times = rng.integers(span, size=settings["batch"])
        batch = train_data[ids[:, None], times[:, None] + horizons]
        weights = warmup if step_ <= settings["warmup_steps"] else joint
        updated, updated_state, loss, finite = update(params, state, jnp.asarray(batch), weights)
        if not bool(finite):
            training.write(folder / "failure.json", dict(step=step_, reason="nonfinite training update"))
            raise FloatingPointError("Nonfinite training update.")
        params, state = updated, updated_state
        if step_ == 1 or step_ % settings["validate_every"] == 0 or step_ == steps:
            score = evaluate(model, params, windows)
            seconds = time.perf_counter() - start_time
            history.append(dict(step=step_, validation=score, training=float(loss), seconds=seconds))
            if score["objective"] < best:
                best, best_step = score["objective"], step_
                training.save_arrays(folder / "weights.npz", params)
            training.write(folder / "history.json", history)
            if progress is not None:
                progress.update(status="training", step=step_, validation=score["objective"], seconds=seconds)
    seconds = time.perf_counter() - start_time
    model.params = training.load_arrays(folder / "weights.npz", params)
    changes = {name: float(np.linalg.norm(np.concatenate([np.asarray(v).ravel() for v in group]) - old))
               for name, group, old in zip(NETWORK_NAMES, model.params, initial_params, strict=True)}
    training.write(folder / "config.json", dict(
        seed=seed, policy=policy, best_step=best_step, best_validation=best, seconds=seconds,
        parameters=model.parameters, parameter_changes=changes, weights_sha256=training.digest(folder / "weights.npz"),
        loss_names=LOSS_NAMES, dtype="float32", settings=settings, versions=versions()))
    if progress is not None:
        progress.update(status="trained", step=steps, validation=best, seconds=seconds)
    return model


def load(folder, policy=None, backend="blocks"):
    """A fit saved by `train`, or by the earlier code, whose flat `config.json` has no gain:
    then `policy` names it."""
    folder = Path(folder)
    config = training.read(folder / "config.json")
    settings = config.get("settings") or {name: config[name] for name in DEFAULTS if name in config}
    policy = config.get("policy", policy)
    if policy is None:
        raise ValueError("The configuration does not name the gain; pass `policy`.")
    model = AttitudeModel(policy, config["seed"], settings, backend)
    model.params = training.load_arrays(folder / "weights.npz", model.params)
    with np.load(folder / "normalization.npz") as saved:
        model.mean, model.scale = saved["mean"], saved["scale"]
    return model


def _batched(function, values, size=2048):
    return np.concatenate([np.asarray(function(jnp.asarray(values[start:start + size], dtype=jnp.float32)))
                           for start in range(0, len(values), size)])


def _normalized(model, observations):
    return ((observations - model.mean) / model.scale).astype(np.float32)


# Morse graphs.

def learned_map(model):
    """The tenth iterate of psi o fbar o psi^-1, evaluated in float32 and returned in binary64."""
    p = model.prepare(model.params)
    advance = jax.jit(lambda u: model.quotient(p, u, steps=model.settings["iterates"]))
    return lambda u: np.asarray(advance(jnp.asarray(u, dtype=jnp.float32)), dtype=np.float64)


def reference_map(policy, model=None, settings=DEFAULTS):
    """The time-one map of z' = a(z)(1 - z^2), in z, or in the coordinate psi of a fit."""
    duration = settings["iterates"] * settings["spacing"]
    if model is None:
        return lambda z: height_flow(z, duration, policy)
    p = model.prepare(model.params)
    values, knots = np.asarray(model.coordinate_values(p)), np.asarray(model.knots)

    def in_coordinate(u):
        z = np.interp(u, values, knots)
        return np.interp(height_flow(z, duration, policy), knots, values)
    return in_coordinate


def image_hulls(function, settings=DEFAULTS):
    """The cells of [-1, 1] and the sampled hulls of their images, cut at the poles."""
    boxes = grids.cells(settings["cells"], [-1.], [1.])
    images = grids.sampled_hulls(lambda points: function(points[:, 0])[:, None], boxes,
                                 settings["samples_per_cell"], settings["padding"], [-1.], [1.])
    return boxes, images


def _check_indices(graph):
    """CMGDB recomputes the index of each Morse set on its own cells and gives its annotation."""
    for row in graph.nodes:
        native = [int(graph.order[i]) for i in row["cell_ids"]]
        if list(CMGDB.ComputeConleyIndexForCells(graph.model, graph.morse, native)) != row["annotations"]:
            raise ValueError(f"The index of Morse set {row['node']} recomputed on its cells differs.")


def prescribed_merge(graph, policy):
    """The Morse graph that the structure of the example prescribes.

    For the nominal gain every Morse set is kept. For the coning gain both poles repel and one
    latitude attracts, so the Morse sets strictly between the poles form one group, merged by
    `grids.merge` with all cells on directed paths between them; CMGDB recomputes its index.
    The structure is checked first: one Morse set at each pole, distinct, not minimal, with
    trivial index; exactly one minimal Morse set between the poles, with index (x-1, 0); any
    other Morse set between the poles with trivial index; no edge leaving the group. A Morse
    set with nontrivial index cannot be absorbed, and a trivial index alone never decides
    the group. Raises ValueError when a check fails.
    """
    _policy(policy)
    _check_indices(graph)
    if policy == "nominal":
        return grids.merge(graph, [])
    count = len(graph.boxes)
    nodes = {row["node"]: row for row in graph.nodes}
    minimal = set(graph.minimal)
    south = [node for node, row in nodes.items() if 0 in row["cell_ids"]]
    north = [node for node, row in nodes.items() if count - 1 in row["cell_ids"]]
    if len(south) != 1 or len(north) != 1 or south == north:
        raise ValueError("The coning structure needs a separate Morse set at each pole.")
    if any(nodes[node]["annotations"] != ["0", "0"] or node in minimal for node in south + north):
        raise ValueError("The Morse sets at the poles do not repel.")
    group = sorted(set(nodes) - set(south + north))
    attracting = [node for node in group if node in minimal]
    if len(attracting) != 1 or nodes[attracting[0]]["annotations"] != ["x-1", "0"]:
        raise ValueError("The coning structure needs one attracting Morse set between the poles with index (x-1, 0).")
    if any(nodes[node]["annotations"] != ["0", "0"] for node in group if node != attracting[0]):
        raise ValueError("A Morse set with nontrivial index between the poles cannot be merged.")
    if any(a in group and b not in group for a, b in graph.edges):
        raise ValueError("An edge leaves the group between the poles.")
    merged = grids.merge(graph, [group])
    row = next(row for row in merged.nodes if row["original_nodes"] == group)
    if row["annotations"] != ["x-1", "0"]:
        raise ValueError("The recomputed index of the merged Morse set is not (x-1, 0).")
    return merged


def reference_graph(policy, model=None, settings=DEFAULTS):
    """The raw Morse graph of the reference time-one map on 2^7 cells, in z, or in the
    coordinate psi of a fit. Its images are sampled hulls of the DOP853 flow."""
    _policy(policy)
    graph = grids.morse_graph(*image_hulls(reference_map(policy, model, settings), settings))
    _check_indices(graph)
    return graph


def learned_graph(model, merge=True):
    """The raw Morse graph of the learned map on 2^7 cells and the graph of `prescribed_merge`.

    With `merge` False the second graph is None; otherwise a failed structure check of the
    merge raises ValueError, and the raw graph is still available with `merge` False.
    """
    raw = grids.morse_graph(*image_hulls(learned_map(model), model.settings))
    _check_indices(raw)
    return raw, (prescribed_merge(raw, model.policy) if merge else None)


# Diagnostics.

def diagnostics(model, data):
    """The held-out errors and the sampled structural checks of a fit, as a dict.

    `one_second_axis_rms` is the componentwise root-mean-square error of the axis predicted
    ten steps ahead on the test trajectories. The entry `arrays` holds profiles for figures.
    """
    if data["policy"] != model.policy:
        raise ValueError("The data belong to the other gain.")
    s = model.settings
    encode, decode, advance, coordinate, inverse, orbit_decode, latitude = model.functions()
    p = model.prepare(model.params)
    grid = np.linspace(-1., 1., 16385, dtype=np.float32)
    test_points = np.stack((np.sqrt(np.maximum(0., 1 - grid * grid)), np.zeros_like(grid), grid), axis=-1)
    image = _batched(advance, test_points)
    coordinates = np.asarray(coordinate(jnp.asarray(grid)))
    values = np.asarray(model.coordinate_values(p))
    knots = np.asarray(model.knots)
    slopes = np.diff(values) / np.diff(knots)
    boosts, angles = jax.jit(lambda z: model.heads(p, z))(jnp.asarray(grid))
    boosts, angles = np.asarray(boosts), np.asarray(angles)
    derivative_b = np.asarray(jax.jit(jax.vmap(jax.grad(lambda z: model.heads(p, z)[0])))(jnp.asarray(grid)))
    denominators = np.cosh(boosts) + grid * np.sinh(boosts)
    derivatives = (1 + (1 - grid * grid) * derivative_b) / denominators ** 2
    roots = []
    scalar_boost = jax.jit(lambda z: model.heads(p, z)[0])
    for a, b, fa, fb in zip(grid[:-1], grid[1:], boosts[:-1], boosts[1:], strict=True):
        if fa * fb < 0:
            roots.append(brentq(lambda z: float(scalar_boost(jnp.asarray(z, dtype=jnp.float32))),
                                float(a), float(b), xtol=1e-8))
    observed = {}
    for split, seed in (("validation", s["validation_window_seed"]), ("test", s["test_window_seed"])):
        physical = data[split]
        normalized = _normalized(model, physical)
        observed[split] = evaluate(model, model.params, _windows(normalized, s["horizons"], seed))
    physical = data["test"]
    normalized = _normalized(model, physical)
    raw_encoder = jax.jit(lambda x: model.network(0, p[0], x))
    raw = _batched(raw_encoder, normalized.reshape(-1, 18))
    minimum_encoder_norm = float(np.linalg.norm(raw, axis=1).min())
    code = _batched(encode, normalized[:, 0])
    next_true = _batched(encode, normalized[:, 1])
    next_pred = _batched(advance, code)
    predicted = code.copy()
    forecasts = [_batched(decode, predicted)]
    closures = [float(np.sqrt(np.mean((code - _batched(encode, normalized[:, 0])) ** 2)))]
    for index in range(1, physical.shape[1]):
        predicted = _batched(advance, predicted)
        forecasts.append(_batched(decode, predicted))
        closures.append(float(np.sqrt(np.mean((predicted - _batched(encode, normalized[:, index])) ** 2))))
    forecasts = np.stack(forecasts, axis=1)
    forecast_rms = np.sqrt(np.mean((forecasts - physical[..., 12:15]) ** 2, axis=(0, 2)))
    matrix = _vertical(.731)
    rotated = observation_action(physical[:256, 0], matrix)
    rotated_code = _batched(encode, _normalized(model, rotated))
    encoded_equivariance = float(np.max(np.abs(rotated_code - code[:256] @ matrix.T)))
    map_equivariance = float(np.max(np.abs(_batched(advance, test_points @ matrix.T) - image @ matrix.T)))
    direct_lower = np.asarray(coordinate(jnp.asarray(next_pred[:, 2])))
    indirect_lower = np.asarray(jax.jit(lambda u: model.quotient(p, u))(coordinate(jnp.asarray(code[:, 2]))))
    support = []
    for split in ("train", "validation"):
        sampled = data[split][:, [0, 1, 4, 10, 32]].reshape(-1, 18)
        support.append(np.asarray(coordinate(jnp.asarray(_batched(encode, _normalized(model, sampled))[:, 2]))))
    cells = s["cells"]
    counts = np.bincount(np.clip(np.floor((np.concatenate(support) + 1) * (cells // 2)).astype(int), 0, cells - 1),
                         minlength=cells)
    endpoint_images = np.asarray(latitude(jnp.array([-1., 1.])))
    raw_decoder = jax.jit(lambda x: model.network(1, p[1], x))
    minimum_decoder_norm = float(np.linalg.norm(_batched(raw_decoder, test_points), axis=1).min())
    results = dict(
        **observed, one_step_physical_closure_rms=float(np.sqrt(np.mean((next_pred - next_true) ** 2))),
        one_second_axis_rms=float(forecast_rms[s["iterates"]]),
        forecast_rms_by_horizon=forecast_rms.tolist(), closure_rms_by_horizon=closures,
        sampled_encoder_equivariance_error=encoded_equivariance, sampled_map_equivariance_error=map_equivariance,
        maximum_sphere_norm_error=float(np.max(np.abs(np.linalg.norm(image, axis=1) - 1))),
        lower_square_maximum_error=float(np.max(np.abs(direct_lower - indirect_lower))),
        endpoint_images=endpoint_images.tolist(), minimum_latitude_derivative=float(derivatives.min()),
        minimum_coordinate_slope=float(slopes.min()), maximum_coordinate_slope=float(slopes.max()),
        minimum_test_raw_encoder_norm=minimum_encoder_norm, minimum_sampled_raw_decoder_norm=minimum_decoder_norm,
        inverse_identity_maximum=float(np.max(np.abs(np.asarray(inverse(coordinate(jnp.asarray(grid)))) - grid))),
        interior_fixed_latitudes=roots, supported_interval_cells=int(np.sum(counts > 0)),
        pole_boosts=[float(boosts[0]), float(boosts[-1])],
        orbit_decoder_rms=float(np.sqrt(np.mean((np.asarray(orbit_decode(coordinate(jnp.asarray(grid)))) - grid) ** 2))))
    assert results["endpoint_images"] == [-1., 1.]
    assert results["maximum_sphere_norm_error"] < 2e-6
    assert results["lower_square_maximum_error"] < 2e-6
    assert results["minimum_coordinate_slope"] > 0
    assert minimum_encoder_norm > 1e-9 and minimum_decoder_norm > 1e-9
    # The fixed latitudes of fbar between the poles, decoded to the plant.
    geometry = []
    for z in roots:
        point = jnp.array([[np.sqrt(1 - z * z), 0., z]], dtype=jnp.float32)
        decoded = np.asarray(model.decode(p, point))[0]
        _, angle = model.heads(p, jnp.asarray(z, dtype=jnp.float32))
        multiplier = float(jax.grad(lambda height: model.latitude(p, height))(jnp.asarray(z, dtype=jnp.float32)))
        geometry.append(dict(latent_height=z, decoded_physical_height=float(decoded[2]),
                             angle_per_step=float(angle), transverse_multiplier=multiplier))
    results["learned_orbit_geometry"] = geometry
    # The block layers against EMLP's own layers on 128 test observations.
    native = AttitudeModel(model.policy, model.seed, model.settings, backend="native")
    native.params = model.params
    x = normalized[:128, 0]
    inputs = [x, x[:, 12:15], x[:, 12:15], x[:, 14], x[:, 14], x[:, 14], x[:, 14]]
    errors = [float(np.max(np.abs(np.asarray(f(jnp.asarray(value))) - np.asarray(g(jnp.asarray(value))))))
              for f, g, value in zip(model.functions(), native.functions(), inputs, strict=True)]
    assert max(errors) < 3e-5
    results["official_emlp_output_maximum_errors"] = errors
    # The image hulls of the graph against 4097 samples per cell.
    function = learned_map(model)
    boxes, hulls = image_hulls(function, s)
    violation = 0.
    for start in range(0, len(boxes), 16):
        chosen = boxes[start:start + 16]
        points = chosen[:, :1] + (chosen[:, 1:] - chosen[:, :1]) * np.linspace(0, 1, 4097)
        dense = function(points.reshape(-1)).reshape(points.shape)
        violation = max(violation, float(np.max(hulls[start:start + 16, 0] - dense.min(axis=1))),
                        float(np.max(dense.max(axis=1) - hulls[start:start + 16, 1])))
    assert violation <= 0
    results.update(dense_image_samples=len(boxes) * 4097, dense_image_violation=violation,
                   status="held-out samples and floating-point checks; no certificate for the plant")
    results["arrays"] = dict(heights=grid, boost=boosts, angle=angles, latitude=image[:, 2], coordinates=coordinates,
                             knots=knots, coordinate_values=values, physical_support_counts=counts,
                             test_physical_axis=physical[:, 0, 12:15], test_latent_axis=code,
                             forecast_rms=forecast_rms, closure_rms=np.asarray(closures),
                             test_forecasts=forecasts[:32], test_targets=physical[:32, :, 12:15])
    return results


# Attracting regions.

def region_samples(policy, data=None, settings=DEFAULTS):
    """Pairs of observations one time unit apart on {eta = 0}.

    All windows of the test trajectories (regenerated from their seed when `data` is None)
    and `fresh_trajectories` new one-second trajectories from `fresh_seed`.
    """
    observations = data["test"] if data is not None else generate_split(policy, "test", settings)["observations"]
    lag = settings["iterates"]
    initial = sample(settings["fresh_trajectories"], settings["fresh_seed"], policy, eta_radius=0.)
    evolved = step(initial, duration=settings["iterates"] * settings["spacing"], policy=policy,
                   dt=settings["integration_step"])
    starts = observations.shape[1] - lag
    source = np.concatenate((observations[:, :starts].reshape(-1, 18), observe(initial)))
    image = np.concatenate((observations[:, lag:].reshape(-1, 18), observe(evolved)))
    return source, image


def attracting_regions(model, graph, data=None, metric="chordal", samples=None):
    """The rows of `regions.compare` for each minimal Morse set of `graph`.

    `graph` is the merged graph of `learned_graph`. Region membership is decided by the
    code psi(E(x)) in the grid. With the chordal metric of the paper, the residual is the
    chordal distance between the heights of fbar^10(E(x)) and E(F(x)) on the latent sphere,
    and the tolerance is measured after undoing psi; with "euclidean" both are measured in
    the coordinate psi. `samples` may give the pairs of `region_samples` directly. Returns
    the rows and the residual at each sample.
    """
    s = model.settings
    source, image = region_samples(model.policy, data, s) if samples is None else samples
    labels = regions.attracting_regions(graph, FACES)
    encode, _, _, coordinate, _, _, latitude = model.functions()
    p = model.prepare(model.params)
    quotient = jax.jit(lambda u: model.quotient(p, u, steps=s["iterates"]))

    def iterate(z):
        for _ in range(s["iterates"]):
            z = latitude(z)
        return z
    height = jax.jit(iterate)
    x, y = _normalized(model, source), _normalized(model, image)
    initial, future = _batched(encode, x, 8192), _batched(encode, y, 8192)
    error = max(float(np.max(np.abs(np.linalg.norm(initial, axis=1) - 1))),
                float(np.max(np.abs(np.linalg.norm(future, axis=1) - 1))))
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError("The encoded samples are not on the latent sphere.")
    codes = _batched(coordinate, initial[:, 2], 8192)
    if metric == "chordal":
        values, knots = np.asarray(model.coordinate_values(p)), np.asarray(model.knots)
        predicted = _batched(height, initial[:, 2], 8192)
        return regions.compare(graph, labels, codes[:, None], predicted[:, None], future[:, 2:3],
                               regions.SphereHeight(), FACES, metric_boxes=np.interp(graph.boxes, values, knots),
                               metric_images=np.interp(graph.images, values, knots))
    if metric == "euclidean":
        predicted = _batched(quotient, codes, 8192)
        encoded_image = _batched(coordinate, future[:, 2], 8192)
        return regions.compare(graph, labels, codes[:, None], predicted[:, None], encoded_image[:, None],
                               regions.Euclidean(), FACES)
    raise ValueError("Use the chordal or the euclidean metric.")


# Names, colors and indices in X.

EXAMPLE = "ex3_attitude_control"
DISPLAY = {"-e3": "−e₃", "e3": "e₃"}
MATH = {"-e3": "$-e_3$", "e3": "$e_3$"}
# Colors follow the dynamical role of a Morse set through the keys of `plotting.ROLE`: the attractor
# (e3 for the nominal gain, the latitude C for the coning gain) is green like an attracting latitude,
# the repelling poles are vermilion (-e3) and orange (e3 for the coning gain), and any other Morse set
# between the poles is sky blue.
COLOR_KEYS = {"nominal": {"-e3": "-e3", "e3": "e3", "C": "A+", "A": "A"},
              "coning": {"-e3": "-e3", "e3": "N", "C": "A+", "A": "A"}}
NEUTRAL = plotting.mix(plotting.GRAY, .1)
# The indices in X = S^2 proved on explicit index pairs in the supplement (attitude index pairs), as
# CMGDB shift classes in degrees 0, 1, 2: H_2 = F at a repelling pole, H_0 = F at the nominal sink e3
# and H_0 = H_1 = F at the coning latitude C, with identity index maps.
PROVED = {"nominal": {"-e3": ["0", "0", "x-1"], "e3": ["x-1", "0", "0"]},
          "coning": {"-e3": ["0", "0", "x-1"], "e3": ["0", "0", "x-1"], "C": ["x-1", "x-1", "0"]}}
FIXED = [(0, "lower"), (0, "upper")]  # P/G = {-1, 1}: the poles are the fixed points of the action


def node_names(graph):
    """The name of each Morse set of a graph on [-1, 1]: -e3 and e3 for the Morse sets that contain
    the cells at the poles, C for a minimal Morse set between them, and A for any other Morse set
    (A1, A2, ... from the bottom when there are several)."""
    last, names, others = len(graph.boxes) - 1, {}, []
    for row in sorted(graph.nodes, key=lambda row: row["cell_ids"][0]):
        if 0 in row["cell_ids"]:
            names[row["node"]] = "-e3"
        elif last in row["cell_ids"]:
            names[row["node"]] = "e3"
        elif row["node"] in graph.minimal:
            names[row["node"]] = "C"
        else:
            others.append(row["node"])
    for number, node in enumerate(others, 1):
        names[node] = "A" if len(others) == 1 else f"A{number}"
    if len(set(names.values())) != len(names):
        raise ValueError("Two Morse sets of the graph get the same name.")
    return names


def color_key(policy, name):
    """The key of `plotting.ROLE` that colors the Morse set of the given name."""
    return COLOR_KEYS[policy].get("A" if name.startswith("A") else name)


def describe(graph):
    """The Morse sets of a graph by name: the order as pairs of names, and the index and cells of each."""
    names = node_names(graph)
    return dict(order=sorted([names[a], names[b]] for a, b in graph.edges),
                indices={names[row["node"]]: list(row["annotations"]) for row in graph.nodes},
                cells={names[row["node"]]: list(row["cell_ids"]) for row in graph.nodes})


def compare_graphs(learned, reference):
    """Whether two Morse graphs on [-1, 1] have the same order and the same indices by name, and
    whether their Morse sets have the same cells, which is meaningful in one coordinate only."""
    a, b = describe(learned), describe(reference)
    return dict(order=a["order"] == b["order"] and set(a["indices"]) == set(b["indices"]),
                indices=a["indices"] == b["indices"], cells=a["cells"] == b["cells"])


def indices_in_X(graph):
    """The Conley index in X = S^2 of the lifted index pair of each Morse set of a graph on [-1, 1].

    Theorem thm:sphere-reconstruction with m = 1 and delta = id (appendix, reduced attitude
    control), evaluated by `homology.reconstructed_index` on the outer approximation of the graph
    with the poles as P/G. Each row gives CMGDB's annotation and the index on [-1, 1] computed by
    `homology`, the index relative to K/G, the index in X, and the sizes of N/G, L/G and N_P/G,
    as CMGDB shift classes over F_5 in degrees 0, 1, ...
    """
    grid = homology.Grid([len(graph.boxes)])
    relation = homology.BoxMap.from_relation(grid, graph.offsets, graph.targets)
    names, rows = node_names(graph), []
    for row in graph.nodes:
        result = homology.reconstructed_index(relation, row["cell_ids"], FIXED, 1)
        rows.append(dict(node=row["node"], name=names[row["node"]], cmgdb=list(row["annotations"]),
                         quotient=result.quotient.shift_class(), relative=result.relative.shift_class(),
                         in_X=result.lifted.shift_class(), betti_in_X=list(result.lifted.betti),
                         N_cells=len(result.N), L_cells=len(result.L), N_P_cells=len(result.N_P)))
    return rows


def _plain(value):
    """Numpy values as plain Python values, for JSON."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def graph_record(graph):
    """A MorseGraph as JSON: its record, the names of its Morse sets, its image hulls and the Morse set of each cell."""
    record = graph.record()
    record.update(names=node_names(graph), images=graph.images, membership=graph.membership)
    return _plain(record)


def parameters_digest(model):
    """SHA-256 of the weights of a fit, independent of the file that holds them."""
    digest = hashlib.sha256()
    for group in model.params:
        for value in group:
            digest.update(_array_digest(np.asarray(value)).encode())
    return digest.hexdigest()


def fixed_heights(settings=DEFAULTS):
    """The fixed heights of the reference time-one map h_1 of each gain and the derivative of h_1 there.

    The zeros z* of g(z) = a(z)(1 - z^2) are fixed by every h_t, and h_1'(z*) = exp(g'(z*)). The
    rows give this value and a second-order difference quotient of the DOP853 flow, one-sided at the poles.
    """
    duration, step, rows = settings["iterates"] * settings["spacing"], 1e-5, []
    for policy in POLICIES:
        for z, name in ([(-1., "-e3"), (1., "e3")] if policy == "nominal" else
                        [(-1., "-e3"), (CONE_HEIGHT, "C"), (1., "e3")]):
            a, da = gain(np.array(z), policy)
            exact = float(np.exp(da * (1 - z * z) - 2 * z * a))
            if abs(z) < 1:
                values = height_flow(np.array([z - step, z + step]), duration, policy)
                quotient = (values[1] - values[0]) / (2 * step)
            else:
                side = -np.sign(z)
                values = height_flow(z + side * step * np.arange(3.), duration, policy)
                quotient = side * (-3 * values[0] + 4 * values[1] - values[2]) / (2 * step)
            rows.append(dict(policy=policy, name=name, height=z, image=float(height_flow(np.array([z]), duration, policy)[0]),
                             derivative=exact, difference_quotient=float(quotient)))
    return rows


def system_record():
    """`controller_checks` without its trajectories, as JSON."""
    checks = controller_checks()
    checks.pop("trajectories")
    return _plain(checks)


class Analysis:
    """Every result of Section 5.3 for one fit, computed when first used and then kept.

    The learned Morse graph on 2^7 cells, raw and merged as the example prescribes; the reference
    graphs in z and in the coordinate psi of the fit, raw and merged; the attracting regions of
    the minimal Morse sets of the merged and of the raw graph, with tolerances and sampled
    residuals; the indices in X; and the held-out diagnostics. `record()` collects them as JSON.
    """

    def __init__(self, model, data, name=None):
        if data["policy"] != model.policy:
            raise ValueError("The data belong to the other gain.")
        self.model, self.data, self.policy = model, data, model.policy
        self.name = name or f"{model.policy}-seed{model.seed}"

    @cached_property
    def learned(self):
        return learned_graph(self.model)

    @property
    def raw(self):
        return self.learned[0]

    @property
    def merged(self):
        return self.learned[1]

    @cached_property
    def reference_z(self):
        raw = reference_graph(self.policy, settings=self.model.settings)
        return raw, prescribed_merge(raw, self.policy)

    @cached_property
    def reference_psi(self):
        raw = reference_graph(self.policy, self.model, self.model.settings)
        return raw, prescribed_merge(raw, self.policy)

    def graphs(self):
        return dict(learned_raw=self.raw, learned_merged=self.merged,
                    reference_z=self.reference_z[0], reference_z_merged=self.reference_z[1],
                    reference_psi=self.reference_psi[0], reference_psi_merged=self.reference_psi[1])

    @cached_property
    def coordinate(self):
        """The knots and values of psi: psi(knots) = values."""
        p = self.model.prepare(self.model.params)
        return (np.asarray(self.model.knots, dtype=np.float64),
                np.asarray(self.model.coordinate_values(p), dtype=np.float64))

    def latent_heights(self, graph):
        """The boundaries of the cells of a graph in u, as heights of the latent sphere: psi^-1(u)."""
        knots, values = self.coordinate
        return np.interp(np.r_[graph.boxes[:, 0], graph.boxes[-1, 1]], values, knots)

    @cached_property
    def samples(self):
        return region_samples(self.policy, self.data, self.model.settings)

    def _regions(self, graph):
        rows, _ = attracting_regions(self.model, graph, self.data, samples=self.samples)
        return dict(rows=rows, labels=regions.attracting_regions(graph, FACES))

    @cached_property
    def regions(self):
        """The attracting region of the minimal Morse set of the merged graph."""
        return self._regions(self.merged)

    @cached_property
    def raw_regions(self):
        """The attracting regions of the minimal Morse sets of the raw graph."""
        return self._regions(self.raw)

    @cached_property
    def diagnostics(self):
        result = diagnostics(self.model, self.data)
        result.pop("arrays")
        return result

    @cached_property
    def indices(self):
        return dict(reference_z=indices_in_X(self.reference_z[1]), reference_psi=indices_in_X(self.reference_psi[1]),
                    learned=indices_in_X(self.merged), learned_raw=indices_in_X(self.raw))

    @cached_property
    def comparison(self):
        """The learned graphs against the reference graphs of the same kind."""
        return dict(reference_z=compare_graphs(self.merged, self.reference_z[1]),
                    reference_psi=compare_graphs(self.merged, self.reference_psi[1]),
                    reference_psi_raw=compare_graphs(self.raw, self.reference_psi[0]))

    def record(self):
        knots, values = self.coordinate
        return _plain(dict(
            fit=self.name, policy=self.policy, parameters_sha256=parameters_digest(self.model),
            coordinate=dict(knots=knots, values=values), diagnostics=self.diagnostics,
            graphs={name: graph_record(graph) for name, graph in self.graphs().items()},
            comparison=self.comparison,
            regions=dict(samples=len(self.samples[0]), merged=self.regions, raw=self.raw_regions),
            indices=self.indices))


def example_record(system, analyses):
    """The record of Section 5.3: the closed-loop checks and the record of the selected fit of each gain."""
    return _plain(dict(example=EXAMPLE, versions=versions(), system=system,
                       **{policy: analyses[policy].record() for policy in POLICIES}))


def selected_analyses(folder):
    """The Analysis of the selected fit of each gain, with the data in `folder`."""
    folder = Path(folder)
    selection = training.read(folder / "selection.json")
    result = {}
    for policy in POLICIES:
        run = selection[policy]["selected"]
        result[policy] = Analysis(load(folder / "models" / run), load_data(folder, policy), run)
    return result


def reference_record(folder):
    """The record that the notebook checks in replay, for the selected fits in `folder`."""
    return example_record(system_record(), selected_analyses(folder))


def differences(computed, saved, rel=1e-9, skip=("versions",)):
    """Compare two records: numbers to relative tolerance `rel`, everything else exactly.

    Returns the count of compared numbers and the paths where the records differ. Top-level
    entries named in `skip` are not compared.
    """
    computed = json.loads(json.dumps(_plain(computed)))
    count, found = 0, []

    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def walk(a, b, path):
        nonlocal count
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                found.append(f"{path}: entries {sorted(set(a) ^ set(b))} in one record only")
            for key in sorted(set(a) & set(b)):
                if not (path == "" and key in skip):
                    walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                found.append(f"{path}: {len(a)} entries against {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}/{i}")
        elif number(a) and number(b):
            count += 1
            if not math.isclose(a, b, rel_tol=rel, abs_tol=0.):
                found.append(f"{path}: {a!r} against {b!r}")
        elif type(a) is not type(b) or a != b:
            found.append(f"{path}: {a!r} against {b!r}")

    walk(computed, saved, "")
    return count, found


def timings(model, repeats=3):
    """Times of the learned graph on this machine, as in the supplement: the sampled image hulls of
    the 2^7 cells after a first call that compiles the map, and the native CMGDB computation of the
    Morse graph and its Conley indices from these hulls, first and warm (median of `repeats`
    further calls on the same CMGDB model). Times depend on the machine and are not in the record."""
    s = model.settings
    function = learned_map(model)
    boxes, images = image_hulls(function, s)
    hulls = []
    for _ in range(repeats):
        start = time.perf_counter()
        image_hulls(function, s)
        hulls.append(time.perf_counter() - start)
    lookup = {tuple(box): image.tolist() for box, image in zip(boxes, images, strict=True)}
    depth = len(boxes).bit_length() - 1
    native = CMGDB.Model(depth, depth, depth, 10000, [-1.], [1.], lambda box: lookup[tuple(box)])
    native.set_batch_map(lambda batch: [lookup[tuple(box)] for box in batch])
    seconds = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        CMGDB.ComputeConleyMorseGraph(native, **grids.NATIVE_OPTIONS)
        seconds.append(time.perf_counter() - start)
    return dict(hulls_seconds=float(np.median(hulls)), native_first_seconds=seconds[0],
                native_warm_seconds=float(np.median(seconds[1:])), repeats=repeats)


# Figures. Each panel is bare, saved as PDF and SVG by `plotting`, under the manuscript's name.

GRAPH_SCALE = .55               # the figures of Section 5.3 include the graphs at scale .55
SPHERE_MAIN = (1.45, 1.6)       # printed at a height of 1.1 in
SPHERE_ROW = (2.2, 2.4)         # printed at .35 of the text width
SPHERE_WITH_STRIP = (2.4, 3.1)  # printed at a height of 2.9 in
STRIP = (3.0, .8)               # printed at .485 of the text width
BAND = .05                      # the tallest band of the sphere drawn as one piece


@contextmanager
def _quiet_fonts():
    """Silence fontTools while Matplotlib subsets fonts for a PDF: it logs a warning for the old
    timestamps of the fonts, which does not affect the panel."""
    logger = logging.getLogger("fontTools")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


def graph_panel(graph, policy, annotations, path):
    """A Morse graph labeled `name: (index)`, where `annotations` maps each Morse set to its index."""
    names, nodes = node_names(graph), {}
    for row in graph.nodes:
        name = names[row["node"]]
        nodes[row["node"]] = (f"{DISPLAY.get(name, name)}: ({', '.join(annotations[row['node']])})",
                              color_key(policy, name))
    return plotting.graph(nodes, graph.edges, path, GRAPH_SCALE)


def cell_colors(graph, policy, labels=None):
    """The color of each cell: the ink of its Morse set, else the tint of the minimal Morse set whose
    attracting region contains it (`labels` of `regions.attracting_regions`), else neutral."""
    names, colors = node_names(graph), []
    for cell, node in enumerate(graph.membership):
        if node >= 0:
            colors.append(plotting.color(color_key(policy, names[node])))
        elif labels is not None and labels[cell] >= 0:
            colors.append(plotting.tint(color_key(policy, names[labels[cell]])))
        else:
            colors.append(NEUTRAL)
    return colors


def bands(heights, colors):
    """Bands (z_lower, z_upper, color) of the sphere from the heights of the cell boundaries and the
    colors of the cells: adjacent cells of one color form a band, cut into pieces no taller than BAND."""
    result, start = [], 0
    for cell in range(1, len(colors) + 1):
        if cell == len(colors) or colors[cell] != colors[start]:
            edges = np.linspace(heights[start], heights[cell],
                                max(1, int(np.ceil((heights[cell] - heights[start]) / BAND))) + 1)
            result += [(float(a), float(b), colors[start]) for a, b in zip(edges[:-1], edges[1:])]
            start = cell
    return result


def sphere_panel(band_list, path, size):
    """The sphere of `plotting.sphere_bands` with the given bands; the bands are rasterized at 600 dpi."""
    plt = plotting.style()
    fig, ax = plt.subplots(figsize=size, dpi=600)
    fig.subplots_adjust(0, 0, 1, 1)
    plotting.sphere_bands(ax, band_list)
    for collection in ax.collections:
        collection.set_rasterized(True)
    with _quiet_fonts():
        path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def strip(ax, graph, policy, xlabel, label=False):
    """The Morse sets of a graph on [-1, 1] as cells on a line, named above when `label` is True."""
    names = node_names(graph)
    roles = {node: color_key(policy, name) for node, name in names.items()}
    plotting.interval_strip(ax, graph.boxes, roles, xlabel, graph.membership)
    ax.set_xticks([-1, 0, 1])
    if label:
        for row in graph.nodes:
            name = names[row["node"]]
            ax.text(sum(row["hull"]) / 2, .16, MATH.get(name, f"${name}$"), ha="center", va="bottom",
                    color=plotting.color(roles[row["node"]]))


def strip_panel(graph, policy, xlabel, path):
    plt = plotting.style()
    fig, ax = plt.subplots(figsize=STRIP)
    fig.subplots_adjust(left=.03, right=.97, bottom=.5, top=.95)
    strip(ax, graph, policy, xlabel)
    with _quiet_fonts():
        path = plotting.save_panel(fig, path)
    plt.close(fig)
    return path


def figure_attitude(analysis, folder):
    """fig:attitude-nominal (nominal gain) or fig:attitude-computation (coning gain).

    (a) `attitude/<gain>/learned-quotient`: the merged learned Morse graph with CMGDB's indices on
    [-1, 1]. (b) `learned-sphere`: the pullback of the attracting block to the latent sphere Z, as
    preimages under psi o pi_Z of the cells: the Morse sets in ink and the attracting region of the
    minimal Morse set of the merged graph in its tint. (c) `learned-reconstructed`: the same graph
    with the reconstructed indices in S^2. Returns the three PDF paths.
    """
    policy, graph = analysis.policy, analysis.merged
    target = Path(folder) / "attitude" / policy
    annotations = {row["node"]: row["annotations"] for row in graph.nodes}
    in_X = {row["node"]: row["in_X"] for row in analysis.indices["learned"]}
    colors = cell_colors(graph, policy, analysis.regions["labels"])
    return [graph_panel(graph, policy, annotations, target / "learned-quotient"),
            sphere_panel(bands(analysis.latent_heights(graph), colors), target / "learned-sphere", SPHERE_MAIN),
            graph_panel(graph, policy, in_X, target / "learned-reconstructed")]


def figure_raw_region(analysis, folder):
    """Panels outside the manuscript: the raw learned Morse graph of a gain with CMGDB's indices
    (`attitude/<gain>/learned-raw-quotient`) and, on the latent sphere, its Morse sets with the
    attracting region of its minimal Morse set (`learned-raw-sphere`). For the coning gain this is
    the region of the minimal Morse set C alone, without the merge. Returns the two PDF paths."""
    policy, graph = analysis.policy, analysis.raw
    target = Path(folder) / "attitude" / policy
    colors = cell_colors(graph, policy, analysis.raw_regions["labels"])
    return [graph_panel(graph, policy, {row["node"]: row["annotations"] for row in graph.nodes},
                        target / "learned-raw-quotient"),
            sphere_panel(bands(analysis.latent_heights(graph), colors), target / "learned-raw-sphere", SPHERE_MAIN)]


def figure_appendix_spheres(analysis, folder):
    """One panel of fig:appendix-notebook-attitude, `notebook-preimages/attitude-<gain>`: the Morse
    sets of the merged learned graph on the latent sphere Z (preimages under psi o pi_Z), above
    their cells in the coordinate u, named as in the graphs. Returns the PDF path."""
    policy, graph = analysis.policy, analysis.merged
    plt = plotting.style()
    fig = plt.figure(figsize=SPHERE_WITH_STRIP, dpi=600)
    top = fig.add_axes([0., .25, 1., .75])
    plotting.sphere_bands(top, bands(analysis.latent_heights(graph), cell_colors(graph, policy)))
    for collection in top.collections:
        collection.set_rasterized(True)
    strip(fig.add_axes([.06, .12, .88, .09]), graph, policy, "$u$", label=True)
    with _quiet_fonts():
        path = plotting.save_panel(fig, Path(folder) / "notebook-preimages" / f"attitude-{policy}")
    plt.close(fig)
    return path


def figure_appendix_comparison(analysis, folder):
    """fig:appendix-attitude-comparison, coning gain: the reference in the coordinate z (left) and
    the merged learned graph in u (right). Rows: the Morse sets on the sphere (X for the reference,
    the latent sphere Z for the fit), their cells on [-1, 1], the Morse graphs with CMGDB's indices,
    and the Morse graphs with the reconstructed indices in S^2. Returns the eight PDF paths in
    reading order."""
    if analysis.policy != "coning":
        raise ValueError("The comparison figure is for the coning gain.")
    reference, learned = analysis.reference_z[1], analysis.merged
    target, row1 = Path(folder) / "attitude" / "coning", Path(folder) / "notebook-preimages"
    z = np.r_[reference.boxes[:, 0], reference.boxes[-1, 1]]
    in_X = {name: {row["node"]: row["in_X"] for row in analysis.indices[name]} for name in ("reference_z", "learned")}
    return [sphere_panel(bands(z, cell_colors(reference, "coning")), row1 / "attitude-coning-original-sphere", SPHERE_ROW),
            sphere_panel(bands(analysis.latent_heights(learned), cell_colors(learned, "coning")),
                         row1 / "attitude-coning-sphere", SPHERE_ROW),
            strip_panel(reference, "coning", "$z$", target / "original-interval"),
            strip_panel(learned, "coning", "$u$", target / "learned-interval"),
            graph_panel(reference, "coning", {row["node"]: row["annotations"] for row in reference.nodes},
                        target / "original-quotient"),
            graph_panel(learned, "coning", {row["node"]: row["annotations"] for row in learned.nodes},
                        target / "learned-quotient"),
            graph_panel(reference, "coning", in_X["reference_z"], target / "original-reconstructed"),
            graph_panel(learned, "coning", in_X["learned"], target / "learned-reconstructed")]


# The notebook ex3_attitude_control.ipynb.

class Table:
    """A table shown as HTML in a notebook and as aligned text elsewhere."""

    def __init__(self, columns, rows, note=None):
        self.columns, self.note = [str(c) for c in columns], note
        self.rows = [[_text(value) for value in row] for row in rows]

    def _repr_html_(self):
        cell = 'style="text-align:left"'
        head = "".join(f"<th {cell}>{escape(c)}</th>" for c in self.columns)
        body = "".join("<tr>" + "".join(f"<td {cell}>{escape(v)}</td>" for v in row) + "</tr>" for row in self.rows)
        note = f"<p>{escape(self.note)}</p>" if self.note else ""
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}"

    def __repr__(self):
        lines = [self.columns, *self.rows]
        widths = [max(len(line[i]) for line in lines) for i in range(len(self.columns))]
        text = ["  ".join(v.ljust(w) for v, w in zip(line, widths)).rstrip() for line in lines]
        text.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(text + ([self.note] if self.note else []))


class Line(str):
    """A one-line result, shown without quotes."""

    def __repr__(self):
        return str(self)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    return str(value)


def _index(annotations):
    return f"({', '.join(annotations)})"


def _interval(row):
    lo, hi = row["hull"]
    return f"[{lo:.6g}, {hi:.6g}]"


def _name(name):
    return DISPLAY.get(name, name)


def _order(graph):
    """The edges of a Morse graph by name, such as `−e₃ → C, e₃ → C`."""
    names = node_names(graph)
    return ", ".join(f"{_name(names[a])} → {_name(names[b])}" for a, b in sorted(graph.edges))


def _sets(graph):
    """The Morse sets of a graph from the bottom, with their indices."""
    names = node_names(graph)
    return ", ".join(f"{_name(names[row['node']])} {_index(row['annotations'])}"
                     for row in sorted(graph.nodes, key=lambda row: row["cell_ids"][0]))


class Notebook:
    """The steps of `ex3_attitude_control.ipynb`, one method per code cell.

    `mode` is "replay" (load the selected fit of each gain from `data/ex3_attitude_control/models`
    and check every result against `data/ex3_attitude_control/reference/results.json`) or "train"
    (retrain the selected seed of each gain with `config.json` on the CPU, then run the same steps).
    `out` is a new folder for the results and figures, by default under `outputs/notebooks/`;
    `root` is the root of the repository, by default the working directory of the notebook. The
    methods return a table or a one-line result, or save the panels of a figure and display them.
    """

    def __init__(self, mode="replay", out=None, root=None):
        if mode not in ("replay", "train"):
            raise ValueError('MODE is "replay" or "train".')
        self.mode = mode
        self.root = Path(root).resolve() if root is not None else Path.cwd().resolve()
        self.folder = self.root / "data" / EXAMPLE
        if not (self.folder / "config.json").exists():
            raise FileNotFoundError("Run the notebook from the root of the repository, next to data/.")
        out = Path(out) if out is not None else Path("outputs", "notebooks", f"{EXAMPLE}-{mode}-{time.strftime('%Y%m%d-%H%M%S')}")
        self.out = out if out.is_absolute() else self.root / out
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"{self.rel(self.out)} is not empty; choose a new OUT.")
        self.out.mkdir(parents=True, exist_ok=True)
        self.settings = training.read(self.folder / "config.json")
        self.selection = training.read(self.folder / "selection.json")
        self._data, self._analyses, self._figures = {}, {}, []
        self._system = self._check = self._summary = None

    def rel(self, path):
        """A path relative to the root of the repository, for display."""
        return os.path.relpath(Path(path).resolve(), self.root)

    def _overview(self):
        device = jax.devices()[0]
        selected = ", ".join(self.selection[policy]["selected"] for policy in POLICIES)
        return Table(["setting", "value"], [
            ["mode", self.mode], ["data", self.rel(self.folder)], ["output folder", self.rel(self.out)],
            ["selected fits", f"{selected} (lowest validation objective, chosen before any CMGDB computation)"],
            ["networks", f"JAX {jax.__version__} on {device.platform}, float32"],
            ["CMGDB", f"{importlib.metadata.version('cmgdb')}, binary64, coefficients in F_5"]])

    def _repr_html_(self):
        return self._overview()._repr_html_()

    def __repr__(self):
        return repr(self._overview())

    # The system and the reference.

    def system(self):
        """Floating-point checks of the closed loop and of the Floquet multipliers of the coning orbit."""
        if self._system is None:
            self._system = system_record()
        s, rows = self._system, []
        for label, key, bound in (("max |η' + 2η|, closed-loop identity", "rate_error_identity", "< 1e-11"),
                                  ("max |q' − V_a(q) − η × q|", "axis_identity", "< 1e-11")):
            rows.append([label, s["nominal"]["identities"][key], s["coning"]["identities"][key], bound])
        equivariance = [max(v for k, v in s[p]["identities"].items() if k not in ("rate_error_identity", "axis_identity"))
                        for p in POLICIES]
        rows.append(["equivariance of q, η, q' and M, max error", *equivariance, "< 1e-11"])
        for label, key in (("RK4 with steps 0.001 and 0.002, max difference over 1 s", "step_comparison_max"),
                           ("observed axis against the flow of V_a on η = 0, max error over 1 s", "sphere_reference_max")):
            rows.append([label, s["nominal"]["integration"][key], s["coning"]["integration"][key], "< 2e-7"])
        decay = [max(row["maximum_rate_decay_error"] for row in s[p]["off_manifold"]) for p in POLICIES]
        rows.append(["|η(t) − exp(−2t) η(0)| for |η(0)| ≤ 6 over 20 s, max", *decay, "< 2e-7"])
        orbit = s["coning_orbit"]
        rows.append(["period of the coning orbit C", "", orbit["period"], f"2 log 2 = {2 * np.log(2):.6g}"])
        rows.append(["Floquet multipliers of C in (q, η)", "", orbit["multipliers"],
                     f"1, exp(−15 log 2/8) = {np.exp(-15 * np.log(2) / 8):.6g}, 1/16 three times"])
        return Table(["check", "nominal", "coning", "bound or exact value"], rows)

    def reference_map(self):
        """The fixed heights of the reference time-one map h_1 and its derivative there."""
        rows = [[row["policy"], _name(row["name"]), row["height"], row["image"], row["derivative"],
                 row["difference_quotient"], "repelling" if row["derivative"] > 1 else "attracting"]
                for row in fixed_heights(self.settings)]
        return Table(["gain", "fixed set", "height z*", "h₁(z*)", "h₁'(z*) = exp(g'(z*))",
                      "difference quotient", ""], rows)

    def reference_graphs(self):
        """The Morse graphs of the reference time-one map on 2^7 cells of [-1, 1] in the coordinate z."""
        rows = []
        for policy in POLICIES:
            graph = reference_graph(policy, settings=self.settings)
            names = node_names(graph)
            for row in sorted(graph.nodes, key=lambda row: row["cell_ids"][0]):
                rows.append([policy, _name(names[row["node"]]), len(row["cell_ids"]), _interval(row),
                             _index(row["annotations"]),
                             ", ".join(_name(names[b]) for a, b in graph.edges if a == row["node"])])
        return Table(["gain", "Morse set", "cells", "cells cover, in z", "Conley index (CMGDB)", "edges to"], rows)

    # The data and the fits.

    def _get_data(self, policy):
        if policy not in self._data:
            self._data[policy] = load_data(self.folder, policy)
        return self._data[policy]

    def data_files(self):
        """The bundled trajectories, checked against the checksums of `metadata.json`."""
        metadata = training.read(self.folder / "metadata.json")
        rows = []
        for policy in POLICIES:
            data = self._get_data(policy)
            for split in SPLITS:
                name = f"{policy}-{split}.npz"
                entry = metadata["files"][name]
                rows.append([policy, split, data[split].shape[0], data[split].shape[1], entry["seed"],
                             entry["sha256"][:16], training.digest(self.folder / name) == entry["sha256"]])
        return Table(["gain", "split", "trajectories", "observations each", "generator seed",
                      "file SHA-256 (first 16)", "checksum matches"], rows)

    def seeds(self):
        """All trained seeds of each gain, ranked by validation objective, with their stored held-out metrics."""
        rows = []
        for policy in POLICIES:
            chosen = self.selection[policy]
            for rank, row in enumerate(chosen["ranking"], 1):
                folder = self.folder / "models" / row["run"]
                config, metrics = training.read(folder / "config.json"), training.read(folder / "metrics.json")
                rows.append([policy, row["run"], rank, row["validation"], metrics["test"]["objective"],
                             metrics["one_second_axis_rms"], config["best_step"],
                             "selected" if row["run"] == chosen["selected"] else ""])
        return Table(["gain", "fit", "rank", "validation objective", "test objective",
                      "one-second axis RMS (test)", "best step", ""], rows)

    def fits(self):
        """Load the selected fit of each gain (replay), or retrain its seed with `config.json` (train)."""
        rows = []
        for policy in POLICIES:
            run = self.selection[policy]["selected"]
            stored = self.folder / "models" / run
            config = training.read(stored / "config.json")
            if self.mode == "replay":
                model, source = load(stored), stored
                check = f"file SHA-256 equals config.json: {_text(training.digest(stored / 'weights.npz') == config['weights_sha256'])}"
            else:
                source = self.out / "models" / run
                train(source, self._get_data(policy), config["seed"], self.settings)
                model = load(source)
                same = parameters_digest(model) == parameters_digest(load(stored))
                check = f"weights equal the stored fit: {_text(same)}"
            self._analyses[policy] = Analysis(model, self._get_data(policy), run)
            rows.append([policy, run, self.rel(source), parameters_digest(model)[:16], check])
        return Table(["gain", "fit", "folder", "weights SHA-256 (first 16)", "check"], rows)

    def _analysis(self, policy):
        if policy not in self._analyses:
            self.fits()
        return self._analyses[policy]

    def errors(self):
        """Held-out errors and sampled checks of the selected fits, recomputed."""
        d = {policy: self._analysis(policy).diagnostics for policy in POLICIES}
        rows = [["one-second axis RMS, componentwise (test trajectories)",
                 *(d[p]["one_second_axis_rms"] for p in POLICIES), "below 0.002"],
                ["validation objective", *(d[p]["validation"]["objective"] for p in POLICIES), ""],
                ["test objective", *(d[p]["test"]["objective"] for p in POLICIES), ""],
                ["axis prediction RMS at 1, 4 and 10 steps (test windows)",
                 *(d[p]["test"]["observable_prediction_rms"] for p in POLICIES), ""],
                ["latent closure RMS at 1, 4 and 10 steps (test windows)",
                 *(d[p]["test"]["physical_closure_rms"] for p in POLICIES), ""],
                ["fixed heights of fbar between the poles", *(d[p]["interior_fixed_latitudes"] for p in POLICIES), ""],
                ["sampled equivariance error of E and f, max",
                 *(max(d[p]["sampled_encoder_equivariance_error"], d[p]["sampled_map_equivariance_error"]) for p in POLICIES), ""],
                ["image hulls against 4,097 samples per cell, largest excess",
                 *(d[p]["dense_image_violation"] for p in POLICIES), "≤ 0"]]
        return Table(["quantity", "nominal", "coning", "Section 5.3"], rows)

    # Morse graphs.

    def learned_graphs(self):
        """The learned Morse graphs on 2^7 cells of [-1, 1] in the coordinate u = psi(z), raw and merged."""
        rows = []
        for policy in POLICIES:
            a = self._analysis(policy)
            raw_names = node_names(a.raw)
            for kind, graph in (("raw", a.raw), ("merged", a.merged)):
                names = node_names(graph)
                for row in sorted(graph.nodes, key=lambda row: row["cell_ids"][0]):
                    merged = ", ".join(_name(raw_names[n]) for n in row.get("original_nodes", []))
                    rows.append([policy, kind, _name(names[row["node"]]), len(row["cell_ids"]), _interval(row),
                                 _index(row["annotations"]),
                                 ", ".join(_name(names[b]) for a_, b in graph.edges if a_ == row["node"]),
                                 merged if kind == "merged" else ""])
        return Table(["gain", "graph", "Morse set", "cells", "cells cover, in u", "Conley index (CMGDB)",
                      "edges to", "Morse sets of the raw graph"], rows)

    def comparison(self):
        """The learned Morse graphs against the reference in z and in the coordinate psi of the fit."""
        rows = []
        for policy in POLICIES:
            a = self._analysis(policy)
            agreement = a.comparison
            for label, graph, flags in (
                    ("reference in z", a.reference_z[1], dict(agreement["reference_z"], cells=None)),
                    ("reference in ψ, raw", a.reference_psi[0], agreement["reference_psi_raw"]),
                    ("reference in ψ, merged", a.reference_psi[1], agreement["reference_psi"]),
                    ("learned, raw", a.raw, None), ("learned, merged", a.merged, None)):
                interior = ", ".join(f"{_name(node_names(graph)[row['node']])} {_interval(row)}" for row in graph.nodes
                                     if 0 not in row["cell_ids"] and len(graph.boxes) - 1 not in row["cell_ids"])
                same = "" if flags is None else "; ".join(f"{key} {_text(value)}" for key, value in flags.items()
                                                          if value is not None)
                rows.append([policy, label, _sets(graph), _order(graph), interior, same])
        return Table(["gain", "graph", "Morse sets and indices", "order", "Morse sets between the poles",
                      "agrees with the learned graph of its kind"], rows,
                     note="Raw graphs are compared with the raw learned graph and merged with merged; cells are "
                          "compared only in the same coordinate.")

    # Attracting regions and indices in X.

    def regions(self):
        """Attracting regions, tolerances and the sampled residual modulo G of the tenth iterate."""
        rows = []
        for policy in POLICIES:
            a = self._analysis(policy)
            cases = [("merged", a.merged, a.regions)] + ([("raw", a.raw, a.raw_regions)] if policy == "coning" else [])
            for kind, graph, found in cases:
                names = node_names(graph)
                for row in found["rows"]:
                    node = next(r for r in graph.nodes if r["node"] == row["node"])
                    members = ", ".join(_name(node_names(a.raw)[n]) for n in node.get("original_nodes", [row["node"]]))
                    rows.append([policy, f"{_name(names[row['node']])} ({members})", kind, row["cells"],
                                 f"[{row['hull'][0]:.6g}, {row['hull'][1]:.6g}]", row["samples"], row["tolerance"],
                                 row["max_residual"], row["passed"]])
        return Table(["gain", "minimal Morse set (Morse sets of the raw graph)", "graph", "cells of 𝒜", "N in u",
                      "samples in E⁻¹(N)", "tolerance", "max residual", "residual < tolerance"], rows,
                     note="Chordal metric on the latent sphere after undoing ψ; samples are pairs of observations "
                          "one time unit apart on η = 0. Image hulls and residuals are sampled, not validated.")

    def indices(self):
        """The Conley indices in X = S^2 reconstructed from the quotient, against the indices proved in the appendix."""
        rows = []
        for policy in POLICIES:
            a = self._analysis(policy)
            for label, key in (("reference in z", "reference_z"), ("reference in ψ, merged", "reference_psi"),
                               ("learned, merged", "learned")):
                for row in sorted(a.indices[key], key=lambda row: ["-e3", "C", "e3"].index(row["name"])
                                  if row["name"] in ("-e3", "C", "e3") else 3):
                    proved = PROVED[policy].get(row["name"])
                    rows.append([policy, label, _name(row["name"]), _index(row["cmgdb"]), _index(row["quotient"]),
                                 row["N_P_cells"], _index(row["in_X"]), _index(proved) if proved else "",
                                 row["in_X"] == proved])
        return Table(["gain", "graph", "Morse set", "index on [-1, 1] (CMGDB)", "index on [-1, 1] (homology)",
                      "cells of N_P/G", "index in S² (reconstructed)", "index in S² (proved)", "agree"], rows,
                     note="Shift classes over F_5 in degrees 0, 1, 2: x-1 is F with the identity, 0 is zero. "
                          "Theorem thm:sphere-reconstruction with m = 1 and δ = id, P/G = {−1, 1}.")

    # Figures.

    def _show(self, paths, titles, columns=None, width=10.):
        self._figures += [Path(path) for path in paths]
        plotting.show(paths, titles, columns, width)

    def figure_nominal(self):
        """fig:attitude-nominal."""
        self._show(figure_attitude(self._analysis("nominal"), self.out / "figures"),
                   ["(a) Computed in [−1, 1]", "(b) Pullback of attracting blocks", "(c) Reconstructed in S²"])

    def figure_coning(self):
        """fig:attitude-computation."""
        self._show(figure_attitude(self._analysis("coning"), self.out / "figures"),
                   ["(a) Computed in [−1, 1]", "(b) Pullback of attracting blocks", "(c) Reconstructed in S²"])

    def figure_coning_raw(self):
        """The raw coning graph and the attracting region of its minimal Morse set C, outside the manuscript."""
        self._show(figure_raw_region(self._analysis("coning"), self.out / "figures"),
                   ["Raw graph, computed in [−1, 1]", "Region of the minimal Morse set C of the raw graph"], width=7.)

    def figure_appendix_spheres(self):
        """fig:appendix-notebook-attitude."""
        self._show([figure_appendix_spheres(self._analysis(policy), self.out / "figures") for policy in POLICIES],
                   ["(a) Nominal control", "(b) Coning control"], width=7.)

    def figure_appendix_comparison(self):
        """fig:appendix-attitude-comparison."""
        self._show(figure_appendix_comparison(self._analysis("coning"), self.out / "figures"),
                   ["Reference: Morse sets on S²", "Learned: Morse sets on Z", "Reference: cells in z",
                    "Learned: cells in u", "Reference: computed in [−1, 1]", "Learned: computed in [−1, 1]",
                    "Reference: reconstructed in S²", "Learned: reconstructed in S²"], columns=2, width=8.)

    # Checks and summary.

    def record(self):
        if self._system is None:
            self._system = system_record()
        return example_record(self._system, {policy: self._analysis(policy) for policy in POLICIES})

    def check(self):
        """The record of this run against `data/ex3_attitude_control/reference/results.json`.

        In replay any difference raises AssertionError; in train mode the differences are reported."""
        record = self.record()
        training.write(self.out / "results.json", record)
        reference = self.folder / "reference" / "results.json"
        count, found = differences(record, training.read(reference))
        graphs = sum(len(record[policy]["graphs"]) for policy in POLICIES)
        self._check = dict(mode=self.mode, reference=self.rel(reference), numbers=count, graphs=graphs,
                           relative_tolerance=1e-9, differences=len(found), first_differences=found[:20],
                           passed=not found)
        if found and self.mode == "replay":
            raise AssertionError(f"The replay differs from {self.rel(reference)} in {len(found)} places: {found[:5]}")
        if found:
            return Line(f"Train mode: {len(found)} of the values differ from {self.rel(reference)} (informational; "
                        f"this run's record is {self.rel(self.out / 'results.json')}).")
        return Line(f"{self.mode.capitalize()} check passed: {count} numbers and {graphs} Morse graphs agree with "
                    f"{self.rel(reference)} (graphs exactly, numbers to relative tolerance 1e-9).")

    def summary(self):
        """Every quantity that the manuscript states for Section 5.3, computed, next to its current value."""
        n, c = self._analysis("nominal"), self._analysis("coning")
        if self._system is None:
            self._system = system_record()
        orbit, rows = self._system["coning_orbit"], []

        def add(statement, where, manuscript, computed, agrees):
            rows.append(dict(statement=statement, where=where, manuscript=manuscript, computed=computed,
                             agrees=agrees if isinstance(agrees, str) else _text(bool(agrees))))

        multipliers = sorted(orbit["multipliers"])
        add("Coning orbit: period and transverse Floquet multipliers in (q, η)", "Section 5.3",
            "period 2 log 2; exp(−15 log 2/8) and 1/16 three times",
            f"period {orbit['period']:.6g}; {multipliers[3]:.6g} and {multipliers[0]:.6g} three times; 1 along C",
            abs(orbit["period"] - 2 * np.log(2)) < 1e-12
            and max(abs(a - b) for a, b in zip(multipliers, sorted(orbit["expected_multipliers"]))) < 1e-11)
        rms = [a.diagnostics["one_second_axis_rms"] for a in (n, c)]
        add("One-second axis RMS on held-out trajectories from η = 0", "Section 5.3", "below 0.002",
            f"{rms[0]:.3g} (nominal), {rms[1]:.3g} (coning)", max(rms) < .002)
        same = n.comparison["reference_z"]
        add("Nominal: reference and learned Morse graphs on 2^7 cells", "Section 5.3",
            "agree; order −e₃ → e₃; matching indices",
            f"reference {_order(n.reference_z[1])}: {_sets(n.reference_z[1])}; learned {_order(n.merged)}: {_sets(n.merged)}",
            same["order"] and same["indices"] and describe(n.merged)["order"] == [["-e3", "e3"]])
        add("Coning: reference Morse graph in z", "Section 5.3", "−e₃ → C ← e₃",
            f"{_order(c.reference_z[1])}: {_sets(c.reference_z[1])}",
            describe(c.reference_z[1])["order"] == [["-e3", "C"], ["e3", "C"]])
        reference_raw, learned_raw = c.reference_psi[0], c.raw

        def fourth(graph):
            names = node_names(graph)
            rows_ = [row for row in graph.nodes if names[row["node"]] == "A"]
            return rows_[0] if len(rows_) == 1 and len(graph.nodes) == 4 else None
        extra = [fourth(reference_raw), fourth(learned_raw)]
        if None in extra:
            computed, ok = (f"reference in ψ: {len(reference_raw.nodes)} Morse sets; learned: "
                            f"{len(learned_raw.nodes)} Morse sets"), False
        else:
            c_cells = describe(learned_raw)["cells"]["C"]
            computed = (f"reference in ψ: A = cell {_interval(extra[0])} {_index(extra[0]['annotations'])}; "
                        f"learned: A = cell {_interval(extra[1])} {_index(extra[1]['annotations'])}")
            ok = (len(extra[1]["cell_ids"]) == 1 and extra[1]["annotations"] == ["0", "0"]
                  and min(abs(extra[1]["cell_ids"][0] - cell) for cell in c_cells) == 1)
        add("Coning: a fourth Morse set in the coordinate ψ", "Section 5.3",
            "reference and learned each have one; learned: one cell next to C with zero index", computed, ok)
        merged_c = next(row for row in c.merged.nodes if node_names(c.merged)[row["node"]] == "C")
        add("Coning: merged Morse set, index recomputed by CMGDB on the merged pair", "Section 5.3",
            "order −e₃ → C ← e₃ of the reference",
            f"C = {_interval(merged_c)} from {len(merged_c['original_nodes'])} Morse sets, {_index(merged_c['annotations'])}; "
            f"{_order(c.merged)}",
            merged_c["annotations"] == ["x-1", "0"] and c.comparison["reference_z"]["order"]
            and c.comparison["reference_z"]["indices"])
        merged_rows = [n.regions["rows"][0], c.regions["rows"][0]]
        add("Sample maxima of the residual modulo G of the tenth iterate (chordal metric)", "Section 5.3",
            "0.015 for both gains", f"{merged_rows[0]['max_residual']:.6g} (nominal), {merged_rows[1]['max_residual']:.6g} (coning)",
            all(round(row["max_residual"], 3) == .015 for row in merged_rows))
        add("Estimated tolerances", "Section 5.3", "0.29 at e₃ (nominal) and 0.17 at the merged region (coning)",
            f"{merged_rows[0]['tolerance']:.6g} at e₃, {merged_rows[1]['tolerance']:.6g} at the merged C",
            round(merged_rows[0]["tolerance"], 2) == .29 and round(merged_rows[1]["tolerance"], 2) == .17
            and all(row["passed"] for row in merged_rows))
        native = c.raw_regions["rows"][0]
        add("Coning: attracting region of the minimal Morse set C of the raw graph", "Section 5, definition of 𝒜",
            "𝒜 is defined for a minimal Morse set; Section 5.3 reports the merged region",
            f"{native['cells']} cells, tolerance {native['tolerance']:.6g} against residual {native['max_residual']:.6g}"
            f" ({'passes' if native['passed'] else 'fails'})", "not stated")
        in_X = []
        for a in (n, c):
            for key in ("reference_z", "learned"):
                in_X += [row["in_X"] == PROVED[a.policy].get(row["name"]) for row in a.indices[key]]
        add("Indices of the axis dynamics in S²", "Section 5.3 and the supplement",
            "H₂ = F at a repelling pole, H₀ = F at the nominal sink, H₀ = H₁ = F at C",
            "; ".join(f"{a.policy}: " + ", ".join(f"{_name(row['name'])} {_index(row['in_X'])}" for row in a.indices["learned"])
                      for a in (n, c)) + " (reference and learned)", all(in_X))
        add("Cells of the attracting regions", "Supplement, attitude computation details", "127 and 126 of the 2^7 cells",
            f"{merged_rows[0]['cells']} and {merged_rows[1]['cells']}",
            [row["cells"] for row in merged_rows] == [127, 126])
        pairs = [(row["tolerance"], row["max_residual"]) for row in merged_rows]
        add("Pairs (tolerance, residual)", "Supplement, attitude computation details",
            "(0.294831, 0.0148710) and (0.172168, 0.0148922)",
            "; ".join(f"({t:.6g}, {r:.6g})" for t, r in pairs),
            [(float(f"{t:.6g}"), float(f"{r:.6g}")) for t, r in pairs] == [(.294831, .014871), (.172168, .0148922)])
        add("Observation pairs per residual", "Supplement, attitude computation details", "98,304",
            f"{len(n.samples[0]):,} (nominal), {len(c.samples[0]):,} (coning)",
            len(n.samples[0]) == len(c.samples[0]) == 98304)
        times = {policy: timings(self._analysis(policy).model) for policy in POLICIES}
        add("Warm native graph and interval-index computation", "Supplement, attitude computation details",
            "0.38 ms (nominal), 0.43 ms (coning)",
            ", ".join(f"{1e3 * times[p]['native_warm_seconds']:.2g} ms ({p})" for p in POLICIES), "depends on the machine")
        add("Image-hull construction", "Supplement, attitude computation details", "about 12 ms per gain",
            ", ".join(f"{1e3 * times[p]['hulls_seconds']:.2g} ms ({p})" for p in POLICIES), "depends on the machine")
        self._summary = dict(rows=rows, timings=times)
        return Table(["statement", "where", "manuscript", "computed", "agrees"],
                     [[row[key] for key in ("statement", "where", "manuscript", "computed", "agrees")] for row in rows])

    def verification(self):
        """Write `verification.json` in the output folder: the check, the summary and the panels."""
        if self._summary is None:
            self.summary()
        path = self.out / "verification.json"
        training.write(path, _plain(dict(
            example=EXAMPLE, mode=self.mode, created=time.strftime("%Y-%m-%d %H:%M:%S"), output=self.rel(self.out),
            fits={policy: self.selection[policy]["selected"] for policy in POLICIES},
            check=self._check, summary=self._summary["rows"], timings=self._summary["timings"],
            figures=sorted({self.rel(p) for p in self._figures}), versions=versions())))
        verdict = "passed" if self._check and self._check["passed"] else "not passed or not run"
        return Line(f"Wrote {self.rel(path)}: check {verdict}, {len(self._summary['rows'])} statements, "
                    f"{len(set(self._figures))} panels as PDF and SVG.")
