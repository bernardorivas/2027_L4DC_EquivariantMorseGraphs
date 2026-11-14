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
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
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
from emlp.groups import SO2eR3
from emlp.nn.objax import EMLP, gated
from emlp.reps import T, V

from . import grids, regions, training

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
    folder.mkdir(parents=True, exist_ok=False)
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
