"""Rigid-body feedback for upright or autonomous coning-axis stabilization."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from common import systems

E3 = systems.E3
J = systems.INERTIA
CONE_HEIGHT = .25
DAMPING = 2.
PRECESSION = systems.PRECESSION
PERIOD = 2*np.pi/PRECESSION


def gain(z, policy):
    if policy == "nominal":
        return np.ones_like(z), np.zeros_like(z)
    if policy != "coning":
        raise ValueError("Choose nominal or coning.")
    return CONE_HEIGHT-z, -np.ones_like(z)


def field(q, policy):
    z = q[..., 2:3]
    a, _ = gain(z, policy)
    return a*(E3-z*q)+PRECESSION*np.cross(E3, q)


def desired_rate(q, policy):
    return np.cross(q, field(q, policy))


def derivative_rate(q, qdot, policy):
    z, zdot = q[..., 2:3], qdot[..., 2:3]
    a, derivative = gain(z, policy)
    return (derivative*zdot*np.cross(q, E3)+a*np.cross(qdot, E3)
            -PRECESSION*(zdot*q+z*qdot))


def components(state, policy):
    r = systems.rotation(state[..., 6:10])
    q, omega = r[..., :, 2], state[..., 10:13]
    w = np.einsum("...ij,...j->...i", r, omega)
    eta, qdot = w-desired_rate(q, policy), np.cross(w, q)
    derivative = derivative_rate(q, qdot, policy)
    moment = (np.cross(omega, J*omega)
              + J*np.einsum("...ji,...j->...i", r, derivative-DAMPING*eta))
    return r, q, eta, qdot, moment


def rhs(state, policy):
    state = np.asarray(state, dtype=np.float64)
    _, q, _, _, moment = components(state, policy)
    omega = state[..., 10:13]
    scalar, vector = state[..., 6:7], state[..., 7:10]
    quaternion_dot = .5*np.concatenate((-np.sum(vector*omega, axis=-1, keepdims=True),
                                         scalar*omega+np.cross(vector, omega)), axis=-1)
    omega_dot = (moment-np.cross(omega, J*omega))/J
    return np.concatenate((state[..., 3:6], systems.GRAVITY*(q-E3), quaternion_dot, omega_dot), axis=-1)


def step(state, duration=.1, policy="coning", dt=.002):
    x = np.array(state, dtype=np.float64, copy=True)
    count = max(1, int(np.ceil(duration/dt)))
    h = duration/count
    for _ in range(count):
        k1 = rhs(x, policy)
        k2 = rhs(x+h*k1/2, policy)
        k3 = rhs(x+h*k2/2, policy)
        k4 = rhs(x+h*k3, policy)
        x += h*(k1+2*k2+2*k3+k4)/6
        x[..., 6:10] /= np.linalg.norm(x[..., 6:10], axis=-1, keepdims=True)
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite controlled rigid-body trajectory.")
    return x


def sample(count, seed, policy, eta_radius=0.):
    rng = np.random.default_rng(seed)
    state = systems.sample_states("attitude", count, seed)
    z = rng.uniform(-1, 1, count)
    n = count//4
    z[:n] = rng.choice([-1., 1.], n)*(1-10.**rng.uniform(-7, -2, n))
    z[-2:] = [-1., 1.]
    angles = np.column_stack((rng.uniform(0, 2*np.pi, count), np.arccos(z),
                              rng.uniform(0, 2*np.pi, count)))
    rot = Rotation.from_euler("ZYZ", angles)
    state[:, 6:10] = rot.as_quat(scalar_first=True)
    r = rot.as_matrix()
    q = r[:, :, 2]
    eta = systems._ball(rng, count, 3, eta_radius)
    state[:, 10:13] = np.einsum("...ji,...j->...i", r, desired_rate(q, policy)+eta)
    return state


def height_flow(z, duration, policy):
    z = np.asarray(z, dtype=np.float64)
    shape = z.shape
    if duration == 0:
        return z.copy()
    answer = solve_ivp(lambda _, y: gain(y, policy)[0]*(1-y*y), (0., duration), z.reshape(-1),
                       method="DOP853", rtol=2e-13, atol=2e-14)
    if not answer.success:
        raise RuntimeError(answer.message)
    return answer.y[:, -1].reshape(shape)


def axis_flow(q, duration, policy):
    z = height_flow(q[..., 2], duration, policy)
    phi = np.arctan2(q[..., 1], q[..., 0])+PRECESSION*duration
    radius = np.sqrt(np.maximum(0., 1-z*z))
    return np.stack((radius*np.cos(phi), radius*np.sin(phi), z), axis=-1)
