"""The rotating sphere flow and its height quotient."""

import numpy as np
from scipy.integrate import solve_ivp

C = 1.0
Z_T = 0.4
OMEGA = 0.6
AZIM = np.deg2rad(-50.0)


def H(z):
    return C * z * (Z_T**2 - z**2) * (1.0 - z**2)


def V(t, p):
    """Vector field on R^3, tangent to S^2."""
    x, y, z = p
    rot = OMEGA * np.array([-y, x, 0.0])
    return rot + C * z * (Z_T**2 - z**2) * (np.array([0.0, 0.0, 1.0]) - z * p)


def latitude(z, n=400):
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    r = np.sqrt(max(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), np.full(n, z)])


def split_runs(mask):
    """Index runs where mask is True."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def trajectory(z0, phi0, t_max=80.0):
    r0 = np.sqrt(1.0 - z0 * z0)
    p0 = np.array([r0 * np.cos(phi0), r0 * np.sin(phi0), z0])

    def near_tropic(t, p):
        return abs(abs(p[2]) - Z_T) - 0.02

    near_tropic.terminal = True
    sol = solve_ivp(V, (0.0, t_max), p0, rtol=1e-9, atol=1e-11, max_step=0.02, events=near_tropic)
    P = sol.y.T
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def hmap(z0, tau, rtol=1e-12, atol=1e-14):
    """h_tau at the points z0, by solve_ivp on the decoupled system z_i' = H(z_i)."""
    sol = solve_ivp(lambda t, z: H(z), (0.0, tau), np.asarray(z0, float),
                    method="DOP853", rtol=rtol, atol=atol)
    return np.clip(sol.y[:, -1], -1.0, 1.0)
