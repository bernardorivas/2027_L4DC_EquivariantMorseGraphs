"""Check the physical dynamics, observation actions, and attitude reference maps."""

import argparse
import json

import numpy as np
from scipy.integrate import solve_ivp

from . import systems


def maximum(x):
    return float(np.max(np.abs(x)))


def eta(state, policy):
    r = systems.rotation(state[..., 6:10])
    q = r[..., :, 2]
    gain, _ = systems.attitude_gain(q[..., 2:3], policy)
    desired = (gain * np.cross(q, systems.E3)
               + systems.PRECESSION * (systems.E3 - q[..., 2:3]*q))
    return np.einsum("...ij,...j->...i", r, state[..., 10:13]) - desired


def check(with_emlp=False):
    angle = 0.731
    q2 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    q3 = np.eye(3)
    q3[:2, :2] = q2
    reflection = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    report = {}
    for name in systems.SYSTEMS:
        x = systems.sample_states(name, 32, seed=701)
        matrix = {"sphere": q2, "o3": reflection, "attitude": q3}[name]
        observations = systems.observe(name, x)
        policies = ("nominal", "coning") if name == "attitude" else ("nominal",)
        for policy in policies:
            advanced = systems.step(name, x, duration=0.1, policy=policy)
            split = systems.step(name, systems.step(name, x, duration=0.04, policy=policy),
                                  duration=0.06, policy=policy)
            transformed = systems.group_action(name, x, matrix)
            action_error = maximum(systems.step(name, transformed, duration=0.1, policy=policy)
                                   - systems.group_action(name, advanced, matrix))
            observation_error = maximum(systems.observe(name, transformed)
                                        - systems.observation_action(name, observations, matrix))
            result = {"semigroup_error": maximum(advanced - split),
                      "flow_equivariance_error": action_error,
                      "observation_equivariance_error": observation_error,
                      "observation_dimension": observations.shape[-1]}
            assert result["semigroup_error"] < 5e-9
            assert action_error < 5e-9 and observation_error < 1e-12
            if name == "sphere":
                result["sphere_norm_error"] = maximum(np.linalg.norm(advanced, axis=-1) - 1)
                assert result["sphere_norm_error"] < 1e-12
            if name == "attitude":
                result["eta_decay_error"] = maximum(eta(advanced, policy) - np.exp(-0.2)*eta(x, policy))
                assert result["eta_decay_error"] < 2e-7
                r = systems.rotation(advanced[:, 6:10])
                result["rotation_error"] = maximum(r.swapaxes(-1, -2) @ r - np.eye(3))
                assert result["rotation_error"] < 1e-12
            report[f"{name}/{policy}"] = result
        if with_emlp:
            group, observation, latent = systems.emlp_representations(name)
            rho = np.asarray(observation(group).rho_dense(matrix))
            error = maximum(observations @ rho.T - systems.observation_action(name, observations, matrix))
            assert error < 2e-6
            report[name + "/representation"] = {
                "observation_dimension": observation(group).size(),
                "latent_dimension": latent(group).size(), "packing_error": error}
    vertices = np.unique(np.r_[np.linspace(-1, 1, 33), 0.25, 0.875])
    for policy in ("nominal", "coning"):
        reference = solve_ivp(lambda t, z: systems.attitude_gain(z, policy)[0] * (1 - z*z),
                              (0, np.log(2)), vertices, method="DOP853", rtol=2e-13, atol=2e-14)
        assert reference.success
        bounds = np.array([systems.attitude_height_enclosure(z, policy) for z in vertices])
        error = maximum(bounds.mean(axis=1) - reference.y[:, -1])
        assert np.all(bounds[:, 0] <= bounds[:, 1]) and error < 2e-10
        assert np.all(np.diff(bounds.mean(axis=1)) > 0)
        report[f"attitude/{policy}/height"] = {"independent_ode_error": error,
                                                "maximum_enclosure_width": float(np.max(np.diff(bounds, axis=1)))}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlp", action="store_true", help="Also check optional EMLP representation packing")
    args = parser.parse_args()
    print(json.dumps(check(args.emlp), indent=2))
