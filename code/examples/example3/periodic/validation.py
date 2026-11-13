"""Physical controller identities, equivariance, integration, and orbital checks."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
from scipy.integrate import solve_ivp
from common import systems
from common.workflow import write
from example3.periodic import controller as ctl


def physical_checks(folder):
    folder.mkdir(parents=True, exist_ok=True)
    results, trajectories = {}, {}
    angle = .731
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    for policy in ("nominal", "coning"):
        x = ctl.sample(258, 1850, policy, eta_radius=3.)[:-2]
        r, q, eta, qdot, _ = ctl.components(x, policy)
        dx = ctl.rhs(x, policy)
        wdot = np.einsum("...ij,...j->...i", r, dx[:, 10:13])
        eta_dot = wdot-ctl.derivative_rate(q, qdot, policy)
        errors = dict(rate_error_identity=float(np.max(np.abs(eta_dot+ctl.DAMPING*eta))),
                      axis_identity=float(np.max(np.abs(qdot-ctl.field(q, policy)-np.cross(eta, q)))))
        transformed = systems.group_action("attitude", x, rotation)
        _, tq, teta, tqdot, tmoment = ctl.components(transformed, policy)
        _, _, _, _, moment = ctl.components(x, policy)
        errors.update(axis_action=float(np.max(np.abs(tq-q @ rotation.T))),
                      eta_action=float(np.max(np.abs(teta-eta @ rotation.T))),
                      axis_field_equivariance=float(np.max(np.abs(tqdot-qdot @ rotation.T))),
                      body_moment_equivariance=float(np.max(np.abs(tmoment-moment))))
        assert max(errors.values()) < 1e-11
        initial = ctl.sample(130, 1851, policy)[:-2]
        fine, coarse = ctl.step(initial, 1., policy, dt=.001), ctl.step(initial, 1., policy, dt=.002)
        fine_obs, coarse_obs = systems.observe("attitude", fine), systems.observe("attitude", coarse)
        initial_axis = systems.observe("attitude", initial)[:, 12:15]
        exact_axis = ctl.axis_flow(initial_axis, 1., policy)
        convergence = dict(step_comparison_max=float(np.max(np.abs(fine_obs-coarse_obs))),
                           sphere_reference_max=float(np.max(np.abs(coarse_obs[:, 12:15]-exact_axis))),
                           maximum_rate_error=float(np.max(np.linalg.norm(ctl.components(coarse, policy)[2], axis=1))))
        assert convergence["step_comparison_max"] < 2e-7
        assert convergence["sphere_reference_max"] < 2e-7
        rows = []
        for radius in (0., .25, 2., 6.):
            initial = ctl.sample(34, 1900, policy, eta_radius=radius)[:-2]
            initial_eta = ctl.components(initial, policy)[2]
            x = initial.copy()
            axes, etas, torques = [], [], []
            for index in range(201):
                _, q, eta, _, moment = ctl.components(x, policy)
                axes.append(q)
                etas.append(eta)
                torques.append(moment)
                if index < 200:
                    x = ctl.step(x, .1, policy)
            axes, etas, torques = np.stack(axes), np.stack(etas), np.stack(torques)
            times = np.arange(201)*.1
            expected = np.exp(-ctl.DAMPING*times)[:, None, None]*initial_eta[None]
            eta_error = float(np.max(np.abs(etas-expected)))
            target = 1. if policy == "nominal" else ctl.CONE_HEIGHT
            distance = np.abs(axes[-1, :, 2]-target)
            # Extremely near-polar starts can have long physical transients.
            rows.append(dict(eta_radius=radius, trajectories=len(initial),
                             maximum_rate_decay_error=eta_error,
                             final_height_error_median=float(np.median(distance)),
                             final_height_error_max=float(distance.max()),
                             maximum_moment_norm=float(np.linalg.norm(torques, axis=-1).max())))
            trajectories[f"{policy}-r{radius}-axis"] = axes
            trajectories[f"{policy}-r{radius}-eta"] = etas
            assert eta_error < 2e-7
        results[policy] = dict(identities=errors, integration=convergence, off_manifold=rows)
        print(policy, results[policy], flush=True)
    c, period = ctl.CONE_HEIGHT, ctl.PERIOD
    radius = np.sqrt(1-c*c)
    def variational(t, values):
        qx, qy = radius*np.cos(ctl.PRECESSION*t), radius*np.sin(ctl.PRECESSION*t)
        a = np.zeros((5, 5))
        a[0, 0] = -(1-c*c)
        a[0, 2:5] = [qy, -qx, 0.]
        a[1, 2:5] = [-c*qx/(1-c*c), -c*qy/(1-c*c), 1.]
        a[2:, 2:] = -ctl.DAMPING*np.eye(3)
        return (a @ values.reshape(5, 5)).ravel()
    answer = solve_ivp(variational, (0., period), np.eye(5).ravel(), method="DOP853", rtol=1e-12, atol=1e-13)
    multipliers = np.sort(np.linalg.eigvals(answer.y[:, -1].reshape(5, 5)).real)
    expected = np.sort([1., np.exp(-(1-c*c)*period), *([np.exp(-ctl.DAMPING*period)]*3)])
    assert np.max(np.abs(multipliers-expected)) < 1e-11
    results["coning_orbit"] = dict(height=c, period=period, angular_rate=ctl.PRECESSION,
        radial_exponent=-(1-c*c), rate_exponents=[-ctl.DAMPING]*3,
        multipliers=multipliers.tolist(), expected_multipliers=expected.tolist(),
        pole_tangent_expansion_rates=dict(south=1+c, north=1-c),
        mathematical_status="analytic identities and Floquet calculation establish local exponential orbital stability in (q,eta)",
        numerical_status="floating-point trajectory and variational checks")
    trajectories["times"] = np.arange(201)*.1
    np.savez_compressed(folder/"trajectories.npz", **trajectories)
    write(folder/"validation.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    physical_checks(args.out)
