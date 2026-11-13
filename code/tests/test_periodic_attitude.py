"""Independent checks of the constructed coning controller."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"))
from common import systems
from example3.periodic import controller as ctl


def test_rate_feedback_matches_a_finite_difference_of_the_desired_rate():
    for policy in ("nominal", "coning"):
        state = ctl.sample(32, 718, policy, eta_radius=3.)
        r = systems.rotation(state[:, 6:10])
        q = r[:, :, 2]
        w = np.einsum("nij,nj->ni", r, state[:, 10:13])
        qdot = np.cross(w, q)
        epsilon = 1e-6
        plus, minus = q+epsilon*qdot, q-epsilon*qdot
        plus /= np.linalg.norm(plus, axis=1, keepdims=True)
        minus /= np.linalg.norm(minus, axis=1, keepdims=True)
        derivative = (ctl.desired_rate(plus, policy)-ctl.desired_rate(minus, policy))/(2*epsilon)
        rate_derivative = np.einsum("nij,nj->ni", r, ctl.rhs(state, policy)[:, 10:13])-derivative
        eta = w-ctl.desired_rate(q, policy)
        np.testing.assert_allclose(rate_derivative, -ctl.DAMPING*eta, atol=3e-8, rtol=0)


def test_the_cone_is_periodic_and_interior_heights_move_toward_it():
    angle = np.linspace(0, 2*np.pi, 17)
    radius = np.sqrt(1-ctl.CONE_HEIGHT**2)
    q = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), np.full_like(angle, ctl.CONE_HEIGHT)))
    np.testing.assert_allclose(ctl.axis_flow(q, ctl.PERIOD, "coning"), q, atol=2e-14, rtol=0)
    height = np.array([-1., -.99, -.3, ctl.CONE_HEIGHT, .6, .99, 1.])
    future = ctl.height_flow(height, 2., "coning")
    np.testing.assert_array_equal(future[[0, 3, 6]], height[[0, 3, 6]])
    assert np.all(np.abs(future[1:6]-ctl.CONE_HEIGHT) <= np.abs(height[1:6]-ctl.CONE_HEIGHT))
    assert np.all((future[1:6]-ctl.CONE_HEIGHT)*(height[1:6]-ctl.CONE_HEIGHT) >= 0)


def test_body_moment_is_unchanged_by_vertical_spatial_rotations():
    state = ctl.sample(32, 971, "coning", eta_radius=2.)
    angle = .859
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    transformed = systems.group_action("attitude", state, rotation)
    moment = ctl.components(state, "coning")[-1]
    transformed_moment = ctl.components(transformed, "coning")[-1]
    np.testing.assert_allclose(moment, transformed_moment, atol=2e-14, rtol=0)


def test_full_plant_agrees_with_the_independent_scalar_axis_flow():
    state = ctl.sample(32, 175, "coning")
    q = systems.observe("attitude", state)[:, 12:15]
    physical = systems.observe("attitude", ctl.step(state, .7, "coning"))[:, 12:15]
    np.testing.assert_allclose(physical, ctl.axis_flow(q, .7, "coning"), atol=1e-9, rtol=0)
