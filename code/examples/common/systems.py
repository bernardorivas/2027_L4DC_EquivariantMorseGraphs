"""Dynamics, observations, group actions, and reference maps for three examples."""

import argparse
from fractions import Fraction
import json
from math import inf, nextafter
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


SYSTEMS = ("sphere", "o3", "attitude")
SPACING = {"sphere": 1.0, "o3": float(np.log(3)), "attitude": 0.1}
E3 = np.array([0.0, 0.0, 1.0])
MASS, GRAVITY = 1.0, 9.81
INERTIA = np.array([0.02, 0.025, 0.04])
PRECESSION = np.pi / np.log(2)


def _system(name):
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system {name!r}; choose one of {SYSTEMS}")


def _policy(policy):
    if policy not in ("nominal", "coning"):
        raise ValueError("The attitude policy must be nominal or coning")


def _ball(rng, count, dimension, radius):
    directions = rng.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    uniform = rng.random((count, 1))
    radial = np.cbrt(uniform) if dimension == 3 else uniform ** (1 / dimension)
    return radius * radial * directions


def sample_states(system, count, seed=0):
    """Independent physical states; attitude quaternions are scalar first.

    Sphere states are (x,y,z), O(3) states are (a,v1,v2,v3), and attitude
    states are (p[3],velocity[3],quaternion[4],Omega_body[3]).
    """
    _system(system)
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    if system == "sphere":
        z, angle = rng.uniform(-1, 1, count), rng.uniform(0, 2 * np.pi, count)
        radius = np.sqrt((1 - z) * (1 + z))
        return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    if system == "o3":
        direction = rng.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        a, radius = rng.uniform(-1.5, 1.5, count), rng.uniform(0, 1.5, count)
        return np.column_stack((a, radius[:, None] * direction))
    position, velocity = _ball(rng, count, 3, 1), _ball(rng, count, 3, 0.5)
    quaternion = rng.normal(size=(count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    omega = _ball(rng, count, 3, 6)
    return np.column_stack((position, velocity, quaternion, omega))


def rotation(quaternion):
    """Rotation matrix of a scalar-first unit quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("A quaternion cannot have zero norm")
    a, b, c, d = np.moveaxis(quaternion / norm, -1, 0)
    entries = (1 - 2 * (c*c + d*d), 2 * (b*c - a*d), 2 * (b*d + a*c),
               2 * (b*c + a*d), 1 - 2 * (b*b + d*d), 2 * (c*d - a*b),
               2 * (b*d - a*c), 2 * (c*d + a*b), 1 - 2 * (b*b + c*c))
    return np.stack(entries, axis=-1).reshape(quaternion.shape[:-1] + (3, 3))


def attitude_gain(z, policy="nominal"):
    """The two closed-loop gains and their derivatives."""
    _policy(policy)
    z = np.asarray(z, dtype=np.float64)
    if policy == "nominal":
        return np.ones_like(z), np.zeros_like(z)
    s = (z - 2/3) / (1 - 2*z/3)
    return 8 * (s*s - 0.25), 16 * s * (5/9) / (1 - 2*z/3)**2


def attitude_rhs(state, policy="nominal"):
    """Full rigid-body dynamics with constant thrust and moment feedback.

    The state has 13 coordinates because the attitude is stored as a quaternion.
    Position is not regulated. The moment has no actuator saturation.
    """
    _policy(policy)
    state = np.asarray(state, dtype=np.float64)
    r = rotation(state[..., 6:10])
    q, omega = r[..., :, 2], state[..., 10:13]
    w = np.einsum("...ij,...j->...i", r, omega)
    z = q[..., 2:3]
    gain, derivative_gain = attitude_gain(z, policy)
    desired = gain * np.cross(q, E3) + PRECESSION * (E3 - z*q)
    eta, qdot = w - desired, np.cross(w, q)
    zdot = qdot[..., 2:3]
    derivative = (derivative_gain * zdot * np.cross(q, E3)
                  + gain * np.cross(qdot, E3)
                  - PRECESSION * (zdot*q + z*qdot))
    moment = (np.cross(omega, INERTIA * omega)
              + INERTIA * np.einsum("...ji,...j->...i", r, derivative - 2*eta))
    omega_dot = (moment - np.cross(omega, INERTIA * omega)) / INERTIA
    scalar, vector = state[..., 6:7], state[..., 7:10]
    quaternion_dot = 0.5 * np.concatenate(
        (-np.sum(vector * omega, axis=-1, keepdims=True),
         scalar * omega + np.cross(vector, omega)), axis=-1)
    thrust = MASS * GRAVITY
    acceleration = -GRAVITY * E3 + (thrust / MASS) * q
    return np.concatenate((state[..., 3:6], acceleration, quaternion_dot, omega_dot), axis=-1)


def step(system, state, duration=None, policy="nominal", corotating=False, dt=0.001):
    """Advance physical states, independently of any learned model.

    corotating removes the constant longitudinal rotation of the sphere flow.
    The attitude step uses fourth-order Runge--Kutta and quaternion normalization.
    """
    _system(system)
    duration = SPACING[system] if duration is None else float(duration)
    if duration < 0 or dt <= 0:
        raise ValueError("duration must be nonnegative and dt must be positive")
    x = np.asarray(state, dtype=np.float64)
    if duration == 0:
        return x.copy()
    if system == "sphere":
        shape = x.shape
        flat = x.reshape(-1, 3)
        z = flat[:, 2]
        solution = solve_ivp(lambda t, h: h * (0.4**2 - h*h) * (1 - h*h),
                             (0, duration), z, method="DOP853", rtol=2e-13, atol=2e-14)
        if not solution.success:
            raise RuntimeError(solution.message)
        height = np.clip(solution.y[:, -1], -1, 1)
        longitude = np.arctan2(flat[:, 1], flat[:, 0])
        if not corotating:
            longitude += 0.6 * duration
        radius = np.sqrt((1 - height) * (1 + height))
        return np.column_stack((radius*np.cos(longitude), radius*np.sin(longitude), height)).reshape(shape)
    if system == "o3":
        a, vector = x[..., :1], x[..., 1:]
        exponential = np.exp(2 * duration)
        denominator = 1 + (exponential - 1) * a*a
        radial = denominator + (exponential - 1) * np.sum(vector*vector, axis=-1, keepdims=True)
        return np.concatenate((np.sqrt(exponential) * a / np.sqrt(denominator),
                               np.sqrt(exponential) * vector / np.sqrt(denominator * radial)), axis=-1)
    _policy(policy)
    x = x.copy()
    count = max(1, int(np.ceil(duration / dt)))
    h = duration / count
    for _ in range(count):
        k1 = attitude_rhs(x, policy)
        k2 = attitude_rhs(x + h*k1/2, policy)
        k3 = attitude_rhs(x + h*k2/2, policy)
        k4 = attitude_rhs(x + h*k3, policy)
        x += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x[..., 6:10] /= np.linalg.norm(x[..., 6:10], axis=-1, keepdims=True)
    if not np.isfinite(x).all():
        raise FloatingPointError("The attitude trajectory became nonfinite")
    return x


def observe(system, state):
    """Return observation coordinates in the order used by the representations.

    Sphere and O(3) observations contain three scalars followed by two vectors.
    Attitude observations contain p, velocity, three columns of R, then Omega.
    """
    _system(system)
    x = np.asarray(state, dtype=np.float64)
    if system == "attitude":
        r = rotation(x[..., 6:10])
        return np.concatenate((x[..., :6], r.swapaxes(-1, -2).reshape(x.shape[:-1] + (9,)),
                               x[..., 10:13]), axis=-1)
    a, vector = (x[..., 2:3], x[..., :2]) if system == "sphere" else (x[..., :1], x[..., 1:])
    a2, r2 = a*a, np.sum(vector*vector, axis=-1, keepdims=True)
    scalars = np.concatenate((a * (1 + 0.15*a2), a * (0.7 + 0.12*r2), r2 + 0.3*a2), axis=-1)
    v1, v2 = (1 + 0.13*a2 + 0.10*r2) * vector, (0.75 + 0.16*a2 + 0.06*r2) * vector
    return np.concatenate((scalars, v1, v2), axis=-1)


def group_action(system, state, matrix):
    """Apply SO(2) to sphere or attitude states, or O(3) to the O(3) system.

    matrix is 2 by 2 for the sphere, and 3 by 3 for the other systems.
    Attitude matrices must rotate about the vertical; body Omega is fixed.
    Quaternion representatives may change sign under composition: q and -q
    describe the same physical attitude, and the observation action is exact.
    """
    _system(system)
    matrix = np.asarray(matrix, dtype=np.float64)
    dimension = 2 if system == "sphere" else 3
    if matrix.shape != (dimension, dimension) or not np.allclose(matrix.T @ matrix, np.eye(dimension)):
        raise ValueError("matrix must be an orthogonal matrix of the required dimension")
    result = np.array(state, dtype=np.float64, copy=True)
    if system == "sphere":
        if not np.isclose(np.linalg.det(matrix), 1):
            raise ValueError("The rotating sphere action consists of SO(2) rotations")
        result[..., :2] = result[..., :2] @ matrix.T
    elif system == "o3":
        result[..., 1:] = result[..., 1:] @ matrix.T
    else:
        if not np.allclose(matrix @ E3, E3) or not np.isclose(np.linalg.det(matrix), 1):
            raise ValueError("The attitude action consists of vertical rotations")
        result[..., :3] = result[..., :3] @ matrix.T
        result[..., 3:6] = result[..., 3:6] @ matrix.T
        angle = np.arctan2(matrix[1, 0], matrix[0, 0])
        a, vector = result[..., 6:7].copy(), result[..., 7:10].copy()
        scalar, left = np.cos(angle/2), np.array([0, 0, np.sin(angle/2)])
        result[..., 6:7] = scalar*a - np.sum(left*vector, axis=-1, keepdims=True)
        result[..., 7:10] = scalar*vector + a*left + np.cross(left, vector)
    return result


def observation_action(system, observations, matrix):
    """Apply the direct-sum observation representation to row vectors."""
    _system(system)
    result = np.array(observations, dtype=np.float64, copy=True)
    if system == "attitude":
        result[..., :15] = (result[..., :15].reshape(result.shape[:-1] + (5, 3))
                           @ np.asarray(matrix).T).reshape(result.shape[:-1] + (15,))
    else:
        dimension = 2 if system == "sphere" else 3
        result[..., 3:] = (result[..., 3:].reshape(result.shape[:-1] + (2, dimension))
                          @ np.asarray(matrix).T).reshape(result.shape[:-1] + (2*dimension,))
    return result


def trajectories(system, initial, intervals=32, spacing=None, policy="nominal", corotating=False, dt=0.001):
    """Return states with shape (trajectory, time, state coordinate)."""
    if intervals < 0:
        raise ValueError("intervals must be nonnegative")
    spacing = SPACING[system] if spacing is None else float(spacing)
    states = [np.array(initial, dtype=np.float64, copy=True)]
    for _ in range(intervals):
        states.append(step(system, states[-1], spacing, policy, corotating, dt))
    return np.stack(states, axis=1)


def emlp_representations(system, latent="sphere"):
    """Construct the group, observation representation, and latent representation.

    EMLP is imported only when this function is called. For O(3), latent is
    a scalar plus a vector. For attitude, sphere and v1 use one vector.
    """
    _system(system)
    from emlp.groups import O, SO, SO2eR3
    from emlp.reps import T, V

    if system in ("sphere", "o3"):
        group = SO(2) if system == "sphere" else O(3)
        return group, 3*T(0) + 2*V, T(0) + V
    group = SO2eR3()
    observation = 5*V + 3*T(0)
    choices = {"sphere": V, "v1": V, "v2": 2*V, "v3": 3*V, "observation": observation}
    if latent not in choices:
        raise ValueError(f"Unknown attitude latent space {latent!r}")
    return group, observation, choices[latent]


def make_emlp_models(system, latent="sphere", width=128, layers=3):
    """Return E, D, and f with the specified input and output representations."""
    from emlp.nn.objax import EMLP

    group, observation, code = emlp_representations(system, latent)
    return tuple(EMLP(source, target, group=group, ch=width, num_layers=layers)
                 for source, target in ((observation, code), (code, observation), (code, code)))


def _mobius(z, shift):
    return (z + shift) / (1 + shift*z)


def attitude_height_enclosure(height, policy="nominal", iterations=54):
    """Enclose the exact time-log(2) reduced map at a binary64 height.

    The coning map is solved by rational bisection of its separated equation.
    Each returned endpoint is rounded outward to binary64.
    """
    _policy(policy)
    z = Fraction(float(height))
    if not -1 <= z <= 1:
        raise ValueError("height must be in [-1,1]")
    if policy == "nominal":
        lo = hi = _mobius(z, Fraction(3, 5))
    else:
        shift = Fraction(2, 3)
        s = _mobius(z, -shift)
        roots = (Fraction(-1), Fraction(-1, 2), Fraction(1, 2), Fraction(1))
        if s in roots:
            return float(z), float(z)
        lo, hi = next((a, b) for a, b in zip(roots[:-1], roots[1:]) if a < s < b)

        def separated(y):
            return ((1 + y) / (1 - y)) * ((y - Fraction(1, 2)) / (y + Fraction(1, 2)))**2

        target, increasing = 2**12 * separated(s), s < -Fraction(1, 2) or s > Fraction(1, 2)
        for _ in range(iterations):
            mid = (lo + hi) / 2
            value = separated(mid)
            if value == target:
                lo = hi = mid
                break
            if (value < target) == increasing:
                lo = mid
            else:
                hi = mid
        lo, hi = _mobius(lo, shift), _mobius(hi, shift)
    lower, upper = float(lo), float(hi)
    if Fraction(lower) > lo:
        lower = nextafter(lower, -inf)
    if Fraction(upper) < hi:
        upper = nextafter(upper, inf)
    return lower, upper


def attitude_baseline(output, policy="nominal", depth=7):
    """Compute the height quotient of the thrust-axis flow at eta=0.

    The full plant retains translation and heading, even on eta=0, and its
    quotient is not one dimensional.
    """
    import CMGDB

    _policy(policy)
    if depth < 2:
        raise ValueError("depth must be at least two")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    count = 2**depth
    vertices = np.linspace(-1, 1, count + 1)
    bounds = dict(zip(vertices, (attitude_height_enclosure(z, policy) for z in vertices)))

    def box_map(box):
        lo, hi = box
        return [bounds[lo][0], bounds[hi][1]]

    model = CMGDB.Model(depth, depth, depth, 10_000, [-1.0], [1.0], box_map)
    morse, phase = CMGDB.ComputeConleyMorseGraph(model)
    heights = {"S": -1.0, "N": 1.0}
    if policy == "coning":
        heights.update(C=0.25, D=0.875)
    rows, names = [], {}
    for node in morse.vertices():
        boxes = morse.morse_set_boxes(node)
        labels = [name for name, z in heights.items() if any(a <= z <= b for a, b in boxes)]
        if len(labels) != 1:
            raise ValueError(f"A component has {len(labels)} named fixed heights")
        names[node] = labels[0]
        rows.append({"node": node, "name": labels[0], "boxes": boxes,
                     "quotient_annotations": list(morse.annotations(node))})
    expected_edges = [["S", "N"]] if policy == "nominal" else [["D", "C"], ["D", "N"], ["S", "C"]]
    named_edges = sorted([names[a], names[b]] for a, b in morse.edges())
    if named_edges != sorted(expected_edges) or sorted(names.values()) != sorted(heights):
        raise ValueError("The native graph disagrees with the reference order")
    expected_annotations = {"S": ["0", "0"], "N": ["x-1", "0"],
                            "C": ["x-1", "0"], "D": ["0", "x-1"]}
    if any(row["quotient_annotations"] != expected_annotations[row["name"]] for row in rows):
        raise ValueError("The native quotient annotations disagree with the reference")
    degrees = {"S": [2], "N": [0], "C": [0, 1], "D": [1, 2]}
    record = {"policy": policy, "subsystem": "thrust-axis flow at eta=0", "sampling_time": float(np.log(2)), "cells": phase.num_vertices(),
              "coefficient_field": 5, "nodes": rows, "edges": named_edges,
              "original_space_degrees": {name: degrees[name] for name in heights}}
    (output / f"{policy}.json").write_text(json.dumps(record, indent=2) + "\n")
    CMGDB.SaveMorseSets(morse, str(output / f"{policy}.csv"))
    (output / f"{policy}.dot").write_text(CMGDB.PlotMorseGraph(morse).source)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate physical trajectories and observations")
    generate.add_argument("system", choices=SYSTEMS)
    generate.add_argument("output", type=Path)
    generate.add_argument("--count", type=int, default=1024)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--intervals", type=int, default=32)
    generate.add_argument("--spacing", type=float)
    generate.add_argument("--policy", choices=("nominal", "coning"), default="nominal")
    generate.add_argument("--corotating", action="store_true")
    generate.add_argument("--physical-only", action="store_true", help="Save physical states and times only")
    baseline = commands.add_parser("attitude-baseline", help="Compute the exact reference quotient graphs")
    baseline.add_argument("output", type=Path)
    baseline.add_argument("--policy", choices=("nominal", "coning", "both"), default="both")
    baseline.add_argument("--depth", type=int, default=7)
    args = parser.parse_args()
    if args.command == "attitude-baseline":
        policies = ("nominal", "coning") if args.policy == "both" else (args.policy,)
        for policy in policies:
            result = attitude_baseline(args.output, policy, args.depth)
            print(policy, result["cells"], result["edges"])
        return
    if args.output.suffix != ".npz":
        parser.error("The trajectory output must have suffix .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("The trajectory output already exists")
    states = trajectories(args.system, sample_states(args.system, args.count, args.seed),
                          args.intervals, args.spacing, args.policy, args.corotating)
    spacing = SPACING[args.system] if args.spacing is None else args.spacing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(states=states, times=np.arange(args.intervals + 1) * spacing)
    if not args.physical_only:
        arrays["observations"] = observe(args.system, states)
    np.savez_compressed(args.output, **arrays)
    metadata = {"system": args.system, "seed": args.seed, "count": args.count,
                "intervals": args.intervals, "spacing": spacing, "policy": args.policy,
                "corotating": args.corotating,
                "physical_only": args.physical_only,
                "integration_method": {"sphere": "DOP853", "o3": "closed form", "attitude": "RK4"}[args.system],
                "integration_step": 0.001 if args.system == "attitude" else None}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.system, states.shape, args.output)


if __name__ == "__main__":
    main()
