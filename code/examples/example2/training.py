"""Joint invariant coordinates and orbit dynamics for the physical O(3) flow."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import objax
from emlp.groups import O
from emlp.nn.objax import EMLP
from emlp.reps import T, V
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from common.checkpoints import load_weights, save_weights

LOWER = np.array([-2., 0.])
UPPER = np.array([2., 2.])
TAU = float(np.log(3.))
NETWORKS = ("encoder", "decoder", "dynamics")
RECIPE = dict(encoder_gain_log_bound=.2, decoder_gain_log_bound=.35, time_gain_log_bound=3.)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(x):
    x = np.asarray(x)
    return np.stack((x[..., 0], np.linalg.norm(x[..., 1:], axis=-1)), axis=-1)


def section(ar):
    ar = np.asarray(ar)
    return np.concatenate((ar, np.zeros((*ar.shape[:-1], 2))), axis=-1)


def physical_step(x, steps=1):
    x = np.asarray(x)
    a, v = x[..., :1], x[..., 1:]
    factor = 9.**steps
    d = 1 + (factor - 1)*a*a
    radial = d + (factor - 1)*np.sum(v*v, axis=-1, keepdims=True)
    return np.concatenate((np.sqrt(factor)*a/np.sqrt(d),
                           np.sqrt(factor)*v/np.sqrt(d*radial)), axis=-1)


def sample(count, seed):
    if count < 8:
        raise ValueError("Each data split needs at least eight starts.")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    a, r = rng.uniform(-1.5, 1.5, count), rng.uniform(0, 1.5, count)
    k = count//8
    a[:k] = rng.choice([-1., 1.], k)*10.**rng.uniform(-8, -1, k)
    r[k:2*k] = 10.**rng.uniform(-8, -1, k)
    centers = np.array([[0., 0.], [0., 1.], [-1., 0.], [1., 0.]])
    perturb = 10.**rng.uniform(-6, -1, (2*k, 2))*rng.choice([-1., 1.], (2*k, 2))
    near = centers[np.arange(2*k) % 4] + perturb
    a[2*k:4*k], r[2*k:4*k] = near[:, 0], np.abs(near[:, 1])
    # Include exact axes and equilibria without merging the signs of a.
    axis = np.arange(4*k, 4*k + max(4, k//4))
    a[axis[::2]] = 0.
    r[axis[1::2]] = 0.
    a[-4:], r[-4:] = centers[:, 0], centers[:, 1]
    return np.column_stack((a, r[:, None]*directions))


def dataset(path, sizes):
    path = Path(path)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        assert all(data[f"{name}_x"].shape == (count, 11, 4)
                   for name, count in zip(("train", "validation", "test"), sizes, strict=True))
        return data
    values = {}
    for name, count, seed in zip(("train", "validation", "test"), sizes,
                                 (20261112, 20261113, 20261114), strict=True):
        x = sample(count, seed)
        values[f"{name}_x"] = np.stack([physical_step(x, k) for k in range(11)], axis=1)
    np.savez_compressed(path, **values)
    write(path.with_suffix(".json"), dict(sizes=list(sizes), seeds=[20261112, 20261113, 20261114],
          coordinates=["a", "v1", "v2", "v3"], time_step=TAU, horizons=list(range(11)),
          formula="closed-form time-log(3) map of the physical vector field",
          near_axis_and_equilibrium_sampling=True, sha256=digest(path)))
    return np.load(path, allow_pickle=False)


class InvariantDynamics(objax.Module):
    """EMLP gain corrections around prescribed orbit coordinates and boundaries."""

    def __init__(self, width=32, layers=2, recipe=None):
        if width < 2 or width % 2:
            raise ValueError("The encoder width must be a positive even integer.")
        self.recipe = dict(RECIPE if recipe is None else recipe)
        group = O(3)
        hidden = (width//2)*T(0) + (width//2)*V
        self.encoder = EMLP(T(0)+V, 2*T(0), group, ch=hidden, num_layers=layers)
        self.decoder = EMLP(2*T(0), 2*T(0), group, ch=width*T(0), num_layers=layers)
        self.dynamics = EMLP(2*T(0), 2*T(0), group, ch=width*T(0), num_layers=layers)

    def encode(self, x):
        ar = jnp.stack((x[..., 0], jnp.linalg.norm(x[..., 1:], axis=-1)), axis=-1)
        gains = jnp.exp(self.recipe["encoder_gain_log_bound"]*jnp.tanh(self.encoder(x/1.5)))
        return ar*gains

    def decode(self, z):
        values = z*jnp.exp(self.recipe["decoder_gain_log_bound"]*jnp.tanh(self.decoder(z/2.)))
        return jnp.clip(values, jnp.array([-1.5, 0.]), jnp.array([1.5, 1.5]))

    def advance(self, z):
        values = z*jnp.exp(self.recipe["time_gain_log_bound"]*jnp.tanh(self.dynamics(z/2.)))
        return 2*jnp.tanh(values/2.)


def orbit(x):
    return jnp.stack((x[..., 0], jnp.linalg.norm(x[..., 1:], axis=-1)), axis=-1)


def loss_parts(model, x, y):
    z, z_next = model.encode(x), model.encode(y)
    predicted = model.advance(z)
    reconstruction = jnp.mean(jnp.sum((model.decode(z)-orbit(x))**2, axis=-1))
    prediction = jnp.mean(jnp.sum((model.decode(predicted)-orbit(y))**2, axis=-1))
    consistency = jnp.mean(jnp.sum((predicted-z_next)**2, axis=-1))
    return jnp.stack((reconstruction, prediction, consistency))


def evaluator(model, method):
    return objax.Jit(objax.Function.with_vars(model.vars())(method))


def train(folder, data, seed, steps, width, layers, batch,
          learning_rate_start=.002, learning_rate_end=.0001, validate_every=100, recipe=None):
    folder.mkdir(parents=True, exist_ok=False)
    np.random.seed(seed)
    objax.random.DEFAULT_GENERATOR.seed(seed)
    model = InvariantDynamics(width, layers, recipe)
    initial = {name: [np.asarray(v.value).copy() for v in getattr(model, name).vars().values()]
               for name in NETWORKS}
    optimizer = objax.optimizer.Adam(model.vars())
    objective = lambda x, y: jnp.sum(loss_parts(model, x, y))
    gradient = objax.GradValues(objective, model.vars())

    @objax.Function.with_vars(model.vars()+optimizer.vars())
    def update(x, y, rate):
        gradients, value = gradient(x, y)
        optimizer(rate, gradients)
        return value[0]

    update = objax.Jit(update)
    validate = evaluator(model, lambda x, y: loss_parts(model, x, y))
    inputs = {name: [jnp.asarray(data[f"{name}_x"][:, i]) for i in (0, 1)]
              for name in ("train", "validation")}
    parts = np.asarray(validate(*inputs["validation"]))
    best, best_step = float(parts.sum()), 0
    save_weights(folder/"weights.npz", model.vars())
    history = [dict(step=0, validation=best, parts=parts.tolist())]
    start = time.perf_counter()
    rng = np.random.default_rng(seed+200)
    write(folder/"status.json", dict(status="training", seed=seed, requested_steps=steps))
    for step in range(1, steps+1):
        ids = rng.integers(0, len(inputs["train"][0]), batch)
        rate = learning_rate_end + (learning_rate_start-learning_rate_end)*(1+np.cos(np.pi*(step-1)/steps))/2
        value = float(update(*(x[ids] for x in inputs["train"]), rate))
        if not np.isfinite(value):
            write(folder/"status.json", dict(status="failed", step=step, reason="nonfinite training loss"))
            raise FloatingPointError("Nonfinite training loss")
        if step == 1 or step % validate_every == 0 or step == steps:
            parts = np.asarray(validate(*inputs["validation"]))
            score = float(parts.sum())
            history.append(dict(step=step, training=value, validation=score, parts=parts.tolist()))
            if score < best:
                best, best_step = score, step
                save_weights(folder/"weights.npz", model.vars())
            if step == 1 or step % 500 == 0 or step == steps:
                print(f"seed {seed}, {step}/{steps}: validation {score:.6g}, elapsed {time.perf_counter()-start:.1f}s", flush=True)
                write(folder/"history.json", history)
    save_weights(folder/"final-weights.npz", model.vars())
    load_weights(folder/"weights.npz", model.vars())
    changes = {name: float(np.sqrt(sum(np.sum((np.asarray(v.value)-old)**2)
                       for v, old in zip(getattr(model, name).vars().values(), initial[name], strict=True))))
               for name in NETWORKS}
    config = dict(seed=seed, steps=steps, width=width, layers=layers, batch=batch,
          best_step=best_step, best_validation=best, seconds=time.perf_counter()-start,
          learning_rate_start=learning_rate_start, learning_rate_end=learning_rate_end,
          validate_every=validate_every, recipe=model.recipe, weights_format="numeric-ordered-v1",
          parameter_changes=changes, parameters={name: int(sum(v.value.size for v in getattr(model, name).vars().values()))
                                                for name in NETWORKS},
          encoder_representation="T(0)+V -> 2*T(0)",
          encoder_hidden_representation=f"{width//2}*T(0)+{width//2}*V",
          decoder_and_dynamics_hidden_representation=f"{width}*T(0)",
          prescribed_geometry=dict(orbit_skip="(a,norm(v))", **model.recipe, latent_lower=LOWER.tolist(),
                 latent_upper=UPPER.tolist(), decoder_clip=[[-1.5, 0.], [1.5, 1.5]],
                 time_output="2*tanh(z*exp(3*tanh(network(z/2)))/2)",
                 scalar_sign_and_radial_zero_preserved=True,
                 learned_object="coordinate change on the known orbit space; not discovery of the quotient"),
          training="joint, three unit-weight squared Euclidean losses", time_step=TAU,
          versions={name: version(name) for name in ("emlp", "objax", "jax", "numpy", "scipy", "cmgdb")},
          weights_sha256=digest(folder/"weights.npz"), source_sha256=digest(__file__))
    write(folder/"config.json", config)
    write(folder/"history.json", history)
    write(folder/"status.json", dict(status="trained", selected_step=best_step))
    return model


def load(folder):
    config = read(folder/"config.json")
    np.random.seed(config["seed"])
    objax.random.DEFAULT_GENERATOR.seed(config["seed"])
    model = InvariantDynamics(config["width"], config["layers"], config.get("recipe"))
    load_weights(folder/"weights.npz", model.vars())
    assert digest(folder/"weights.npz") == config["weights_sha256"]
    return model


def batched(function, values, size=8192):
    return np.concatenate([np.asarray(function(jnp.asarray(values[i:i+size])))
                           for i in range(0, len(values), size)])


def diagnostics(model, data, folder, restrict_axes=False):
    encode, decode, advance = [evaluator(model, getattr(model, name)) for name in ("encode", "decode", "advance")]
    x = data["test_x"]
    codes = batched(encode, x[:, 0])
    predicted = codes.copy()
    forecasts, errors = {}, []
    for horizon in range(11):
        decoded = batched(decode, predicted)
        error = np.linalg.norm(decoded-project(x[:, horizon]), axis=1)
        errors.append(error)
        forecasts[str(horizon)] = dict(orbit_rms=float(np.sqrt(np.mean(error**2))),
                                       orbit_maximum=float(error.max()))
        if horizon < 10:
            predicted = batched(advance, predicted)
    residual = batched(advance, codes)-batched(encode, x[:, 1])
    rng = np.random.default_rng(1929)
    invariance = []
    for reflection in (False, True):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix[:, 0] *= np.linalg.det(matrix)*(-1 if reflection else 1)
        transformed = x[:, 0].copy()
        transformed[:, 1:] = transformed[:, 1:]@matrix.T
        invariance.append(float(np.max(np.abs(batched(encode, transformed)-codes))))
    a, r = np.linspace(-1.5, 1.5, 257), np.linspace(0, 1.5, 257)
    aa, rr = np.meshgrid(a, r, indexing="ij")
    ar = np.column_stack((aa.ravel(), rr.ravel()))
    encoded_grid = batched(encode, section(ar))
    decoded_grid = batched(decode, encoded_grid)
    # Evaluate the section Jacobian away from r=0, where norm has a singular derivative.
    derivative = evaluator(model, jax.vmap(jax.jacfwd(lambda t: model.encode(jnp.array([[t[0],t[1],0.,0.]]))[0])))
    check_grid = ar.reshape(257, 257, 2)[::8, 1::8].reshape(-1, 2)
    jac = batched(derivative, check_grid)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    determinant = np.linalg.det(jac)
    support_codes = np.concatenate([batched(encode, data[f"{name}_x"].reshape(-1, 4))
                                    for name in ("train", "validation")])
    anchors = np.array([[0.,0.],[0.,1.],[-1.,0.],[1.,0.]])
    encoded_anchors = batched(encode, section(anchors))
    roots = [dict(kind="O", latent=[0.,0.])]
    scalar_min, scalar_max, radial_max = -2., 2., 2.
    if restrict_axes:
        scalar_axis = encoded_grid[ar[:,1] == 0,0]
        radial_axis = encoded_grid[ar[:,0] == 0,1]
        scalar_min, scalar_max = float(scalar_axis.min()), float(scalar_axis.max())
        radial_max = float(radial_axis.max())
    for coordinate, intervals, kind in ((0, [(scalar_min,-1e-7),(1e-7,scalar_max)], "scalar"),
                                         (1, [(1e-7,radial_max)], "radial")):
        for left, right in intervals:
            sweep = np.linspace(left, right, 4001)
            points = np.zeros((len(sweep),2))
            points[:,coordinate] = sweep
            displacement = batched(advance, points)[:,coordinate]-sweep
            for index in np.flatnonzero(displacement[:-1]*displacement[1:] < 0):
                def scalar(value):
                    z = np.zeros((1,2)); z[0,coordinate] = value
                    return float(np.asarray(advance(jnp.asarray(z)))[0,coordinate]-value)
                root = brentq(scalar, sweep[index], sweep[index+1], xtol=1e-13)
                z = np.zeros(2); z[coordinate] = root
                roots.append(dict(kind=kind, latent=z.tolist()))
    root_points = np.array([row["latent"] for row in roots])
    decoded_roots = batched(decode, root_points)
    for row, decoded in zip(roots, decoded_roots, strict=True):
        row["decoded_orbit"] = decoded.tolist()
    result = dict(forecasts=forecasts, sampled_encoder_invariance_error=max(invariance),
          invariance_by_rotation_and_reflection=invariance,
          residual_rms=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
          residual_maximum=float(np.linalg.norm(residual, axis=1).max()),
          minimum_sampled_encoder_singular_value=float(singular_values.min()),
          minimum_sampled_encoder_jacobian_determinant=float(determinant.min()),
          maximum_sampled_encoder_jacobian_determinant=float(determinant.max()),
          physical_anchor_names=["O","C","A-","A+"], encoded_anchors=encoded_anchors.tolist(),
          anchor_time_residuals=np.linalg.norm(batched(advance, encoded_anchors)-encoded_anchors, axis=1).tolist(),
          observed_axis_fixed_points=roots,
          support_code_range=[support_codes.min(axis=0).tolist(), support_codes.max(axis=0).tolist()],
          support_source="encoded training and validation trajectories, horizons 0 through 10",
          support_samples=len(support_codes),
          test_code_covariance=np.cov(codes.T).tolist(),
          evidence_status="sampled numerical diagnostics; no global injectivity or transfer certificate")
    np.savez_compressed(folder/"diagnostics.npz", section_ar=ar, section_codes=encoded_grid,
          section_decoded=decoded_grid, jacobian_ar=check_grid, jacobian_determinant=determinant,
          jacobian_singular_values=singular_values, test_codes=codes, test_orbit_errors=np.stack(errors,axis=1),
          support_codes=support_codes, encoded_anchors=encoded_anchors)
    write(folder/"metrics.json", result)
    print(f"seed {read(folder/'config.json')['seed']}: one-step {forecasts['1']['orbit_rms']:.5g}, ten-step {forecasts['10']['orbit_rms']:.5g}", flush=True)
    return result


def verify_physics(out):
    points = sample(64, 2229)
    values = []
    for x in points:
        def rhs(_, y):
            a, v = y[0], y[1:]
            return np.r_[a*(1-a*a), (1-2*a*a-v@v)*v]
        solution = solve_ivp(rhs, (0., 10*TAU), x, t_eval=[TAU,10*TAU],
                             method="DOP853", rtol=2e-13, atol=2e-14)
        assert solution.success
        values.append(solution.y.T)
    values = np.asarray(values)
    exact = np.stack([physical_step(points,k) for k in (1,10)],axis=1)
    errors = np.linalg.norm(values-exact, axis=-1)
    composition = np.max(np.abs(physical_step(physical_step(points, 1), 9)-physical_step(points, 10)))
    result = dict(independent_method="DOP853 on the four-dimensional vector field", starts=64,
                  maximum_state_error_one_step=float(errors[:,0].max()),
                  maximum_state_error_ten_steps=float(errors[:,1].max()),
                  composition_maximum=float(composition), time_step=TAU,
                  passed=bool(errors.max() < 1e-9 and composition < 1e-11))
    assert result["passed"]
    np.savez_compressed(out/"physical-verification.npz", starts=points, integrated=values, closed_form=exact)
    write(out/"physical-verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--sizes", type=int, nargs=3, default=[16384,4096,4096])
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = dataset(args.out/"physical-trajectories.npz", args.sizes)
    verify_physics(args.out)
    for seed in args.seeds:
        folder = args.out/f"seed{seed}"
        model = load(folder) if args.replay else train(folder, data, seed, args.steps, args.width, args.layers, args.batch)
        if not args.skip_diagnostics:
            previous = read(folder/"metrics.json") if args.replay and (folder/"metrics.json").exists() else None
            result = diagnostics(model, data, folder)
            if previous is not None:
                error = max(abs(previous["forecasts"][str(k)]["orbit_rms"]-result["forecasts"][str(k)]["orbit_rms"]) for k in range(11))
                assert error < 1e-12
                write(folder/"checkpoint-replay.json", dict(passed=True, maximum_orbit_rms_difference=error,
                                                           weights_sha256=digest(folder/"weights.npz")))
    ranking = sorted([dict(seed=s, validation=read(args.out/f"seed{s}"/"config.json")["best_validation"])
                      for s in args.seeds], key=lambda row:row["validation"])
    write(args.out/"selection.json", dict(criterion="lowest validation loss, selected before CMGDB",
                                          selected_seed=ranking[0]["seed"], ranking=ranking))
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
