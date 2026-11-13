"""Joint EMLP learning from physical sphere states to an interval, then CMGDB."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache" / "matplotlib"))
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
import CMGDB

from common.exact_enclosures import sphere_box_images
from common import figure_style as style

HERE = Path(__file__).resolve().parent
ENCODER_SHIFT = 0.35
TIME_SHIFT = 1.4
RECIPE = dict(encoder_shift_bound=.35, decoder_shift_bound=.35, time_shift_bound=1.4,
              orbit_log_weight=.03, consistency_log_weight=.08, ordering_weight=.1,
              ordering_minimum_slope=.2, ordering_points=65, ordering_domain=[-.999, .999])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def meridian(z):
    z = jnp.asarray(z)
    return jnp.stack((jnp.sqrt(jnp.maximum(0., (1-z)*(1+z))), jnp.zeros_like(z), z), axis=-1)


def move(u, shift):
    """A bounded change of log latitude, with both endpoints fixed."""
    gain = jnp.tanh(shift)
    return (u + gain) / (1 + u*gain)


class SphereInterval(objax.Module):
    """An invariant encoder, an orbit decoder, and an interval time map.

    The bounded residual heads prescribe the interval endpoints. The decoder
    returns a height, representing a whole rotation orbit. Hidden channels may
    be scalar representations or the upstream mixture of tensor representations.
    """

    def __init__(self, width=32, layers=2, hidden="scalar", recipe=None):
        self.recipe = dict(RECIPE if recipe is None else recipe)
        group = SO2eR3()
        channels = width if hidden == "uniform" else width*T(0)
        self.encoder = EMLP(V, T(0), group=group, ch=channels, num_layers=layers)
        self.decoder = EMLP(T(0), T(0), group=group, ch=channels, num_layers=layers)
        self.dynamics = EMLP(T(0), T(0), group=group, ch=channels, num_layers=layers)

    def encoder_shift(self, p):
        return self.recipe["encoder_shift_bound"]*jnp.tanh(self.encoder(p)[..., 0])

    def encode(self, p):
        return move(p[..., 2], self.encoder_shift(p))

    def decoder_shift(self, u):
        return self.recipe["decoder_shift_bound"]*jnp.tanh(self.decoder(u[..., None])[..., 0])

    def decode(self, u):
        return move(u, self.decoder_shift(u))

    def time_shift(self, u):
        return self.recipe["time_shift_bound"]*jnp.tanh(self.dynamics(u[..., None])[..., 0])

    def advance(self, u):
        return move(u, self.time_shift(u))


def dataset(path, sizes, seed=20261112):
    """Physical starts and their time-one and time-ten states, split by start."""
    if path.exists():
        data = np.load(path, allow_pickle=False)
        for split, count in zip(("train", "validation", "test"), sizes, strict=True):
            if data[f"{split}_p0"].shape != (count, 3):
                raise ValueError("Existing trajectory data have different split sizes.")
        return data
    rng = np.random.default_rng(seed)
    arrays = {}
    for split, count in zip(("train", "validation", "test"), sizes, strict=True):
        z = rng.uniform(-1, 1, count)
        polar = count//4
        z[:polar] = rng.choice([-1., 1.], polar)*(1 - 10.**rng.uniform(-10, -2, polar))
        rng.shuffle(z)
        angle = rng.uniform(0, 2*np.pi, count)
        w = np.arctanh(z)
        solution = solve_ivp(lambda _, a: np.tanh(a)*(.16-np.tanh(a)**2),
                            (0., 10.), w, t_eval=[1., 10.], method="DOP853",
                            rtol=2e-13, atol=2e-14)
        if not solution.success:
            raise RuntimeError(solution.message)
        for tag, latitude, duration in (("0", w, 0.), ("1", solution.y[:, 0], 1.),
                                         ("10", solution.y[:, 1], 10.)):
            radius = 1/np.cosh(latitude)
            phi = angle + .6*duration
            arrays[f"{split}_p{tag}"] = np.column_stack((radius*np.cos(phi), radius*np.sin(phi),
                                                         np.tanh(latitude)))
            arrays[f"{split}_w{tag}"] = latitude
    np.savez_compressed(path, **arrays)
    return np.load(path, allow_pickle=False)


def objective(model, p0, p1, w0, w1):
    """The three joint losses, with reconstruction and prediction modulo rotations."""
    e0 = w0 + model.encoder_shift(p0)
    u0 = jnp.tanh(e0)
    reconstructed = e0 + model.decoder_shift(u0)
    predicted = e0 + model.time_shift(u0)
    decoded_prediction = predicted + model.decoder_shift(jnp.tanh(predicted))
    e1 = w1 + model.encoder_shift(p1)

    def orbit_error(a, b):
        return jnp.mean((jnp.tanh(a)-jnp.tanh(b))**2 +
                        (1/jnp.cosh(a)-1/jnp.cosh(b))**2 + model.recipe["orbit_log_weight"]*(a-b)**2)

    reconstruction = orbit_error(reconstructed, w0)
    prediction = orbit_error(decoded_prediction, w1)
    consistency = jnp.mean((jnp.tanh(predicted)-jnp.tanh(e1))**2
                           + model.recipe["consistency_log_weight"]*(predicted-e1)**2)
    # Ordering is a condition for using the learned interval as orbit coordinates.
    z = jnp.linspace(*model.recipe["ordering_domain"], model.recipe["ordering_points"])
    differences = jnp.diff(model.encode(meridian(z)))/jnp.diff(z)
    ordering = jnp.mean(jax.nn.relu(model.recipe["ordering_minimum_slope"]-differences)**2)
    return reconstruction + prediction + consistency + model.recipe["ordering_weight"]*ordering


def save_weights(path, variables):
    """Store ordered numeric parameters without executable or string metadata."""
    np.savez_compressed(path, **{str(i): np.asarray(v.value) for i, v in enumerate(variables.values())})


def load_weights(path, variables):
    with np.load(path, allow_pickle=False) as saved:
        if "names" in saved.files:
            objax.io.load_var_collection(str(path), variables)
            return
        if set(saved.files) != {str(i) for i in range(len(variables))}:
            raise ValueError("The checkpoint has a different parameter count.")
        values = [saved[str(i)] for i in range(len(variables))]
        if any(a.shape != v.value.shape or a.dtype != v.value.dtype
               for a, v in zip(values, variables.values(), strict=True)):
            raise ValueError("The checkpoint has incompatible parameter shapes or dtypes.")
        variables.assign([jnp.asarray(a) for a in values])


def train(folder, data, seed, width, layers, steps, batch, hidden,
          learning_rate_start=.002, learning_rate_end=.0001, validate_every=100, recipe=None):
    folder.mkdir(parents=True, exist_ok=False)
    np.random.seed(seed)
    objax.random.DEFAULT_GENERATOR.seed(seed)
    model = SphereInterval(width, layers, hidden, recipe)
    initial_parameters = {name: [np.asarray(v.value).copy() for v in getattr(model, name).vars().values()]
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
    inputs = {split: [jnp.asarray(data[f"{split}_{key}"]) for key in ("p0", "p1", "w0", "w1")]
              for split in ("train", "validation")}
    rng = np.random.default_rng(seed+100)
    best, best_step = float(validate(*inputs["validation"])), 0
    save_weights(folder/"weights.npz", model.vars())
    history = [dict(step=0, validation=best)]
    start = time.perf_counter()
    for step in range(1, steps+1):
        ids = rng.integers(0, len(inputs["train"][0]), batch)
        rate = learning_rate_end + (learning_rate_start-learning_rate_end)*(1 + np.cos(np.pi*(step-1)/steps))/2
        training_loss = float(update(*(values[ids] for values in inputs["train"]), rate))
        if not np.isfinite(training_loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        if step == 1 or step % validate_every == 0 or step == steps:
            valid = float(validate(*inputs["validation"]))
            history.append(dict(step=step, training=training_loss, validation=valid))
            if valid < best:
                best, best_step = valid, step
                save_weights(folder/"weights.npz", model.vars())
            if step == 1 or step % 500 == 0 or step == steps:
                print(f"seed {seed}, step {step}/{steps}: validation {valid:.4g}, "
                      f"elapsed {time.perf_counter()-start:.1f}s", flush=True)
    load_weights(folder/"weights.npz", model.vars())
    changed = {name: float(np.sqrt(sum(np.sum((np.asarray(v.value)-a)**2)
                       for v, a in zip(getattr(model, name).vars().values(), initial_parameters[name], strict=True))))
               for name in initial_parameters}
    config = dict(seed=seed, width=width, layers=layers, hidden=hidden, steps=steps, batch=batch,
                  training="joint orbit reconstruction, orbit prediction, and latent consistency",
                  physical_input_coordinates=3, latent_coordinates=1,
                  encoder_shift_bound=ENCODER_SHIFT, decoder_shift_bound=ENCODER_SHIFT,
                  time_shift_bound=TIME_SHIFT, best_step=best_step, best_validation=best,
                  learning_rate_start=learning_rate_start, learning_rate_end=learning_rate_end,
                  validate_every=validate_every, weights_format="numeric-ordered-v1",
                  recipe=model.recipe,
                  seconds=time.perf_counter()-start, parameter_changes=changed,
                  parameters={name: int(sum(v.value.size for v in getattr(model, name).vars().values()))
                              for name in changed},
                  versions={p: version(p) for p in ("emlp", "objax", "jax", "numpy", "scipy", "cmgdb")})
    write(folder/"config.json", config)
    write(folder/"history.json", history)
    return model


def load(folder):
    config = json.loads((folder/"config.json").read_text())
    np.random.seed(config["seed"])
    objax.random.DEFAULT_GENERATOR.seed(config["seed"])
    model = SphereInterval(config["width"], config["layers"], config.get("hidden", "scalar"), config.get("recipe"))
    load_weights(folder/"weights.npz", model.vars())
    return model


def evaluator(model, method):
    return objax.Jit(objax.Function.with_vars(model.vars())(method))


def diagnostics(model, data, folder):
    encode = evaluator(model, model.encode)
    decode = evaluator(model, model.decode)
    advance = evaluator(model, model.advance)
    eta = evaluator(model, lambda z: model.encode(meridian(z)))
    derivative_e = evaluator(model, jax.vmap(jax.grad(lambda z: model.encode(meridian(z[None]))[0])))
    derivative_f = evaluator(model, jax.vmap(jax.grad(lambda u: model.advance(u[None])[0])))
    p0 = jnp.asarray(data["test_p0"])
    encoded = np.asarray(encode(p0))
    values = jnp.asarray(encoded)
    forecasts = {}
    for horizon in range(11):
        if horizon in (0, 1, 10):
            decoded = np.asarray(decode(values))
            target = data[f"test_p{horizon}"][:, 2]
            radius = np.sqrt(np.maximum(0., (1-decoded)*(1+decoded)))
            target_radius = np.linalg.norm(data[f"test_p{horizon}"][:, :2], axis=1)
            forecasts[str(horizon)] = dict(height_rms=float(np.sqrt(np.mean((decoded-target)**2))),
                orbit_chordal_rms=float(np.sqrt(np.mean((decoded-target)**2+(radius-target_radius)**2))),
                height_maximum=float(np.max(np.abs(decoded-target))))
        if horizon < 10:
            values = advance(values)
    test_next = np.asarray(encode(jnp.asarray(data["test_p1"])))
    residual = np.asarray(advance(jnp.asarray(encoded)))-test_next
    residual_ten = np.asarray(values)-np.asarray(encode(jnp.asarray(data["test_p10"])))
    errors = []
    for angle in (.13, .731, 1.7):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = data["test_p0"].copy()
        rotated[:, :2] = rotated[:, :2]@rotation.T
        errors.append(float(np.max(np.abs(np.asarray(encode(jnp.asarray(rotated)))-encoded))))
    sweep = np.unique(np.r_[np.linspace(-1, 1, 4001),
                     1-10.**np.linspace(-10, -2, 200), -1+10.**np.linspace(-10, -2, 200)])
    enc_sweep = np.asarray(eta(jnp.asarray(sweep)))
    inside = sweep[(sweep > -1) & (sweep < 1)]
    e_derivative = np.asarray(derivative_e(jnp.asarray(inside)))
    f_derivative = np.asarray(derivative_f(jnp.asarray(sweep)))
    displacement = np.asarray(advance(jnp.asarray(sweep)))-sweep
    roots = [-1., 1.]
    for i in np.flatnonzero(displacement[:-1]*displacement[1:] < 0):
        roots.append(float(brentq(lambda u: float(advance(jnp.array([u]))[0])-u,
                                  sweep[i], sweep[i+1], xtol=1e-13)))
    roots.extend(map(float, sweep[(displacement == 0) & (np.abs(sweep) < 1-1e-7)]))
    roots = sorted(set(roots))
    fixed_points = []
    for u in roots:
        physical = u if abs(u) == 1 else float(brentq(lambda z: float(eta(jnp.array([z]))[0])-u,
                                                     -1., 1., xtol=1e-13))
        fixed_points.append(dict(latent=u, physical_height=physical,
                                  decoded_height=float(decode(jnp.array([u]))[0])))
    result = dict(forecasts=forecasts, sampled_encoder_invariance_error=max(errors),
                  sampled_residual_rms=float(np.sqrt(np.mean(residual**2))),
                  sampled_residual_maximum=float(np.max(np.abs(residual))),
                  sampled_time_ten_residual_rms=float(np.sqrt(np.mean(residual_ten**2))),
                  sampled_time_ten_residual_maximum=float(np.max(np.abs(residual_ten))),
                  minimum_sampled_encoder_derivative=float(e_derivative.min()),
                  minimum_sampled_time_map_derivative=float(f_derivative.min()),
                  encoder_endpoints=enc_sweep[[0, -1]].tolist(),
                  time_map_endpoints=np.asarray(advance(jnp.array([-1., 1.]))).tolist(),
                  sampled_encoder_increasing=bool(np.all(np.diff(enc_sweep) > 0)),
                  sampled_time_map_increasing=bool(np.all(f_derivative > 0)),
                  observed_sign_change_fixed_points=fixed_points,
                  evidence_status="sampled diagnostics; no uniform residual certificate")
    np.savez_compressed(folder/"diagnostics.npz", physical_height=sweep, encoded_height=enc_sweep,
                        encoder_derivative=e_derivative, time_map_derivative=f_derivative,
                        test_encoded=encoded)
    write(folder/"metrics.json", result)
    return result


def native(folder, depth, boxes, images, status):
    replay = folder.exists()
    if replay:
        with np.load(folder/"cell-images.npz") as saved:
            difference = float(np.max(np.abs(saved["images"]-images)))
            if not np.array_equal(saved["boxes"], boxes) or difference > 2e-13:
                raise ValueError("New image data differ from this run. Use a new output folder.")
    else:
        folder.mkdir(parents=True)
    lookup = dict(zip(map(tuple, boxes), images.tolist(), strict=True))
    model = CMGDB.Model(depth, depth, depth, 10000, [-1.], [1.], lambda b: lookup[tuple(b)])
    morse, phase = CMGDB.ComputeConleyMorseGraph(model)
    nodes = []
    for node in morse.vertices():
        ids = sorted(map(int, morse.morse_set(node)))
        nodes.append(dict(node=int(node), cell_indices=ids,
                          boxes=[morse.phase_space_box(i) for i in ids],
                          native_annotations=list(morse.annotations(node))))
    record = dict(uniform_depth=depth, map_graph_vertex_count=int(phase.num_vertices()),
                  nodes=nodes, edges=sorted(map(list, morse.edges())), image_status=status)
    if replay:
        previous = json.loads((folder/"native-result.json").read_text())
        if record["nodes"] != previous["nodes"] or record["edges"] != previous["edges"]:
            raise ValueError("Recomputed native graph differs from the saved run.")
        write(folder/"replay-verification.json", dict(maximum_image_endpoint_difference=difference,
                                                    cells_edges_and_annotations_match=True))
        return record
    write(folder/"native-result.json", record)
    np.savez_compressed(folder/"cell-images.npz", boxes=boxes, images=images)
    CMGDB.SaveMorseSets(morse, str(folder/"morse-sets.csv"))
    graph = CMGDB.PlotMorseGraph(morse, clist=style.MORSE)
    (folder/"morse-graph.dot").write_text(graph.source)
    (folder/"morse-graph.pdf").write_bytes(graph.pipe(format="pdf"))
    (folder/"morse-graph.png").write_bytes(graph.pipe(format="png"))
    return record


def analyze(model, folder, depths, samples):
    def iterate(u):
        for _ in range(10):
            u = model.advance(u)
        return u
    time_ten = evaluator(model, iterate)
    eta = evaluator(model, lambda z: model.encode(meridian(z)))
    summaries = []
    for depth in depths:
        count = 2**depth
        boundaries = np.linspace(-1, 1, count+1)
        boxes = np.column_stack((boundaries[:-1], boundaries[1:]))
        points = boxes[:, :1] + (boxes[:, 1:]-boxes[:, :1])*np.linspace(0, 1, samples)
        values = np.asarray(time_ten(jnp.asarray(points.reshape(-1)))).reshape(points.shape)
        images = np.column_stack((np.maximum(-1., np.nextafter(values.min(axis=1)-1e-10, -np.inf)),
                                  np.minimum(1., np.nextafter(values.max(axis=1)+1e-10, np.inf))))
        record = native(folder/f"K{count}", depth, boxes, images,
                        f"sampled image hulls, {samples} points per cell, padding 1e-10; not validated")
        physical = []
        for u in boundaries:
            if abs(u) == 1:
                physical.append(float(u))
            else:
                physical.append(float(brentq(lambda z: float(eta(jnp.array([z]))[0])-u,
                                              -1., 1., xtol=1e-13)))
        physical = np.asarray(physical)
        np.savez_compressed(folder/f"K{count}"/"lifted-grid.npz", latent_boundaries=boundaries,
                            physical_boundaries=physical,
                            encoder_inverse_residual=np.asarray(eta(jnp.asarray(physical)))-boundaries)
        summaries.append(dict(cells=count, morse_sets=len(record["nodes"]), edges=record["edges"],
                              annotations=[n["native_annotations"] for n in record["nodes"]]))
        print(f"{folder.name}, K{count}: {len(record['nodes'])} native Morse sets", flush=True)
    write(folder/"cmgdb-summary.json", summaries)
    return summaries


def reference(out, depths):
    for depth in depths:
        count = 2**depth
        folder = out/"reference"/f"K{count}"
        boundaries = np.linspace(-1, 1, count+1)
        boxes = np.column_stack((boundaries[:-1], boundaries[1:]))
        native(folder, depth, boxes, sphere_box_images(boxes), "explicit rational endpoint enclosures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", choices=("uniform", "scalar"), default="scalar")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--sizes", nargs=3, type=int, default=[16384, 4096, 4096])
    parser.add_argument("--depths", nargs="+", type=int, default=[6, 7, 8])
    parser.add_argument("--image-samples", type=int, default=65)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.image_samples < 2:
        parser.error("--image-samples must include both cell endpoints")
    args.out.mkdir(parents=True, exist_ok=True)
    data = dataset(args.out/"physical-trajectories.npz", args.sizes)
    reference(args.out, args.depths)
    report = json.loads((args.out/"summary.json").read_text()) if (args.out/"summary.json").exists() else []
    for seed in args.seeds:
        folder = args.out/f"seed{seed}"
        model = load(folder) if args.analyze_only else train(folder, data, seed, args.width,
                                                           args.layers, args.steps, args.batch, args.hidden)
        metrics = diagnostics(model, data, folder)
        print(f"seed {seed}: ten-step orbit RMS {metrics['forecasts']['10']['orbit_chordal_rms']:.4g}, "
              f"minimum sampled encoder derivative {metrics['minimum_sampled_encoder_derivative']:.4g}", flush=True)
        if not metrics["sampled_encoder_increasing"]:
            raise RuntimeError("The fitted encoder failed the sampled ordering check; keep this failed fit.")
        graphs = analyze(model, folder, args.depths, args.image_samples)
        report = [row for row in report if row["seed"] != seed]
        report.append(dict(seed=seed, metrics=metrics, cmgdb=graphs))
        report.sort(key=lambda row: row["seed"])
        write(args.out/"summary.json", report)
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
