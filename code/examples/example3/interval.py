"""Learn invariant interval coordinates, their orbit decoder, and dynamics."""

import time

from . import sphere
import jax
import jax.numpy as jnp
import numpy as np
import objax
from emlp.nn.objax import EMLP
from emlp.reps import T, V

from common.checkpoints import load_weights, save_weights
from common.workflow import digest, read, write


def meridian(heights):
    return jnp.stack((jnp.sqrt(jnp.maximum(0., 1-heights*heights)), jnp.zeros_like(heights), heights), axis=-1)


class IntervalModel(objax.Module):
    """Positive piecewise-linear coordinates and a generic bounded scalar map."""

    def __init__(self, width=32, layers=2, knots=65, coordinate_log_bound=.35, time_gain=2.):
        self.encoder = EMLP(V, T(0), sphere.GROUP, ch=width*T(0), num_layers=layers)
        self.decoder = EMLP(T(0), T(0), sphere.GROUP, ch=width*T(0), num_layers=layers)
        self.dynamics = EMLP(T(0), T(0), sphere.GROUP, ch=width*T(0), num_layers=layers)
        self.knots = jnp.linspace(-1., 1., knots, dtype=jnp.float32)
        self.midpoints = (self.knots[:-1]+self.knots[1:])/2
        self.coordinate_log_bound, self.time_gain = coordinate_log_bound, time_gain

    def coordinate_values(self, decoder=False):
        inputs = self.midpoints[:, None] if decoder else meridian(self.midpoints)
        raw = (self.decoder if decoder else self.encoder)(inputs)[:, 0]
        increments = jnp.exp(self.coordinate_log_bound*jnp.tanh(raw))
        cumulative = jnp.concatenate((jnp.zeros(1, dtype=raw.dtype), jnp.cumsum(increments)))
        return 2*cumulative/cumulative[-1]-1

    def encode_height(self, heights):
        return jnp.interp(heights, self.knots, self.coordinate_values())

    def encode(self, points):
        return self.encode_height(points[..., 2])

    def decode(self, codes):
        return jnp.interp(codes, self.knots, self.coordinate_values(decoder=True))

    def inverse(self, codes):
        return jnp.interp(codes, self.coordinate_values(), self.knots)

    def advance(self, codes):
        return jnp.clip(codes+self.time_gain*jnp.tanh(self.dynamics(codes[..., None])[..., 0]), -1., 1.)


def functions(model):
    return tuple(objax.Jit(objax.Function.with_vars(model.vars())(method)) for method in
                 (model.encode, model.decode, model.advance, model.inverse))


def objective(model, points, images):
    code, target_code = model.encode(points), model.encode(images)
    predicted = model.advance(code)
    rec = jnp.mean((model.decode(code)-points[:, 2])**2)
    pred = jnp.mean((model.decode(predicted)-images[:, 2])**2)
    consistency = jnp.mean((predicted-target_code)**2)
    return jnp.stack((rec, pred, consistency))


def sample_sources(count, seed):
    rng = np.random.default_rng(seed)
    heights = rng.uniform(-1, 1, count)
    n = count//4
    heights[:n] = rng.choice([-1., 1.], n)*(1-10.**rng.uniform(-7, -2, n))
    angles = rng.uniform(0, 2*np.pi, count)
    radius = np.sqrt(np.maximum(0., 1-heights**2))
    points = np.column_stack((radius*np.cos(angles), radius*np.sin(angles), heights)).astype(np.float32)
    points[-2:] = [[0., 0., -1.], [0., 0., 1.]]
    return points


def make_pairs(sphere_model, folder, sizes=(16384, 4096, 4096)):
    _, _, advance = sphere_model.functions()
    values = {}
    for name, count, seed in zip(("train", "validation", "test"), sizes,
                                 (2026111301, 2026111302, 2026111303), strict=True):
        points = sample_sources(count, seed)
        images = sphere.batched(advance, points)
        if not np.isfinite(images).all():
            raise FloatingPointError("Nonfinite sphere map on the interval-stage source sample.")
        values[name+"_points"], values[name+"_images"] = points, images
    np.savez_compressed(folder / "latent-pairs.npz", **values)
    return values


def train(folder, pairs, seed, settings):
    folder.mkdir(parents=True, exist_ok=False)
    np.random.seed(seed+1000)
    objax.random.DEFAULT_GENERATOR.seed(seed+1000)
    parameters = {name: settings[name] for name in ("width", "layers", "knots", "coordinate_log_bound", "time_gain")}
    model = IntervalModel(**parameters)
    initial = {name: [np.asarray(v.value).copy() for v in getattr(model, name).vars().values()]
               for name in ("encoder", "decoder", "dynamics")}
    optimizer = objax.optimizer.Adam(model.vars())
    loss = lambda a, b: jnp.sum(objective(model, a, b))
    gradient = objax.GradValues(loss, model.vars())

    @objax.Function.with_vars(model.vars()+optimizer.vars())
    def update(a, b, rate):
        gradients, result = gradient(a, b)
        optimizer(rate, gradients)
        finite = jnp.isfinite(result[0])
        for value in gradients:
            finite = finite & jnp.all(jnp.isfinite(value))
        for value in model.vars().tensors():
            finite = finite & jnp.all(jnp.isfinite(value))
        for value in optimizer.vars().tensors():
            finite = finite & jnp.all(jnp.isfinite(value))
        return result[0], finite

    update = objax.Jit(update)
    validate = objax.Jit(objax.Function.with_vars(model.vars())(lambda a, b: objective(model, a, b)))
    sources = {name: [jnp.asarray(pairs[name+suffix]) for suffix in ("_points", "_images")]
               for name in ("train", "validation")}
    best, best_step = float(np.asarray(validate(*sources["validation"])).sum()), 0
    history = [dict(step=0, validation=best)]
    save_weights(folder / "weights.npz", model.vars())
    rng, start_time = np.random.default_rng(seed+1200), time.perf_counter()
    steps = settings["steps"]
    for step in range(1, steps+1):
        ids = rng.integers(len(sources["train"][0]), size=settings["batch"])
        rate = settings["learning_rate_end"]+(settings["learning_rate_start"]-settings["learning_rate_end"])*(1+np.cos(np.pi*(step-1)/steps))/2
        score, finite = update(*(value[ids] for value in sources["train"]), rate)
        score = float(score)
        if not bool(finite):
            write(folder / "failure.json", {"step": step, "reason": "nonfinite interval loss, gradient, or optimizer state"})
            raise FloatingPointError("Nonfinite interval training update.")
        if step == 1 or step % settings["validate_every"] == 0 or step == steps:
            pieces = np.asarray(validate(*sources["validation"]))
            valid = float(pieces.sum())
            history.append(dict(step=step, training=score, validation=valid, parts=pieces.tolist(),
                                seconds=time.perf_counter()-start_time))
            if valid < best:
                best, best_step = valid, step
                save_weights(folder / "weights.npz", model.vars())
            write(folder / "history.json", history)
            if step == 1 or step % 1000 == 0 or step == steps:
                print(f"interval seed {seed}, {step}/{steps}: validation {valid:.6g}, elapsed {time.perf_counter()-start_time:.1f}s", flush=True)
    save_weights(folder / "final-weights.npz", model.vars())
    load_weights(folder / "weights.npz", model.vars())
    changes = {name: float(np.sqrt(sum(np.sum((np.asarray(value.value)-old)**2)
                    for value, old in zip(getattr(model, name).vars().values(), initial[name], strict=True))))
               for name in initial}
    write(folder / "config.json", dict(seed=seed, **settings, best_step=best_step, best_validation=best,
          seconds=time.perf_counter()-start_time, parameter_changes=changes, dtype="float32",
          weights_sha256=digest(folder / "weights.npz"),
          coordinate_geometry="positive piecewise-linear homeomorphism of the known latitude quotient",
          coordinate_derivative_lower_bound=float(np.exp(-2*settings["coordinate_log_bound"])),
          coordinate_derivative_upper_bound=float(np.exp(2*settings["coordinate_log_bound"])),
          time_head="clip(u + time_gain*tanh(network(u)), -1, 1); endpoint images and interior fixed points are learned"))
    return model


def load(folder):
    config = read(folder / "config.json")
    model = IntervalModel(**{name: config[name] for name in
                            ("width", "layers", "knots", "coordinate_log_bound", "time_gain")})
    load_weights(folder / "weights.npz", model.vars())
    return model


def diagnostics(model, pairs, folder):
    encode, decode, advance, inverse = functions(model)
    points, images = pairs["test_points"], pairs["test_images"]
    code = np.asarray(encode(jnp.asarray(points)))
    predicted = np.asarray(advance(jnp.asarray(code)))
    decoded = np.asarray(decode(jnp.asarray(predicted)))
    true_next = np.asarray(encode(jnp.asarray(images)))
    rec = np.asarray(decode(jnp.asarray(code)))
    heights = jnp.linspace(-1., 1., 16385)
    section = meridian(heights)
    coordinates = np.asarray(encode(section))
    knots = np.asarray(model.knots)
    values = np.asarray(model.coordinate_values())
    slopes = np.diff(values)/np.diff(knots)
    rotated = points.copy()
    angle = .731
    rotated[:, 0] = np.cos(angle)*points[:, 0]-np.sin(angle)*points[:, 1]
    rotated[:, 1] = np.sin(angle)*points[:, 0]+np.cos(angle)*points[:, 1]
    invariance = float(np.max(np.abs(np.asarray(encode(jnp.asarray(rotated)))-code)))
    pole_points = jnp.asarray(pairs["train_points"][-2:])
    assert np.array_equal(np.asarray(pole_points), [[0., 0., -1.], [0., 0., 1.]])
    endpoint_images = np.asarray(advance(encode(pole_points)))
    endpoint_targets = np.asarray(encode(jnp.asarray(pairs["train_images"][-2:])))
    result = dict(reconstruction_height_rms=float(np.sqrt(np.mean((rec-points[:, 2])**2))),
          prediction_height_rms=float(np.sqrt(np.mean((decoded-images[:, 2])**2))),
          consistency_rms=float(np.sqrt(np.mean((predicted-true_next)**2))),
          sampled_invariance_error=invariance, minimum_coordinate_slope=float(slopes.min()),
          maximum_coordinate_slope=float(slopes.max()), endpoints=[float(coordinates[0]), float(coordinates[-1])],
          minimum_monotone_grid_increment=float(np.diff(coordinates).min()),
          inverse_identity_maximum=float(np.max(np.abs(np.asarray(inverse(jnp.asarray(coordinates)))-np.asarray(heights)))),
          interval_endpoint_images=endpoint_images.tolist(),
          sphere_quotient_endpoint_targets=endpoint_targets.tolist(),
          endpoint_compatibility_maximum_error=float(np.max(np.abs(endpoint_images-endpoint_targets))),
          finite=bool(np.isfinite(predicted).all()))
    assert result["finite"] and result["minimum_coordinate_slope"] > 0 and result["minimum_monotone_grid_increment"] > 0
    assert result["endpoints"] == [-1., 1.] and invariance == 0
    write(folder / "metrics.json", result)
    np.savez_compressed(folder / "diagnostics.npz", heights=np.asarray(heights), coordinates=coordinates,
                        knot_heights=knots, knot_coordinates=values, test_codes=code, test_predictions=predicted,
                        test_target_codes=true_next)
    return result
