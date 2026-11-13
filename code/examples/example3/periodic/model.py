"""Joint task-aware EMLPs with sphere preservation and a shared quotient map."""

import time

from example3 import sphere
import jax
import jax.numpy as jnp
import numpy as np
import objax
import optax
from emlp.nn.objax import EMLP
from emlp.reps import T, V

from common.emlp_blocks import install
from common.workflow import digest, read, write

HORIZONS = np.array([0, 1, 4, 10])
DEFAULTS = dict(width=64, layers=2, scalar_width=24, knots=33, boost_bound=.5,
                angle_bound=1., coordinate_log_bound=.35, steps=6000, warmup_steps=1000, batch=256,
                learning_rate_start=.002, learning_rate_end=.00005, validate_every=250)


class OrbitModel:
    """Learn E, the axis decoder, latitude and angle maps, pi, and orbit decoder.

    The quotient coordinate factors through the known SO(2) orbit height.
    The boost head has no fixed interior zeros or sign restrictions.
    """

    def __init__(self, seed=0, backend="blocks", **settings):
        self.settings = DEFAULTS | settings
        self.seed = seed
        np.random.seed(seed)
        objax.random.DEFAULT_GENERATOR.seed(seed)
        s = self.settings
        reps = [(sphere.OBSERVATION, V, s["width"]), (V, V, s["width"]),
                (T(0), 2*T(0), s["scalar_width"]*T(0)),
                (T(0), T(0), s["scalar_width"]*T(0)),
                (T(0), T(0), s["scalar_width"]*T(0))]
        self.models = [EMLP(a, b, sphere.GROUP, ch=width, num_layers=s["layers"])
                       for a, b, width in reps]
        self.variables = [model.vars() for model in self.models]
        self.params = tuple(tuple(jnp.asarray(v.value, dtype=jnp.float32) for v in values.values())
                            for values in self.variables)
        self.projectors = [install(model) for model in self.models] if backend == "blocks" else None
        self.parameters = sum(v.size for group in self.params for v in group)
        self.knots = jnp.linspace(-1., 1., s["knots"], dtype=jnp.float32)
        self.midpoints = (self.knots[:-1]+self.knots[1:])/2

    prepare = sphere.SphereModel.prepare

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
        return raw/jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)

    def decode(self, p, q):
        raw = self.network(1, p[1], q)
        return raw/jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)

    def heads(self, p, z):
        raw = self.network(2, p[2], z.reshape(-1, 1)).reshape(z.shape+(2,))
        return (self.settings["boost_bound"]*jnp.tanh(raw[..., 0]),
                self.settings["angle_bound"]*jnp.tanh(raw[..., 1]))

    def latitude(self, p, z):
        b, _ = self.heads(p, z)
        ch, sh = jnp.cosh(b), jnp.sinh(b)
        return (z*ch+sh)/(ch+z*sh)

    def advance(self, p, q):
        z = q[..., 2]
        b, alpha = self.heads(p, z)
        ch, sh = jnp.cosh(b), jnp.sinh(b)
        den = ch+z*sh
        ca, sa = jnp.cos(alpha), jnp.sin(alpha)
        return jnp.stack(((ca*q[..., 0]-sa*q[..., 1])/den,
                          (sa*q[..., 0]+ca*q[..., 1])/den, (z*ch+sh)/den), axis=-1)

    def coordinate_values(self, p, decoder=False):
        i = 4 if decoder else 3
        raw = self.network(i, p[i], self.midpoints[:, None])[:, 0]
        increments = jnp.exp(self.settings["coordinate_log_bound"]*jnp.tanh(raw))
        cumulative = jnp.concatenate((jnp.zeros(1, dtype=raw.dtype), jnp.cumsum(increments)))
        return 2*cumulative/cumulative[-1]-1

    def coordinate(self, p, z):
        return jnp.interp(z, self.knots, self.coordinate_values(p))

    def inverse(self, p, u):
        return jnp.interp(u, self.coordinate_values(p), self.knots)

    def orbit_decode(self, p, u):
        return jnp.interp(u, self.knots, self.coordinate_values(p, decoder=True))

    def quotient(self, p, u, steps=1):
        z = self.inverse(p, u)
        for _ in range(steps):
            z = self.latitude(p, z)
        return self.coordinate(p, z)

    def functions(self):
        p = self.prepare(self.params)
        methods = (self.encode, self.decode, self.advance, self.coordinate, self.inverse,
                   self.orbit_decode, self.latitude)
        return tuple(jax.jit(lambda x, method=method: method(p, x)) for method in methods)

    def forecast(self, p, x):
        q = self.encode(p, x)
        def body(code, _):
            advanced = self.advance(p, code)
            return advanced, advanced
        _, sequence = jax.lax.scan(body, q, None, length=10)
        codes = jnp.stack((q, sequence[0], sequence[3], sequence[9]), axis=1)
        decoded = self.decode(p, codes.reshape(-1, 3)).reshape(len(x), 4, 3)
        return codes, decoded


def pieces(model, params, batch):
    p = model.prepare(params)
    codes, predicted = model.forecast(p, batch[:, 0])
    target = batch[..., 12:15]
    encoded = model.encode(p, batch.reshape(-1, 18)).reshape(len(batch), 4, 3)
    rec = jnp.mean((predicted[:, 0]-target[:, 0])**2)
    pred = jnp.mean((predicted[:, 1:]-target[:, 1:])**2)
    calibration = jnp.mean((encoded-target)**2)
    closure = jnp.mean((codes[:, 1:]-encoded[:, 1:])**2)
    z = encoded[..., 2]
    h = model.coordinate(p, z)
    orbit_rec = jnp.mean((model.orbit_decode(p, h)-z)**2)
    hp = model.coordinate(p, codes[..., 2])
    orbit_pred = jnp.mean((model.orbit_decode(p, hp[:, 1:])-z[:, 1:])**2)
    quotient_consistency = jnp.mean((hp[:, 1:]-h[:, 1:])**2)
    return jnp.stack((rec, pred, calibration, closure, orbit_rec, orbit_pred, quotient_consistency))


WEIGHTS = jnp.array([1., 3., 2., .5, .5, .5, .1])
WARMUP_WEIGHTS = jnp.array([1., 0., 4., 0., .5, 0., 0.])


def evaluate(model, params, windows, size=256):
    if not hasattr(model, "validation_function"):
        model.validation_function = jax.jit(lambda p, x: pieces(model, p, x))
    total = np.zeros(7)
    for start in range(0, len(windows), size):
        batch = windows[start:start+size]
        n = len(batch)
        if n < size:
            batch = np.concatenate((batch, np.repeat(batch[-1:], size-n, axis=0)))
        value = np.asarray(model.validation_function(params, jnp.asarray(batch)))
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite validation objective.")
        total += value*n
    total /= len(windows)
    return dict(objective=float(total @ np.asarray(WEIGHTS)), parts=total.tolist(),
                observable_prediction_rms=float(np.sqrt(total[1])),
                encoder_calibration_rms=float(np.sqrt(total[2])),
                physical_closure_rms=float(np.sqrt(total[3])))


def train(folder, observations, validation, seed, settings):
    folder.mkdir(parents=True, exist_ok=False)
    settings = DEFAULTS | settings
    model = OrbitModel(seed=seed, **settings)
    mean, scale = sphere.normalization(observations)
    # The physical axis is both input and target, so preserve its original scale.
    mean[12:15], scale[12:15] = 0., 1.
    train_data = ((observations-mean)/scale).astype(np.float32)
    valid = ((validation-mean)/scale).astype(np.float32)
    model.mean, model.scale = mean, scale
    np.savez_compressed(folder/"normalization.npz", mean=mean, scale=scale)
    starts = np.random.default_rng(8181).integers(23, size=len(valid))
    windows = valid[np.arange(len(valid))[:, None], starts[:, None]+HORIZONS]
    np.save(folder/"validation-starts.npy", starts)
    params = model.params
    initial_params = [np.concatenate([np.asarray(v).ravel() for v in group]) for group in params]
    steps = settings["steps"]
    schedule = optax.cosine_decay_schedule(settings["learning_rate_start"]-settings["learning_rate_end"], steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1.),
                           optax.adamw(lambda t: schedule(t)+settings["learning_rate_end"], weight_decay=1e-6))
    state = optimizer.init(params)

    @jax.jit
    def update(p, optimizer_state, batch, weights):
        def objective(values):
            parts = pieces(model, values, batch)
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
    sphere.save_parameters(folder/"weights.npz", params)
    rng, start_time = np.random.default_rng(10900+seed), time.perf_counter()
    for step in range(1, steps+1):
        ids = rng.integers(len(train_data), size=settings["batch"])
        times = rng.integers(23, size=settings["batch"])
        batch = train_data[ids[:, None], times[:, None]+HORIZONS]
        weights = WARMUP_WEIGHTS if step <= settings["warmup_steps"] else WEIGHTS
        updated, updated_state, loss, finite = update(params, state, jnp.asarray(batch), weights)
        if not bool(finite):
            sphere.save_parameters(folder/"last-finite-weights.npz", params)
            write(folder/"failure.json", dict(step=step, reason="nonfinite training update"))
            raise FloatingPointError("Nonfinite periodic-model training update.")
        params, state = updated, updated_state
        if step == 1 or step % settings["validate_every"] == 0 or step == steps:
            score = evaluate(model, params, windows)
            history.append(dict(step=step, validation=score, training=float(loss), seconds=time.perf_counter()-start_time))
            if score["objective"] < best:
                best, best_step = score["objective"], step
                sphere.save_parameters(folder/"weights.npz", params)
            write(folder/"history.json", history)
            print(f"joint seed {seed}, {step}/{steps}: prediction {score['observable_prediction_rms']:.5g}, "
                  f"closure {score['physical_closure_rms']:.5g}, {time.perf_counter()-start_time:.1f}s", flush=True)
    sphere.save_parameters(folder/"final-weights.npz", params)
    model.params = sphere.load_parameters(folder/"weights.npz", params)
    changes = {name: float(np.linalg.norm(np.concatenate([np.asarray(v).ravel() for v in group])-old))
               for name, group, old in zip(("encoder", "axis_decoder", "time_heads", "coordinate", "orbit_decoder"),
                                            model.params, initial_params, strict=True)}
    write(folder/"config.json", dict(seed=seed, **settings, parameters=model.parameters,
          best_step=best_step, best_validation=best, parameter_changes=changes,
          seconds=time.perf_counter()-start_time, weights_sha256=digest(folder/"weights.npz"),
          objective="task-aware axis reconstruction, prediction, calibration, and consistency",
          loss_names=["axis_reconstruction", "axis_prediction", "axis_calibration", "physical_closure",
                      "orbit_reconstruction", "orbit_prediction", "quotient_consistency"],
          loss_weights=np.asarray(WEIGHTS).tolist(), horizons=HORIZONS.tolist(), dtype="float32"))
    return model


def load(folder, backend="blocks"):
    config = read(folder/"config.json")
    model = OrbitModel(seed=config["seed"], backend=backend,
                       **{name: config.get(name, default) for name, default in DEFAULTS.items()})
    model.params = sphere.load_parameters(folder/"weights.npz", model.params)
    with np.load(folder/"normalization.npz") as saved:
        model.mean, model.scale = saved["mean"], saved["scale"]
    return model
