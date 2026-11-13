"""Equivariant full-plant encoder, sphere dynamics, and observation decoder."""

import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache/matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
import objax
import optax
from emlp.groups import SO2eR3
from emlp.nn.objax import EMLP
from emlp.reps import T, V

from common.emlp_blocks import install
from common.workflow import digest, read, write

GROUP = SO2eR3()
OBSERVATION = 5*V + 3*T(0)
HORIZONS = np.array([0, 1, 4, 16])


def block_errors(error):
    return jnp.stack([jnp.mean(error[..., a:b]**2, axis=-1)
                     for a, b in ((0, 3), (3, 6), (6, 15), (15, 18))], axis=-1)


def normalization(observations):
    values = observations.reshape(-1, 18).astype(np.float64)
    mean, scale = np.zeros(18), np.ones(18)
    for i in range(5):
        block = slice(3*i, 3*i+3)
        mean[3*i+2] = values[:, 3*i+2].mean()
        scale[block] = max(float(np.sqrt(np.mean((values[:, block]-mean[block])**2))), 1e-8)
    mean[15:] = values[:, 15:].mean(axis=0)
    scale[15:] = np.maximum(values[:, 15:].std(axis=0), 1e-8)
    return mean.astype(np.float32), scale.astype(np.float32)


class SphereModel:
    """The normalized upstream EMLPs, with the same ordered trainable variables."""

    def __init__(self, width=128, layers=3, seed=0, backend="blocks"):
        np.random.seed(seed)
        objax.random.DEFAULT_GENERATOR.seed(seed)
        self.width, self.layers, self.seed = width, layers, seed
        self.models = [EMLP(source, target, GROUP, ch=width, num_layers=layers)
                       for source, target in ((OBSERVATION, V), (V, OBSERVATION), (V, V))]
        self.variables = [model.vars() for model in self.models]
        self.params = tuple(tuple(jnp.asarray(v.value, dtype=jnp.float32) for v in values.values())
                            for values in self.variables)
        self.projectors = [install(model) for model in self.models] if backend == "blocks" else None
        self.parameters = sum(value.size for group in self.params for value in group)

    def prepare(self, params):
        if self.projectors is None:
            return params
        result = []
        for values, projections in zip(params, self.projectors, strict=True):
            effective = list(values)
            for iw, ib, pw, pb in projections:
                effective[iw] = (pw@values[iw].reshape(-1)).reshape(values[iw].shape)
                effective[ib] = pb@values[ib]
            result.append(tuple(effective))
        return tuple(result)

    def apply(self, index, params, inputs, normalized=True):
        variables = self.variables[index]
        old = variables.tensors()
        try:
            variables.assign(list(params))
            output = self.models[index](inputs)
        finally:
            variables.assign(old)
        if normalized and index in (0, 2):
            output = output/jnp.linalg.norm(output, axis=-1, keepdims=True)
        return output

    def functions(self, params=None):
        effective = self.prepare(self.params if params is None else params)
        return tuple(jax.jit(lambda x, i=i: self.apply(i, effective[i], x)) for i in range(3))

    def forecast(self, params, inputs):
        effective = self.prepare(params)
        z = self.apply(0, effective[0], inputs)
        def body(code, _):
            advanced = self.apply(2, effective[2], code)
            return advanced, advanced
        _, sequence = jax.lax.scan(body, z, None, length=16)
        chosen = jnp.stack((z, sequence[0], sequence[3], sequence[15]), axis=1)
        decoded = self.apply(1, effective[1], chosen.reshape(-1, 3)).reshape(len(inputs), 4, 18)
        return decoded, z, sequence[0]


def save_parameters(path, params):
    np.savez_compressed(path, **{str(i): np.asarray(value) for i, value in
                                enumerate(value for group in params for value in group)})


def load_parameters(path, template):
    with np.load(path, allow_pickle=False) as values:
        expected = [value for group in template for value in group]
        if set(values.files) != {str(i) for i in range(len(expected))}:
            raise ValueError("The sphere checkpoint has a different parameter count.")
        flat = []
        for i, value in enumerate(expected):
            array = values[str(i)]
            if array.shape != value.shape or array.dtype != np.float32:
                raise ValueError("The sphere checkpoint has incompatible shapes or dtypes.")
            flat.append(jnp.asarray(array))
    result, start = [], 0
    for group in template:
        result.append(tuple(flat[start:start+len(group)]))
        start += len(group)
    return tuple(result)


def load(folder, backend="blocks"):
    config = read(folder / "config.json")
    model = SphereModel(config["width"], config["layers"], config["seed"], backend)
    model.params = load_parameters(folder / "weights.npz", model.params)
    with np.load(folder / "normalization.npz") as saved:
        model.mean, model.scale = saved["mean"], saved["scale"]
    return model


def batched(function, values, size=2048):
    return np.concatenate([np.asarray(function(jnp.asarray(values[start:start+size], dtype=jnp.float32)))
                           for start in range(0, len(values), size)])


def evaluate(model, params, windows, size=256):
    if not hasattr(model, "validation_function"):
        model.validation_function = jax.jit(lambda p, x: block_errors(model.forecast(p, x[:, 0])[0]-x))
    function = model.validation_function
    total, count = np.zeros((4, 4)), 0
    for start in range(0, len(windows), size):
        batch = windows[start:start+size]
        n = len(batch)
        if n < size:
            batch = np.concatenate((batch, np.repeat(batch[-1:], size-n, axis=0)))
        scores = np.asarray(function(params, jnp.asarray(batch)))[:n]
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite sphere validation forecast.")
        total += scores.sum(axis=0)
        count += n
    errors = total/count
    return dict(prediction=float(errors[1:].mean()), reconstruction=float(errors[0].mean()),
                block_mse_by_horizon=errors.tolist(), windows=count)


def train(folder, train_observations, validation, seed, settings):
    folder.mkdir(parents=True, exist_ok=False)
    model = SphereModel(settings["width"], settings["layers"], seed)
    mean, scale = normalization(train_observations)
    train_data = ((train_observations-mean)/scale).astype(np.float32)
    valid = ((validation-mean)/scale).astype(np.float32)
    np.savez_compressed(folder / "normalization.npz", mean=mean, scale=scale)
    starts = np.random.default_rng(8181).integers(17, size=len(valid))
    windows = valid[np.arange(len(valid))[:, None], starts[:, None]+HORIZONS]
    np.save(folder / "validation-starts.npy", starts)
    params = model.params
    steps = settings["steps"]
    schedule = optax.cosine_decay_schedule(settings["learning_rate_start"]-settings["learning_rate_end"], steps)
    optimizer = optax.adamw(lambda t: schedule(t)+settings["learning_rate_end"], weight_decay=settings["weight_decay"])
    state = optimizer.init(params)

    def objective(p, batch):
        predicted, code, next_code = model.forecast(p, batch[:, 0])
        effective = model.prepare(p)
        target_code = model.apply(0, effective[0], batch[:, 1])
        errors = block_errors(predicted-batch)
        rec, pred = jnp.mean(errors[:, 0]), jnp.mean(errors[:, 1:])
        consistency = jnp.mean((next_code-target_code)**2)
        return rec+pred+.1*consistency, jnp.stack((rec, pred, consistency))

    @jax.jit
    def update(p, optimizer_state, batch):
        (loss, pieces), gradient = jax.value_and_grad(objective, has_aux=True)(p, batch)
        changes, optimizer_state = optimizer.update(gradient, optimizer_state, p)
        updated = optax.apply_updates(p, changes)
        finite = jnp.isfinite(loss)
        for value in jax.tree_util.tree_leaves((gradient, updated, optimizer_state)):
            finite = finite & jnp.all(jnp.isfinite(value))
        return updated, optimizer_state, loss, pieces, finite

    initial = evaluate(model, params, windows)
    best, best_step = (initial["prediction"], initial["reconstruction"]), 0
    history = [dict(step=0, validation=initial)]
    save_parameters(folder / "weights.npz", params)
    rng, start_time = np.random.default_rng(10900+seed), time.perf_counter()
    for step in range(1, steps+1):
        ids = rng.integers(len(train_data), size=settings["batch"])
        times = rng.integers(17, size=settings["batch"])
        batch = train_data[ids[:, None], times[:, None]+HORIZONS]
        updated, updated_state, loss, pieces, finite = update(params, state, jnp.asarray(batch))
        if not bool(finite):
            save_parameters(folder / "last-finite-weights.npz", params)
            write(folder / "failure.json", {"step": step, "reason": "nonfinite sphere loss, gradient, or optimizer state"})
            raise FloatingPointError("Nonfinite sphere training update.")
        params, state = updated, updated_state
        if step == 1 or step % settings["validate_every"] == 0 or step == steps:
            valid = evaluate(model, params, windows)
            history.append(dict(step=step, validation=valid, training=float(loss), seconds=time.perf_counter()-start_time))
            key = (valid["prediction"], valid["reconstruction"])
            if key < best:
                best, best_step = key, step
                save_parameters(folder / "weights.npz", params)
            write(folder / "history.json", history)
            print(f"sphere seed {seed}, {step}/{steps}: prediction {valid['prediction']:.6g}", flush=True)
    save_parameters(folder / "final-weights.npz", params)
    write(folder / "config.json", dict(seed=seed, **settings, dtype="float32", parameters=model.parameters,
          best_step=best_step, best_validation_prediction=best[0], best_validation_reconstruction=best[1],
          origin="fresh joint training", weights_sha256=digest(folder / "weights.npz")))
    return load(folder)
