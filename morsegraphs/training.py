"""Checkpoints, learning-rate schedules, progress records, and selection by validation error.

Each example module trains its own networks with its own objective. What they share is
here: weights are saved as ordered numeric arrays, a run folder holds `config.json`,
`history.json` and a `status.json` progress record, and among several seeds the fit with
the lowest validation objective is selected before any CMGDB computation.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine_rate(step, steps, start, end):
    """The learning rate at `step` of `steps`, decreasing from `start` to `end`."""
    return end + (start - end) * (1 + np.cos(np.pi * (step - 1) / steps)) / 2


def save_weights(path, variables):
    """Save an Objax variable collection as ordered numeric arrays."""
    np.savez_compressed(path, **{str(i): np.asarray(v.value) for i, v in enumerate(variables.values())})


def load_weights(path, variables):
    """Load ordered numeric arrays into an Objax variable collection of the same shapes."""
    import jax.numpy as jnp
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {str(i) for i in range(len(variables))}:
            raise ValueError("The checkpoint has a different parameter count.")
        values = [saved[str(i)] for i in range(len(variables))]
    if any(a.shape != v.value.shape or a.dtype != v.value.dtype
           for a, v in zip(values, variables.values(), strict=True)):
        raise ValueError("The checkpoint has incompatible parameter shapes or dtypes.")
    variables.assign([jnp.asarray(a) for a in values])


def save_arrays(path, groups):
    """Save a tuple of tuples of arrays, such as optax parameters, as ordered numeric arrays."""
    flat = [np.asarray(value) for group in groups for value in group]
    np.savez_compressed(path, **{str(i): value for i, value in enumerate(flat)})


def load_arrays(path, template):
    """Load arrays saved by `save_arrays` into the structure and dtypes of `template`."""
    import jax.numpy as jnp
    expected = [value for group in template for value in group]
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {str(i) for i in range(len(expected))}:
            raise ValueError("The checkpoint has a different parameter count.")
        flat = [saved[str(i)] for i in range(len(expected))]
    if any(a.shape != v.shape or a.dtype != v.dtype for a, v in zip(flat, expected, strict=True)):
        raise ValueError("The checkpoint has incompatible parameter shapes or dtypes.")
    result, start = [], 0
    for group in template:
        result.append(tuple(jnp.asarray(a) for a in flat[start:start + len(group)]))
        start += len(group)
    return tuple(result)


class Progress:
    """A progress record in `<folder>/status.json`, read by anyone watching a sweep."""

    def __init__(self, folder, example, run, steps, device="cpu"):
        self.path = Path(folder) / "status.json"
        self.record = dict(example=example, run=run, status="queued", device=device,
                           step=0, steps=steps, validation=None, seconds=None)
        self.update()

    def update(self, **values):
        self.record.update(values, updated=datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"))
        if isinstance(self.record.get("validation"), float):
            self.record["validation"] = float(f"{self.record['validation']:.6g}")
        if isinstance(self.record.get("seconds"), float):
            self.record["seconds"] = round(self.record["seconds"], 1)
        write(self.path, self.record)


def select(folders):
    """Rank fits by the validation objective in their `config.json`; the first is selected."""
    ranking = sorted((dict(run=Path(folder).name, validation=read(Path(folder) / "config.json")["best_validation"])
                      for folder in folders), key=lambda row: row["validation"])
    return dict(criterion="lowest validation objective, before any CMGDB computation",
                selected=ranking[0]["run"], ranking=ranking)
