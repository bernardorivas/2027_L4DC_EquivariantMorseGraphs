"""Check EMLP training and a CMGDB computation in one Python process."""

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from importlib.metadata import version
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import objax
from emlp.groups import SO
from emlp.nn.objax import EMLP
from emlp.reps import T, V
import CMGDB


def check():
    np.random.seed(0)
    model = EMLP(T(0), T(0), group=SO(2), ch=16, num_layers=1)
    x = jnp.linspace(-1, 1, 32).reshape(-1, 1)

    def loss():
        return jnp.mean((model(x) - x/4)**2)

    gradients = objax.GradValues(loss, model.vars())
    optimizer = objax.optimizer.Adam(model.vars())
    initial = float(loss())
    for _ in range(30):
        values, _ = gradients()
        assert all(np.isfinite(np.asarray(value)).all() for value in values)
        optimizer(0.01, values)
    final = float(loss())
    assert np.isfinite(final) and final < initial

    group = SO(2)
    inputs = T(0) + V
    invariant = EMLP(inputs, T(0), group=group, ch=16, num_layers=1)
    points = jnp.asarray(np.random.default_rng(0).normal(size=(8, 3)))
    angle = 0.731
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    action = jnp.asarray(inputs(group).rho_dense(rotation))
    defect = float(jnp.max(jnp.abs(invariant(points @ action.T) - invariant(points))))
    assert np.isfinite(defect) and defect < 1e-5

    # Exact interval images of a separate contraction check the native engine.
    grid = CMGDB.Model(4, 4, 4, 10000, [-1.], [1.],
                       lambda rectangle: [rectangle[0]/4, rectangle[1]/4])
    morse, phase = CMGDB.ComputeConleyMorseGraph(grid)
    vertices = list(morse.vertices())
    assert phase.num_vertices() == 16 and len(vertices) == 1
    annotations = list(morse.annotations(vertices[0]))
    assert annotations[0] == "x-1"
    return {"python": sys.version.split()[0],
            "jax_backend": jax.default_backend(), "jax_float64_enabled": bool(jax.config.jax_enable_x64),
            "versions": {name: version(name) for name in
                         ("emlp", "cmgdb", "jax", "jaxlib", "objax", "numpy", "scipy")},
            "emlp_initial_loss": initial, "emlp_final_loss": final,
            "emlp_sampled_invariance_error": defect,
            "cmgdb_cells": phase.num_vertices(), "cmgdb_morse_sets": len(vertices),
            "cmgdb_annotations": annotations, "passed": True}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
