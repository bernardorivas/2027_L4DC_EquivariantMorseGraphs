"""Ordered numeric checkpoints for the pinned Objax model definitions."""

import jax.numpy as jnp
import numpy as np
import objax


def save_weights(path, variables):
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
