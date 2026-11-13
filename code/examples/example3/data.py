"""Generate and validate the fixed full-plant trajectory splits."""

from pathlib import Path

import numpy as np
from common import systems
from common.workflow import digest, write

SPLITS = ("train", "validation", "test")
SEEDS = (2026111111, 2026111112, 2026111113)


def generate(folder, policies, sizes=(16384, 2048, 4096)):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for policy in policies:
        for split, count, seed in zip(SPLITS, sizes, SEEDS, strict=True):
            path = folder / f"{policy}-{split}.npz"
            if path.exists():
                raise FileExistsError(path)
            initial = systems.sample_states("attitude", count, seed)
            batches = []
            for start in range(0, count, 1024):
                states = systems.trajectories("attitude", initial[start:start+1024], intervals=32,
                                              spacing=.1, policy=policy, dt=.001)
                batches.append(systems.observe("attitude", states))
            observations = np.concatenate(batches)
            np.savez_compressed(path, observations=observations, initial_state=initial, times=np.arange(33)*.1)
            metadata[path.name] = dict(policy=policy, split=split, seed=seed, trajectories=count,
                shape=list(observations.shape), sha256=digest(path), spacing=.1, integration_step=.001,
                coordinates="p, velocity, R_column_1, R_column_2, R_column_3, Omega_body",
                split_rule="independent trajectories; split before choosing rollout windows")
            print(path, observations.shape, flush=True)
    write(folder / "metadata.json", metadata)


def load(folder, policy, split):
    path = Path(folder) / f"{policy}-{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        required = {"observations", "initial_state", "times"}
        if not required.issubset(data.files):
            raise ValueError(f"Wrong Example 3 dataset format: {path.name}")
        observations, initial, times = [data[name] for name in ("observations", "initial_state", "times")]
    if observations.ndim != 3 or observations.shape[1:] != (33, 18) or len(observations) < 8:
        raise ValueError("Example 3 observations must have shape (count, 33, 18), with count >= 8.")
    if initial.shape != (len(observations), 13) or times.shape != (33,) or not np.allclose(times, np.arange(33)*.1):
        raise ValueError("The physical initial states or observation times have incompatible shapes.")
    if observations.dtype != np.float64 or not np.isfinite(observations).all() or not np.isfinite(initial).all():
        raise ValueError("Use finite float64 physical observation data.")
    return observations, initial, times
