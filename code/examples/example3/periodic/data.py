"""Full-plant samples on the closed invariant thrust-axis subsystem."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
from common import systems
from common.workflow import digest, write
from example3.periodic import controller

SPLITS = ("train", "validation", "test")
SEEDS = (2026111311, 2026111312, 2026111323)


def generate_split(folder, policy, split, count, seed):
    initial = controller.sample(count, seed, policy)
    batches = []
    for start in range(0, count, 1024):
        state = initial[start:start+1024].copy()
        observed = [systems.observe("attitude", state)]
        for _ in range(32):
            state = controller.step(state, policy=policy)
            observed.append(systems.observe("attitude", state))
        batches.append(np.stack(observed, axis=1))
    observations = np.concatenate(batches)
    path = Path(folder)/f"{policy}-{split}.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, observations=observations, initial_state=initial, times=np.arange(33)*.1)
    print(f"data {policy} {split}: {observations.shape}", flush=True)
    return dict(policy=policy, split=split, seed=seed, trajectories=count,
        shape=list(observations.shape), sha256=digest(path), spacing=.1, integration_step=.002,
        initialization="eta=0 with independent translation and heading; quarter near poles",
        observable_target="thrust axis, columns 12:15 of full observation",
        split_rule="independent complete trajectories")


def generate(folder, sizes=(8192, 2048, 4096), policies=("nominal", "coning")):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for policy in policies:
        for split, count, seed in zip(SPLITS, sizes, SEEDS, strict=True):
            metadata[f"{policy}-{split}.npz"] = generate_split(folder, policy, split, count, seed)
            write(folder/"metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", nargs=3, type=int, default=[8192, 2048, 4096])
    args = parser.parse_args()
    generate(args.out, args.sizes)
