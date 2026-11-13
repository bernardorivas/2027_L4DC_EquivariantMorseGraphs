# Examples

The numbered folders follow the order of the three examples in the manuscript. Start with the [installation and environment check](../README.md#setup). Run every command below from `code/` with that environment activated. Each command requires a new output directory under `outputs/`.

| Folder | Model | Available workflows |
|---|---|---|
| [example1](example1/README.md) | SO(2) flow on the sphere | Replay the joint interval fit; train a new joint fit; reproduce the older manuscript panels |
| [example2](example2/README.md) | O(3) flow on a subset of R4 | Replay or train the joint quotient model; compute BoxMapData graphs and numerical basins at 64 x 64; generate data |
| [example3](example3/README.md) | Reduced attitude control | Compute the reference thrust-axis quotient graphs; generate full-plant trajectories |

## Start with Example 1

```bash
python examples/example1/run.py replay --out outputs/example1-replay
```

This loads the supplied data and weights, recomputes the diagnostics and native CMGDB results, checks them against the frozen reference, and writes `outputs/example1-replay/sphere-emlp-comparison.pdf`. The report includes all three seeds. Seed 1 is selected by validation loss.

To train new models with the same data and settings:

```bash
python examples/example1/run.py train --out outputs/example1-new-fit
```

Replay checks an existing learned map. Training fits a new one and reports its agreement or disagreement with the supplied fit. A different graph from a new training run is a numerical result, not by itself an installation failure. The Example 1 README gives the expected metrics and graph counts for each seed.

## Other examples

```bash
python examples/example2/run.py replay --out outputs/example2-replay
python examples/example3/run.py replay --out outputs/example3-reference
```

Example 2 uses the joint quotient fit on a fixed $64\times64$ latent grid. Native `CMGDB.BoxMapData` uses only pairs from encoded physical sources, comparing three sampling densities without interpolating empty cells. `results.pdf` includes the data-map graphs, numerical regions of attraction, and physical inverse images. Its `replay`, `train`, and `data` commands are documented in its README. Example 3 includes a reference calculation for the thrust-axis subsystem at zero angular-velocity error; a full-plant learned checkpoint is not included.

## Files

Each example has its own README and `run.py`. Examples 1 and 2 also have editable training settings in `config.json`, fixed numerical datasets in `data/`, and immutable saved experiments in `reference/`. Their `reference/joint/` and `reference/manuscript/` directories identify different learned models. Example 2's `reference/data-map/` stores the data-map analysis of the joint checkpoints. Input manifests record SHA-256 checksums. All new results go to the requested output directory, which is ignored by Git.

`common/` contains the physical simulators, exact reference enclosures, native replay helpers, geometry, and figure style. It is shared implementation; use the numbered entry points for normal runs. `check.py` checks the pinned environment and the physical systems.

Native quotient annotations, original-space index reconstruction, and transfer to the physical dynamics have separate justifications. Sampled learned image hulls and sampled residuals are numerical diagnostics rather than mathematical certificates.
