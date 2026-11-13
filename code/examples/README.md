# Examples

The numbered folders follow the order of the three examples in the manuscript. Start with the [installation and environment check](../README.md#setup). Run every command below from `code/` with that environment activated. Each command requires a new output directory under `outputs/`.

| Folder | Model | Available workflows |
|---|---|---|
| [example1](example1/README.md) | SO(2) flow on the sphere | Replay the joint interval fit; train a new joint fit; reproduce the older manuscript panels |
| [example2](example2/README.md) | O(3) flow on a subset of R4 | Replay or train the joint quotient model; compute BoxMapData graphs and numerical basins at 64 x 64; generate data |
| [example3](example3/README.md) | Reduced attitude control | Constructed upright and coning controllers, task-aware joint sphere and interval models, three seeds per policy, 128-cell CMGDB, concise presentation, and checked replay |

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
python examples/example3/run.py periodic presentation --out outputs/example3-presentation
python examples/example3/run.py periodic replay --out outputs/example3-replay
```

Example 2 uses the joint quotient fit on a fixed $64\times64$ latent grid. Native `CMGDB.BoxMapData` uses only pairs from encoded physical sources, comparing three sampling densities without interpolating empty cells. `results.pdf` includes the data-map graphs, numerical regions of attraction, and physical inverse images. Example 3 supplies a nominal upright controller and an attracting coning orbit, three task-aware learned models per controller, and regular CMGDB on 128 interval cells. Its PDF and SVG presentation panels pair original and learned sphere preimages, interval cells, aggregate quotient graphs, and reconstructed sphere indices. Each README documents reproduction and training commands and the actual numerical outcomes.

Example 3 uses full-plant trajectories initialized on the closed invariant sphere subsystem, with explicit calibration toward the thrust axis. The [earlier arbitrary-rate reconstruction experiment](example3/full-state.md) is preserved separately.

## Attracting-region tolerances

The [shared tolerance workflow](common/attracting_regions.md) accepts a checkpoint directory and recomputes its numerical attracting-region margins and physical residuals. Use `example1/run.py tolerances`, `example2/run.py tolerances`, or `example3/run.py periodic tolerances`, each with `--model MODEL --out OUTPUT`. The reference READMEs retain the current values and settings. New weights are evaluated through new native graphs and basin labels. Failed comparisons remain in the output.

## Files

Each example has its own README, `run.py`, editable settings in `config.json`, fixed numerical datasets in `data/`, and immutable saved experiments in `reference/`. Examples 1 and 2 retain separate historical manuscript inputs, and Example 2's `reference/data-map/` stores the current data-map analysis. Example 3 groups its sphere, interval, and CMGDB references by controller and seed. Input manifests record SHA-256 checksums. All new results go to the requested output directory, which is ignored by Git.

`common/` contains the physical simulators, exact reference enclosures, native replay helpers, geometry, and figure style. It is shared implementation; use the numbered entry points for normal runs. `check.py` checks the pinned environment and the physical systems.

Native quotient annotations, original-space index reconstruction, and transfer to the physical dynamics have separate justifications. Sampled learned image hulls and sampled residuals are numerical diagnostics rather than mathematical certificates.
