# Example 2 attracting-region reference

The selected checkpoint is `reference/joint/seed1`. On the fixed `64 x 64` latent grid, the negative and positive sink regions each contain 1,160 cells. With a `1025 x 1025` physical-section source grid plus four equilibrium controls, their numerical tolerances are `0.123086621853` and `0.122486552088`, against maximum sampled residuals `0.0195666330457` and `0.0180490993481`. Both sampled comparisons pass in the Euclidean metric of the two invariant coordinates, at time `log(3)`.

From `code/`:

```bash
python examples/example2/run.py tolerances --model examples/example2/reference/joint/seed1 --out outputs/example2-tolerances
```

The command regenerates the encoded physical sources and their learned images, constructs native `CMGDB.BoxMapData`, and recomputes the graph and safe basin regions. Empty cells are unavailable, with no interpolation or domain padding. The residual uses 1,152,136 physical-section points: a `1024 x 1024` center grid, logarithmic refinement near the separating axis, radial-boundary samples, and four controls. All three retained seeds pass at all three saved source densities.

For the density comparison:

```bash
python examples/example2/run.py tolerances --model examples/example2/reference/joint/seed1 --source-sizes 257 513 1025 --out outputs/example2-tolerance-densities
```

For another trained checkpoint, replace `--model` with its directory containing its matching `config.json` and `weights.npz`. The latent grid stays fixed. Reports are under `S257/`, `S513/`, and `S1025/`. [summary.json](summary.json) retains every seed, density, checkpoint hash, and timing stage from the reference calculation.

The original warm timing medians for the selected largest source set were 16.7 s for neural pair generation, 0.084 s for the data index, 0.367 s for native CMGDB including data queries, 0.082 s for tolerance geometry, and 35.9 s for the dense residual evaluation. The separate full-grid query time is diagnostic and is already included in the native time. New runs store their own measurements.

These are sampled data-map checks, conditional on continuous image coverage. [The shared method and output format](../../../common/attracting_regions.md) specify how to change sampling and model adapters. The separate [manifest](manifest.json) protects this summary without modifying earlier fixed inputs.
