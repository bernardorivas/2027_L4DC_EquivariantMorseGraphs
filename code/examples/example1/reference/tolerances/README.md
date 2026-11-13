# Example 1 attracting-region reference

The selected joint interval checkpoint is `reference/joint/seed1`. On 128 cells, the two conservative regions contain 41 and 84 cells. At time ten, their Euclidean numerical tolerances are `0.0363731190533` and `0.0433472850194`, against maximum sampled residuals `0.0255590098426` and `0.0333389678503`. Both sampled comparisons pass.

The sphere-height lift uses the chordal quotient metric. Its corresponding tolerance/residual pairs are `(0.0390195843723, 0.0299066460199)` and `(0.0455395244620, 0.0335466922780)`. The two other checkpoint seeds fail the sampled comparison at every saved grid. [summary.json](summary.json) retains all three seeds at 64, 128, and 256 cells, both metrics, checkpoint hashes, and the measured timing stages.

From `code/`:

```bash
python examples/example1/run.py tolerances --model examples/example1/reference/joint/seed1 --cells 128 --out outputs/example1-tolerances
```

Defaults reproduce 69,537 meridian samples: 65,537 uniform angles and 2,000 logarithmic angles near each pole. Each cell image uses 65 points with padding `1e-10`. The learned map is iterated ten times, matching the rotating physical flow at time ten. The invariant encoder removes the longitudinal phase from this comparison.

For another trained checkpoint, replace `--model` with its directory containing `config.json` and `weights.npz`. `--cells 64` and `--cells 256` reproduce the other stored grids. The command reevaluates the model and rebuilds the native graph and regions. It does not load the saved graph for that seed.

The original warm native graph-and-index median at 128 cells was 0.72 ms with a precomputed image table. Image construction was 24.5 ms and residual evaluation was 0.255 s. These times exclude model loading, compilation, plotting, and physical sample generation. New runs store their own measurements.

These are sampled numerical checks. [The shared method and output format](../../../common/attracting_regions.md) specify the metrics, settings, and limitations. The separate [manifest](manifest.json) protects this reference summary without changing earlier fixed inputs.
