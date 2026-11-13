# Numerical tolerances for attracting regions

The numbered `tolerances` commands load one checkpoint, rebuild its native cell map and Morse graph, select conservative regions of attraction of its minimal nodes, and compare numerical tolerance margins with sampled physical residuals. Results are written to a new output directory. Saved graphs and basin labels are not used to analyze new weights.

Run commands from `code/` in the shared Python 3.13 environment. The model directory contains `config.json` and `weights.npz`, together with `normalization.npz` for Example 3. Use the complete checkpoint directory produced by training. The loader reads the architecture settings from its configuration, so another width or layer count in the same model family is supported.

```bash
python examples/example1/run.py tolerances --model examples/example1/reference/joint/seed1 --out outputs/example1-tolerances
python examples/example2/run.py tolerances --model examples/example2/reference/joint/seed1 --out outputs/example2-tolerances
python examples/example3/run.py periodic tolerances --policy coning --model examples/example3/periodic/reference/coning/seed0/model --out outputs/example3-coning-tolerances
```

Replace `--model` with a training output, such as `outputs/new-fit/seed1` for Examples 1 and 2 or `outputs/new-fit/coning/seed0/model` for Example 3. Model files are never updated by the calculation. Both failed comparisons and changed graphs are retained as results. The adjacent reference READMEs contain the numerical values and settings used for the working examples:

- [Example 1](../example1/reference/tolerances/README.md).
- [Example 2](../example2/reference/tolerances/README.md).
- [Example 3](../example3/periodic/reference/tolerances/README.md).

## Region, margin, and residual

For each minimal node, select cells for which it is the only reachable Morse node. This uses all Morse nodes in the reachability test, including nonminimal nodes. Exclude unavailable cells, every cell that can reach them, and cells whose image hulls can leave an artificial window boundary. The resulting compact cell union is checked for forward invariance in the finite graph.

For coning, the prescribed aggregate includes the attracting component, the adjacent trivial-index component, and their connecting cells. Order convexity and the merged index pair are checked, and native CMGDB recomputes the index. If another checkpoint does not satisfy this structure, the report states why the aggregate is unavailable and retains the raw graph. A trivial index by itself does not authorize a merge.

For a region `B`, the numerical tolerance is the distance from the union of its sampled image hulls to the complement of `B`. Genuine orbit-space boundary faces are treated relative to the orbit space. Unavailable cells and artificial window boundaries count as complement. A separate exposed-boundary-face calculation checks this distance and retains a limiting witness. The physical residual compares `f(E(x))` with `E(F(x))` at the map time used for the graph, in the same metric as the tolerance.

| Example | Physical map time | Primary metric | Additional comparison |
|---|---|---|---|
| 1 | 10, using ten learned time-one steps | Euclidean distance in the invariant interval code | Chordal metric of the sphere-height lift |
| 2 | log(3), using one learned step | Euclidean distance in the two invariant coordinates | None |
| 3 | 1, using ten learned time-0.1 steps | Chordal metric on the latent sphere after undoing the learned monotone coordinate | Euclidean distance in that scalar coordinate |

For Example 3, the sphere comparison uses the physical encoder residual. The shared scalar/sphere identity alone is not evidence of physical prediction accuracy. Generated physical trajectories start on `eta=0`, with translation and heading variation. Optional supplied data retain the scope of the supplied observations.

All image hulls and residual extrema are sampled. A positive margin and a passing sampled comparison do not establish a uniform physical attracting-block certificate. The reports always retain `certified: false`. Whole-domain regions can have infinite tolerance relative to their true boundary. This is represented explicitly and is not an informative separation test.

## Sampling and timings

`--samples` controls the base sampling size. In Example 1 it is the number of evenly spaced meridian angles, supplemented with logarithmic angles near both poles. In Example 2 it is the number of physical-section cell centers per axis, supplemented with points near the separating axis, radial boundary, and equilibrium controls. In Example 3 it is the number of held-out trajectories to generate and the number of fresh one-second trajectories. A supplied `--data` file or directory replaces held-out generation.

Examples 1 and 3 accept `--cells` and `--image-samples`. Example 2 keeps the `64 x 64` latent grid and accepts increasing `--source-sizes` of the form `2^k+1`, for example `--source-sizes 257 513 1025`. Its native `BoxMapData` is built only from encoded physical sources and their learned images, without domain padding or empty-cell interpolation. Residual samples are evaluated at actual encoded physical points.

`--repeats` controls warm repetitions of neural image or source-pair evaluation and residual evaluation. It defaults to three. `--native-repeats` defaults to five. Zero retains just the first call and labels its timing accordingly. Numerical results do not depend on these repetition counts. First-call costs and warm medians are stored separately. Native CMGDB time includes graph and quotient-index computation, with data-map queries included in Example 2. Neural image construction is separate in Examples 1 and 3. Do not add the diagnostic Example 2 grid-query time again to its native time.

## Output files

- `REPORT.md`: numerical comparisons and their scope.
- `metadata.json` and `settings.json`: checkpoint and source hashes, architecture, numerical versions, sampling, metric definitions, and map time.
- `raw-graph.json` and `raw-phase.npz`: the native graph, all components, sampled images, and cell relation.
- `graph.json` and `phase.npz`: the structural aggregate when available, safe basin labels, and unavailable-cell mask.
- `regions.json`: region cells, tolerances, limiting boundary witnesses, residual maxima, and every failed comparison.
- `residuals.npz`: encoded starts, physical prediction comparisons, and residual samples.
- `timings.json`: first and repeated computation times, with stages separated.

Example 2 creates one report per source density and stores the newly generated largest source set in `source-pairs.npz`. Example 3 stores any generated held-out observations under `data/`.

## Another architecture

The checkpoint adapters are in `attracting_region_models.py`. Keep the encoder, learned time map, physical map, and metric explicit when adding another architecture. Changing the learned one-step duration also requires changing the physical comparison duration and the number of learned iterates together.

The geometry functions in `attracting_regions.py` accept NumPy arrays and native CMGDB callbacks independently of a neural-network library. `compute_graph(boxes, images, boundary_faces=..., map_time=...)` rebuilds the graph from cell-image hulls. For a data construction, supply native `data_map=CMGDB.BoxMapData(...)` and a boolean support mask. It returns raw and aggregate records and arrays.

`compare_regions(record, arrays, codes, specifications, boundary_faces=...)` accepts the encoded physical starts and one specification per metric. A specification contains `metric`, `boxes`, `images`, `predicted`, and `encoded_image`. The metric-coordinate boxes may differ from the grid-coordinate boxes, as for the learned monotone coordinate in Example 3. Membership is always checked in the original grid coordinates. This interface permits new weights or architectures without depending on the frozen reference graphs.
