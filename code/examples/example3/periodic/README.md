# Autonomous coning stabilization

This separate variant of Example 3 constructs a controller whose thrust axis approaches the periodic latitude `q3=1/4`. Both poles repel on the invariant sphere, so its axis order is `S -> C <- N`. The nominal comparison stabilizes the upright pole. The feedback is a constructed benchmark and is not attributed to the controllers in the cited papers. Heading is free and translation is unregulated. Constant thrust does not maintain altitude during coning.

Run the commands from `code/` in the shared Python 3.13 environment with Graphviz. LaTeX with `latexmk` is optional and compiles the mathematical draft. The numeric report and native graphs are produced without LaTeX.

For the concise presentation and paired panels used by the reduced-attitude example:

```bash
python examples/example3/run.py periodic presentation --out outputs/example3-presentation
```

This command uses saved inputs and performs no training, trajectory generation, or new image-hull evaluation. It recomputes the indices of the structural aggregates with native CMGDB. It writes `presentation.pdf` when LaTeX is available, its source, and bare PDF and SVG panels under `attitude/` for both controllers. SVG panels contain vector geometry and editable text without embedded images. The coning comparison separates quotient annotations from reconstructed sphere indices. Eleven additional immutable presentation inputs are checked by their own manifest under `reference/presentation/`.

To replay the 53 frozen numeric inputs, regenerate the physical data, and check all six graphs and the independent audits:

```bash
python examples/example3/run.py periodic replay --out outputs/example3-periodic-replay
```

To generate data and train new models:

```bash
python examples/example3/run.py periodic data --out outputs/example3-periodic/data
python examples/example3/periodic/validation.py --out outputs/example3-periodic/controller
python examples/example3/run.py periodic train --data outputs/example3-periodic/data --out outputs/example3-periodic/trained --seeds 0 1 2
python examples/example3/periodic/audit.py --data outputs/example3-periodic/data --out outputs/example3-periodic/trained
```

Data generation and training require new output directories. To recompute diagnostics and figures from an existing local training run:

```bash
python examples/example3/run.py periodic analyze --data outputs/example3-periodic/data --out outputs/example3-periodic/trained --seeds 0 1 2
```

The final mathematical report is `outputs/example3-periodic/trained/periodic-orbit-stabilization.pdf` when LaTeX is available. It contains the controller, stability proofs, physical validation, all-seed errors, aggregate graphs, and preimages on the learned sphere. `results.pdf`, `all-aggregate-graphs.pdf`, and the raw `all-native-graphs.pdf` are always produced. `summary.json` contains the numeric results. `controller/validation.json` separates numerical checks from the analytic calculation. The [draft source](draft.tex) is independent of the main manuscript. The [controller](controller.py), [learning model](model.py), and [analysis](analysis.py) contain the implementation. Fixed checkpoints and raw graph references are in `reference/`, with checksums in [manifest.json](manifest.json). Replay requires no retraining.

## Mathematical construction

For `q=R e3`, `z=q3`, and spatial angular velocity `w=R Omega`, define `V(q)=a(z)(e3-z q)+omega(e3 cross q)`, `w_d=q cross V(q)`, and `eta=w-w_d`. The moment `M=Omega cross (J Omega)+J R^T(Dw_d(q)[w cross q]-2 eta)` implies `q_dot=V(q)+eta cross q` and `eta_dot=-2 eta`. The coning law is `a(z)=1/4-z`. On `eta=0`, the height satisfies `z_dot=(1/4-z)(1-z^2)` and azimuth advances at `omega=pi/log(2)`. The periodic latitude has period `2 log(2)` and transverse Floquet multipliers `exp(-(15/16) T)` and `exp(-2 T)` three times in the rate-augmented reduced space. The draft derives these properties and the sphere Conley indices.

## Data and learned maps

Each controller uses 8,192 training, 2,048 validation, and 4,096 test trajectories, each with 33 observations separated by 0.1 seconds. The full plant uses normalized-quaternion RK4 with step 0.002. Translation and heading vary, and initialization satisfies `eta=0`. A quarter of initial heights are near the poles. Physical stress tests with rate-error radii 0, 0.25, 2, and 6 are separate from the training data.

The encoder receives all 18 full-plant observation coordinates. It is calibrated toward the known thrust-axis observable. Its learned decoder reconstructs that observable rather than the full state. This is a task-aware experiment. It changes the earlier full-state unsupervised reconstruction objective, and it does not establish a closed sphere model for arbitrary initial rates.

The normalized encoder and axis decoder are EMLPs with two hidden layers of width 64. Scalar heads have width 24. A learned boost and azimuthal increment define a sphere-preserving, equivariant time map with fixed poles and freely learned interior fixed latitudes. A learned monotone coordinate with 33 knots defines `pi=h(q3)`. The scalar time map shares the learned latitude dynamics through `f_bar=h o m o inverse(h)`, so the lower square commutes by construction. The orbit decoder is also learned. There are 20,795 parameters in total.

The first 1,000 updates calibrate the encoder and decoders, followed by joint fitting through update 6,000. Batch size is 256. Prediction horizons are 1, 4, and 10 steps. Checkpoint and seed selection use the joint validation objective before graph computation. All three seeds per policy are retained. The initial two-policy pilot is separate and is not used as a reference result. The final test seed, 2026111323, was generated after pilot development and was not used for further tuning.

## Retained results

| Policy | Seed | One-second axis RMS | Raw components | Aggregate components | Sinks |
|---|---|---|---|---|---|
| Nominal | 0 | 0.0010055 | 2 | 2 | 1 |
| Nominal | 1 | 0.0014311 | 2 | 2 | 1 |
| Nominal | 2 | 0.0012477 | 2 | 2 | 1 |
| Coning | 0 | 0.0018308 | 4 | 3 | 1 |
| Coning | 1 | 0.0020604 | 4 | 3 | 1 |
| Coning | 2 | 0.0017331 | 4 | 3 | 1 |

Seed 0 is selected by validation for both policies. Each coning fit has one attracting interior latent latitude, between 0.24687 and 0.24710, with a learned azimuth increment near 0.45324 radians per step. The attracting latitude decoded into physical axis coordinates has height between 0.24734 and 0.24760, compared with the designed height 0.25. Each nominal fit has no interior fixed latitude. All 128 interval cells have encoded physical sample support.

The ideal coning order has three orbits. Its raw graph in the learned coordinate has four components because an additional cell near the attracting latitude has a self-loop and trivial index. The matched physical reference in that coordinate also has four components. The raw physical-height reference has three. The displayed aggregate combines the attracting component, the adjacent trivial-index component, and all cells on their connecting paths. Order convexity is checked, and `CMGDB.ComputeConleyIndexForCells` checks the merged index pair and recomputes its index as `(x-1,0)`. The aggregate order is `S -> C <- N`, and its singleton sink labels cover 126 of 128 cells. Raw graph records remain available under `cmgdb/`, and aggregate records are under `cmgdb/aggregate/`. The analytic on-manifold basin of the coning orbit is all nonpolar axes.

## CMGDB and evidence

Native CMGDB analyzes ten composed learned steps, or one second, on 128 interval cells. Regular callbacks sample 257 points per cell with padding `2e-6`. The physical scalar flow is compared at the same time, padding, grid, and learned quotient coordinate. Every native component, including those with trivial annotations, is retained. Graphs and interval cells use native CMGDB plotting. The separate audit checks SCCs and sink labels independently and tests the image hulls at 4,097 points per cell.

The physical controller and its sphere indices have analytic proofs. Learned prediction and graph agreement are numerical observations. Native interval annotations are not sphere indices. Sampled image hulls and derivative checks are not validated enclosures, and no physical transfer certificate is claimed. Vector normalization is checked on held-out observations and sampled latent points. Global avoidance of zero raw encoder and decoder outputs is not proved.
