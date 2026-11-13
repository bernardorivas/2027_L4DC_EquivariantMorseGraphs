# Example 3: Reduced attitude control

Construct a nominal upright controller and an autonomous coning controller, learn an equivariant sphere model and its interval quotient, and compute native Conley--Morse graphs on 128 interval cells. The encoder is calibrated toward the thrust axis using full-plant trajectories on the invariant sphere subsystem. Three learned seeds per controller and their checked numerical references are supplied.

[Read the concise presentation and view the SVG panels](https://github.com/bernardorivas/latent-equivariant/tree/main/code/examples/example3/presentation).

Use the [shared Python setup](../../README.md#setup) and run these commands from `code/`:

```bash
python examples/example3/run.py periodic presentation --out outputs/example3-presentation
python examples/example3/run.py periodic replay --out outputs/example3-replay
```

The first command renders the concise presentation and paired original/learned PDF and SVG panels from saved inputs. It requires no training or trajectory generation. Structural aggregates include connecting cells and have their indices recomputed by CMGDB. LaTeX with `latexmk` compiles `presentation.pdf` when available. The second command regenerates the physical data, replays all six frozen models, and checks raw and aggregate graphs, predictions, and independent audits.

The [periodic-control instructions](periodic/README.md) contain the controller, data and training settings, all-seed results, and fresh-training commands. The [earlier full-state reconstruction experiment](full-state.md) and all its data, checkpoints, and observed failures remain available through the original `replay`, `train`, and `data` commands.
