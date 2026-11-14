# Examples

The sphere and O(3) examples include the numerical inputs and plotting code for their twelve figure panels. The third example is reduced attitude control on the sphere. The model interfaces and the EMLP fitting workflow are described in [emlp_handoff.md](emlp_handoff.md).

Run the commands below from `code/`. Install Python dependencies and CMGDB as described in [../README.md](../README.md), then install the plotting dependencies with `.venv313/bin/pip install -e ".[examples]"`. Rendering also requires Graphviz (`dot`) and Poppler (`pdftoppm` and `pdfinfo`).

## Figures

```bash
.venv313/bin/python examples/render.py --out examples/output
```

The output has `sphere/flow/`, `sphere/morse/`, and `o3/` directories containing six sphere panels and six O(3) panels. Use `--example sphere` or `--example o3` to select one example. `layout.json` contains the printed widths used to set graph strokes.

## Computations

```bash
.venv313/bin/python examples/replay.py --out examples/runs
.venv313/bin/python examples/render.py --data examples/runs --out examples/output-replayed
```

This runs CMGDB on the supplied image enclosures and checks the resulting cells, graph edges, and quotient index annotations against the supplied results. For O(3), it also recomputes the single-sink regions, directed-path cells, one-step image cover, and the index of the grouped saddle region. The explicit O(3) grid has five native components, and its displayed graph groups the empty component with the saddle. The fitted grid has four native components.

To recompute the explicit image enclosures before running CMGDB:

```bash
.venv313/bin/python examples/replay.py --recompute-explicit --out examples/runs-explicit
```

The sphere enclosures use the separated height equation with rational bounds for its exponential factor. The O(3) enclosures use exact coordinate extrema and rational square comparisons. The command checks that these image enclosures agree bit for bit with the supplied arrays. The learned columns use the supplied fitted image enclosures. Fitting new models and constructing their enclosures are separate steps in [emlp_handoff.md](emlp_handoff.md).

CMGDB quotient annotations use coefficients in the field with five elements. The O(3) original-space graph uses the analytic index formulas. The renderer checks the quotient graph against the expected order and annotations before drawing it.

## Files

| File or directory | Contents |
|---|---|
| `systems.py` | Dynamics, state sampling, observations, and symmetry actions for the three examples |
| `render.py` | Figure generation from supplied or replayed results |
| `replay.py` | Native CMGDB computations and comparisons |
| `exact_enclosures.py` | Explicit sphere and O(3) image enclosures |
| `data/sphere/` | The 128-cell explicit and fitted height-quotient results, vertex images, and callback intervals |
| `data/o3/` | The matched 128 by 128 quotient grids, box images, Morse graphs, and displayed regions |
| `data/manifest.json` | SHA-256 checksums of the supplied numerical inputs |

The scripts resolve input paths relative to their own directory and accept `--data` for another compatible result directory. The data manifest is checked before plotting or replaying. The supplied data directory is read-only input to both commands. Replay requires a new output directory and CMGDB `1.5.3+fork.7`.
