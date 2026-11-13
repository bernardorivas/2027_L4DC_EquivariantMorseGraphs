# Python code

The numbered examples share one Python 3.13 environment with EMLP, native CMGDB, and the plotting dependencies. The source package also provides equivariant grids, graph lifting, and cellular homology.

## Setup

Install Python 3.13, Graphviz (`dot`), and Poppler (`pdftoppm` and `pdfinfo`). The reference environment is CPU. Examples 1 and 2 use JAX float64, while Example 3 retains float32 learned models; the commands configure their precision automatically. From this `code/` directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
python examples/check.py
```

[requirements.txt](requirements.txt) pins the shared environment, including EMLP 1.0.3, Objax 1.8.0, JAX/JAXlib 0.4.38, NumPy 2.3.5, SciPy 1.15.3, and CMGDB 1.5.3+fork.7. It includes the link to the CMGDB fork release wheels. The checked reference platform is macOS arm64 with Python 3.13.18. The CMGDB release also supplies CPython 3.13 manylinux x86_64 wheels; other platforms require a compatible build from the fork.

The environment check exercises EMLP optimization, sampled invariance, native CMGDB, physical flow composition and equivariance, representation packing, and the reduced-attitude reference map. It exits with an error if a required numerical version or plotting executable is missing. Keep the complete output when diagnosing a setup difference.

## Run an example

```bash
python examples/example1/run.py replay --out outputs/example1-replay
python examples/example1/run.py train --out outputs/example1-new-fit
```

Replay loads the bundled data and checkpoint and checks against the saved result. Training fits a new model with the supplied data and configuration. Both produce diagnostics, native graph outputs, and PDF reports. Example 2 uses the same `replay` and `train` commands and adds native `CMGDB.BoxMapData` construction from encoded physical sources, sampling-density comparisons, numerical regions of attraction, and physical inverse images. Its latent grid is fixed at $64\times64$, with empty cells left unavailable, and its report is `results.pdf`. The [example index](examples/README.md) lists the commands available for all three examples. Each README states its mathematical setup, data format, expected results, and numerical limitations.

Example 3 learns task-aware sphere and interval models of nominal upright and constructed coning control, with three seeds each. `python examples/example3/run.py periodic presentation --out outputs/example3-presentation` renders the concise presentation and paired original/learned PDF and SVG panels from saved inputs. It displays structural aggregate graphs with recomputed Conley indices, retains raw 128-cell graphs for verification, and distinguishes quotient annotations from reconstructed sphere indices.

The [reproduction instructions](examples/example3/periodic/README.md) include the controller and local stability proof. `python examples/example3/run.py periodic replay --out outputs/example3-periodic-replay` regenerates data and checks the six frozen fits, native graphs, and independent audits. Numeric reports require the shared environment and Graphviz. LaTeX with `latexmk` optionally compiles the mathematical presentation. The [earlier full-state reconstruction experiment](examples/example3/full-state.md) and its original runner commands are preserved.

Use a new output directory for every run. `outputs/` is ignored by Git, and supplied data and references stay unchanged. Reference inputs have SHA-256 manifests. Examples 1 and 2 keep their new joint fits and older manuscript fits in separate reference directories.

## Library and tests

- `src/latent_equivariance/equivariant_grid.py`: equivariant uniform grids and box images on cell representatives.
- `src/latent_equivariance/equivariant_graph.py`: quotient graphs, group labels, stabilizers, and lifted order.
- `src/latent_equivariance/homology.py`: relative cubical homology, signed actions, orbit and twisted complexes, index maps, and reconstruction.
- `src/latent_equivariance/certificates.py`: geometric blocks and sampled residual checks.
- `src/latent_equivariance/learned.py`: PyTorch components used by the library tests.

```bash
python -m pytest
python tests/test_homology.py homology-results.json
```

PyTorch is optional for the numbered examples. Install the `nn` extra to run its library tests with `python -m pip install -e ".[nn]"`. Check the pytest skip report. The examples use Objax EMLP.
