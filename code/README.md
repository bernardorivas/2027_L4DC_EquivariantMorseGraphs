# latent_equivariance

Equivariant outer approximations, Conley index reconstruction from the orbit space, and equivariant encoders for latent dynamics.

## Setup

CMGDB is the fork `github.com/bernardorivas/CMGDB`, whose release wheels cover CPython 3.11-3.13 on macOS arm64 and manylinux x86_64.

```bash
python3.13 -m venv .venv313
.venv313/bin/pip install -e ".[dev,nn]"
.venv313/bin/pip install cmgdb==1.5.3+fork.7 \
  --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.5.3%2Bfork.7
```

The extra `nn` installs PyTorch, which `learned.py` needs. `requirements-py313.lock.txt` records the exact versions in use. On other platforms, build CMGDB from source as described in the fork's README (a C++17 compiler and Boost 1.66 or later).

## Contents

- `src/latent_equivariance/equivariant_grid.py`: equivariant uniform grids and float64 box images on cell representatives.
- `src/latent_equivariance/equivariant_graph.py`: quotient digraphs with voltages, recurrent components, and the full lifted order.
- `src/latent_equivariance/homology.py`: relative cubical homology, signed actions, orbit and twisted complexes, index maps and reconstruction.
- `src/latent_equivariance/certificates.py`: geometric block families and sampled residual checks on nonperiodic two-dimensional grids.
- `src/latent_equivariance/learned.py`: group-averaged networks and float64 box enclosures.
- `examples/`: dynamics, saved numerical results, figure reproduction, and interfaces for equivariant learning. See [the example commands](examples/README.md) and [the EMLP handoff](examples/emlp_handoff.md).

## Tests

```bash
.venv313/bin/python -m pytest
```

The homology tests that need CMGDB are skipped when it is not installed; `tests/test_learned.py` needs PyTorch. `python tests/test_homology.py [OUTPUT]` also writes the expected and computed value of every homology check to a JSON file.
