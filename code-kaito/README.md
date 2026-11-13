# Interactive examples

Open the executed notebooks to inspect the equations, training losses, learned
maps, and analytical-versus-learned CMGDB comparisons:

- [01_sphere.ipynb](notebooks/01_sphere.ipynb): sphere to an interval, with joint
  O(2)-invariant EMLP encoder, interval map, and orbit decoder.
- [02_o3_flow.ipynb](notebooks/02_o3_flow.ipynb): O(3) quotient dynamics, with a
  joint invariant EMLP encoder, quotient map, and orbit decoder.

These are Kaito's saved experiments and figure assemblies. They are separate
from the current numbered workflows in [`../code/examples`](../code/examples).
The analytical replay code is pinned to Bernardo's commit `21acca6`; its source
and numerical inputs are included under `vendor/`, with checksums in
`upstream.json`. The two learned checkpoints and associated numerical evidence
are pinned in `reproducibility/manifest.json`.

## Run locally

The notebooks use a Python 3.13 environment for CMGDB and plotting, and a separate
Python 3.11 environment for the official EMLP/Objax implementation. Graphviz,
Poppler and librsvg must be available on your PATH. On macOS:

```bash
brew install graphviz poppler librsvg
cd code-kaito
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sh setup_fitting.sh
sh launch_jupyter.sh
```

The CMGDB requirement is Bernardo's `1.5.3+fork.7` release, including the Atlas,
aggregation, and relative-homology interfaces used here. Its wheel must support
your platform; this package was checked on Apple Silicon macOS. EMLP is 1.0.3,
Objax is 1.8.0, and JAX is 0.4.23; see `requirements-emlp.txt`.

Choose **Restart Kernel and Run All**. The first setup cell restores the bundled
checkpoints and numerical evidence into ignored `build/`. It checks the archive
and pinned source hashes. Default execution reuses the saved fits and regenerates
the notebook outputs; no training is required just to inspect the comparisons.

For a command-line replay:

```bash
.venv/bin/python tools/execute_notebook.py 01_sphere.ipynb
.venv/bin/python tools/execute_notebook.py 02_o3_flow.ipynb
```

## Train and interpret

Each notebook exposes one `fit_quotient(...)` call. Set `retrain=True` to train
again; sample counts, seed and time interval are explicit. EMLP/Objax provides
the network layers, differentiation and optimizer. `src/quotient_fitting.py`
contains the joint reconstruction, prediction and latent-consistency losses.
Training and validation curves update in the notebook.

The sphere experiment uses 4,096 uniformly sampled training states, a time-one
training map and ten learned iterations for CMGDB. The O(3) experiment uses
16,384 training states and time `log(3)`. Neither training set repeats special
points. Orbit reconstruction compares states modulo the specified symmetry.

The output heads are part of the model specification: sphere coordinates are
calibrated at the poles and the interval map preserves its endpoints. The O(3)
gain parameterization uses the known invariants `(a, ||v||)` with learned positive
corrections. The notebook states this prior explicitly.

Saved local index checks apply to the bundled checkpoints and specified lifts.
They must be rerun for a different trained model; the supplied O(3) aggregation
and figure assembly are tied to its saved checkpoint and reject a mismatch.
Changing the training configuration is supported, but does not automatically
regenerate checkpoint-specific local certificates or all O(3) comparison panels.

Sampled cell images, basin labels and graph agreement are numerical evidence.
The local interval checks concern specified lifts of the learned maps; they do
not certify a conjugacy to the physical ODE. The notebooks keep these claims
separate. See the restored `build/*-certification/local-index-argument.md` files
for the reconstruction assumptions and bounds.

All generated outputs go to `build/` or `artifacts/`. The public notebook cells
contain the systems, encoder/map/decoder fitting, and CMGDB comparison; detailed
plotting and verification routines live in `src/`.
