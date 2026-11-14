# An Equivariant Approach to Topological Dynamics

Code for the paper *An Equivariant Approach to Topological Dynamics* by Bernardo Rivas, Kaito Iwasaki and Ram Vasudevan, submitted to L4DC 2027.

For a map that commutes with the action of a compact group, the paper computes Morse graphs and Conley indices on the orbit space with CMGDB, learns the reduced dynamics with equivariant networks (EMLP), pulls attracting blocks back through the learned encoder, and reconstructs the Conley indices in the original space. The paper notebooks below introduce each example and explain its equations, learned model, and figures.

## Start with the paper notebooks

For a guided reading of the paper's examples, start with these notebooks. They include explanations and saved figures, so you can read them directly on GitHub without installing Python.

| In the paper | Paper notebook |
|---|---|
| Section 5.1, an SO(2)-equivariant flow on $S^2$ | [01_sphere.ipynb](paper-notebooks/notebooks/01_sphere.ipynb) |
| Section 5.2, an O(3)-equivariant flow on $\mathbb R^4$ | [02_o3_flow.ipynb](paper-notebooks/notebooks/02_o3_flow.ipynb) |
| Section 5.3, reduced attitude control on $S^2$ | [03_attitude.ipynb](paper-notebooks/notebooks/03_attitude.ipynb) |

See [paper notebook setup and reproduction](paper-notebooks/README.md) before running them. Examples 1 and 2 use bundled, pinned experiments. Example 3 currently requires the legacy source checkout specified in its setup cell. These requirements differ from the root-level workflows below.

## Implementation workflows

The root-level [ex1_so2_flow.ipynb](ex1_so2_flow.ipynb), [ex2_o3_flow.ipynb](ex2_o3_flow.ipynb), and [ex3_attitude_control.ipynb](ex3_attitude_control.ipynb) provide the workflows for the current `morsegraphs/` package. Use them to run that implementation or retrain its models. Each workflow identifies its own data and checkpoints; they are not interchangeable with the pinned paper-notebook experiments.

For Section 6 and the timing table, use [timings.ipynb](timings.ipynb).

The following installation and execution instructions apply to these root-level workflows.

## Install

Python 3.13, Graphviz (`dot`) and Poppler (`pdftoppm`) are needed. CMGDB 1.5.3+fork.7 is the version of CMGDB used in the paper; its wheels are built for macOS arm64 and Linux x86_64.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks,test]" \
  --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.5.3%2Bfork.7
python -m pytest
jupyter lab
```

## Run a notebook

The first code cell of each notebook chooses the mode. `MODE = "replay"` loads the fit selected by validation from `data/<example>/models/`, recomputes every result, and checks it against `data/<example>/reference/results.json`. `MODE = "train"` retrains that fit from its saved settings and runs the same steps. Results and panels, as PDF and SVG, go to a new folder under `outputs/notebooks/`. On a laptop CPU a replay takes one to a few minutes. To run one without a browser:

```bash
jupyter nbconvert --to notebook --execute --output-dir outputs ex1_so2_flow.ipynb
```

## Reproduce the fits

The scripts regenerate the data, train five seeds of an example in parallel processes, and select one by validation error before any CMGDB computation:

```bash
python scripts/generate_data.py ex1_so2_flow
python scripts/train_seeds.py ex1_so2_flow --seeds 0 1 2 3 4
python scripts/select_seed.py ex1_so2_flow
python scripts/record_reference.py ex1_so2_flow --replace
```

`data/<example>/` holds the data with SHA-256 checksums, the settings (`config.json`), every trained fit, the selection and the reference results. Training runs on the CPU, or on a CUDA GPU when JAX has one; EMLP does not run on Apple GPUs. On the reference machine (Apple M4 Pro) training takes about two minutes per seed for Sections 5.1 and 5.3 and thirty minutes for Section 5.2.

## What is computed, observed and proved

- The Conley indices in the original space are proved in the paper on explicit index pairs. The notebooks also compute them from the quotient with the reconstruction formula of the paper's appendix, for the reference maps and for the learned maps.
- The reference quotient maps of Sections 5.1 and 5.2 are enclosed by validated binary64 bounds. The learned maps are analyzed through sampled image hulls, which are not validated enclosures, so their Morse graphs are numerical evidence.
- A sampled residual below an estimated tolerance is a sampled check of the hypothesis under which the paper pulls back attracting blocks, not a uniform bound.

## Layout

- `morsegraphs/`: the package. `grids.py` (grids, cell images, CMGDB, merged Morse sets), `regions.py` (attracting regions, tolerances, residuals), `homology.py` (Conley indices in the original space), `timing.py` (CMGDB timings), `plotting.py`, `training.py`, `device.py`, and one module per example: `so2_flow.py`, `o3_flow.py`, `attitude_control.py`.
- `scripts/`: data, training, selection, reference records, and the timing runs (`run_timings.py`, which refuses to start on a busy machine).
- `data/`: data, fits and reference results of each example.
- `tests/`: `python -m pytest`.
- `paper-notebooks/`: the guided paper notebooks for Sections 5.1–5.3, plotting helpers, and pinned reproduction material. Start here to read the examples.

## Citation

```bibtex
@unpublished{rivas:iwasaki:vasudevan:26,
  author = {Rivas, Bernardo and Iwasaki, Kaito and Vasudevan, Ram},
  title  = {An Equivariant Approach to Topological Dynamics},
  note   = {Submitted to L4DC 2027},
  year   = {2026}
}
```
