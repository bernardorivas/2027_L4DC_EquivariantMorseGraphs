# An Equivariant Approach to Topological Dynamics

Code for the paper *An Equivariant Approach to Topological Dynamics* by Bernardo Rivas, Kaito Iwasaki and Ram Vasudevan, submitted to L4DC 2027.

For a map that commutes with the action of a compact group, the paper computes Morse graphs and Conley indices on the orbit space with CMGDB, learns the reduced dynamics with equivariant networks, and relates the results to the original space. This repository contains the three examples of Section 5 with their saved models, data and checked reference results.

## Paper to code

Run the commands from `code/` after the [setup](code/README.md#setup). Each command needs a new output directory.

| In the paper | Folder | Commands |
|---|---|---|
| Section 5.1, SO(2) flow on the sphere | [code/examples/example1](code/examples/example1/README.md) | `python examples/example1/run.py replay --out outputs/example1-replay` |
| Section 5.2, O(3) flow | [code/examples/example2](code/examples/example2/README.md) | `python examples/example2/run.py replay --out outputs/example2-replay` |
| Figures of Sections 5.1 and 5.2 | [code-kaito](code-kaito/README.md) | executed notebooks, with their own setup |
| Section 5.3, reduced attitude control | [code/examples/example3](code/examples/example3/README.md) | `python examples/example3/run.py periodic replay --out outputs/example3-replay` and `python examples/example3/run.py periodic presentation --out outputs/example3-presentation` |
| Residuals and tolerances of the attracting regions in Sections 5.1 to 5.3 | [code/examples/common](code/examples/common/attracting_regions.md) | `python examples/example1/run.py tolerances --model examples/example1/reference/joint/seed1 --out outputs/example1-tolerances` for Example 1. The linked page lists the commands for Examples 2 and 3. |

A replay loads the saved models and the bundled data, recomputes the prediction errors, the Morse graphs and the index annotations that CMGDB returns, and checks them against the saved reference. Example 3 regenerates its trajectories instead of loading them. The prediction errors, residuals and tolerances quoted in Section 5 are those of the model selected by validation error, which is seed 1 in Examples 1 and 2 and seed 0 for each controller in Example 3. The figures of Sections 5.1 and 5.2 are drawn from separate fits in the notebooks of `code-kaito/`. The `periodic presentation` command writes the panels of the attitude figures.

This repository does not produce the illustration of Section 2, the CMGDB timings of Section 6 with the timing table of the supplement, or the evaluation of the index reconstruction formula on the sphere example. The formula itself is `reconstructed_index` in `code/src/latent_equivariance/homology.py`.

## Quick start

```bash
cd code
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
python examples/check.py
python examples/example1/run.py replay --out outputs/example1-replay
```

On the reference machine (macOS arm64, CPU only), the environment check takes about 15 seconds and each replay takes about two minutes (110 to 150 seconds). The `train` commands of the examples refit the models and take longer. The tests run with `python -m pytest` from `code/`, as described in [code/README.md](code/README.md#library-and-tests).

## Requirements

- Python 3.13, Graphviz (`dot`) and Poppler (`pdftoppm` and `pdfinfo`).
- The packages pinned in [code/requirements.txt](code/requirements.txt), which include CMGDB 1.5.3+fork.7, the adapted version of CMGDB used in the paper. Its wheels are provided for CPython 3.13 on macOS arm64 and Linux x86_64. Other platforms need a build from the fork.
- Optionally LaTeX with `latexmk`, which compiles the Example 3 presentation and the report of the Example 3 replay. Everything else runs without it.

## What the computations show

- The Conley indices in the original space are proved in the paper, for the reference maps, on explicit index pairs. In Example 3 this concerns the thrust-axis dynamics on the sphere, not the plant. The code does not prove these indices, and the reconstructed panels of the attitude figures display them at the nodes with the same labels.
- Agreement between the Morse graph of a learned map and that of the reference map is numerical evidence. It holds for the fit selected by validation error, not for every trained fit. The image hulls of the learned maps come from sampled points and are not validated enclosures.
- A sampled residual below an estimated tolerance is a sampled check of the hypothesis under which the paper pulls back attracting blocks. It is not a uniform bound on the residual, so the resulting attracting blocks are observed, not proved. In Example 3 the check concerns trajectories with $\eta=0$ only.

## Terms

The READMEs of the examples use some names that the paper does not use.

| Here | In the paper |
|---|---|
| native component, raw component | Morse set |
| aggregate | union of Morse sets with the cells on paths between them |
| conservative region, safe basin region | the set of cells from which no path leaves the box or reaches another Morse set, and its region |
| physical residual | residual modulo the group, sampled |

## Layout

- `code/`: the Python package `latent_equivariance`, its tests, and the examples. See [code/README.md](code/README.md) and the [example index](code/examples/README.md).
- `code-kaito/`: executed notebooks for the examples, with training and CMGDB comparisons, in their own environment. See [code-kaito/README.md](code-kaito/README.md).

## Citation

```bibtex
@unpublished{rivas:iwasaki:vasudevan:26,
  author = {Rivas, Bernardo and Iwasaki, Kaito and Vasudevan, Ram},
  title  = {An Equivariant Approach to Topological Dynamics},
  note   = {Submitted to L4DC 2027},
  year   = {2026}
}
```
