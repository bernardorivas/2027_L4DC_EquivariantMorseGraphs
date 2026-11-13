# Joining the project

The manuscript, An Equivariant Approach to Topological Dynamics, is on [Overleaf](https://www.overleaf.com/project/6ad128ca1e6dd6d3000000ad). Its `main.tex` is the reference for definitions, hypotheses, and claims. This repository contains the Python code and Lean 4 formalization. The guide follows the three current examples as of November 12, 2026.

## Run the code first

Follow the [Python setup](code/README.md#setup), run the environment check, and start with:

```bash
python examples/example1/run.py replay --out outputs/example1-replay
```

Run this from `code/`. It loads the exact data and saved weights for the joint interval experiment, recomputes diagnostics and native CMGDB results, verifies them against the reference, and writes a comparison PDF. The [example index](code/examples/README.md) lists all available commands.

| Folder | Physical system | Available work |
|---|---|---|
| [example1](code/examples/example1/README.md) | SO(2) flow on the sphere | Checked checkpoint replay, joint training, and separate reproduction of the older manuscript panels |
| [example2](code/examples/example2/README.md) | O(3) flow on a subset of R4 | Joint learning and checked BoxMapData replay at 64 x 64, physical sampling comparisons, numerical basins and physical preimages, and data generation |
| [example3](code/examples/example3/README.md) | Reduced attitude control | Reference thrust-axis quotient calculations and full-plant data generation |

The READMEs identify the actual dataset, representations, learned maps, and time steps. Each workflow writes to a new directory in `code/outputs/`. Input data and frozen references have separate locations. Fresh training results can differ from saved checkpoints, so compare both numerical errors and actual native graphs.

## Read the manuscript

1. Read the Introduction and Combinatorial dynamics and latent dynamics for the problem, outer approximations, Morse graphs, and Conley indices. The flow examples use time maps.
2. Read Equivariant outer approximations and Equivariant approximate semi-conjugacy for the reduced computations and the hypotheses for pulling back attracting blocks.
3. Read Examples together with CMGDB computations and reconstruction and Numerical details in the appendices. Distinguish the manuscript's saved models from the separately identified joint interval experiment in Example 1.
4. Use the [Lean guide](lean/README.md) for the formalized statements, conventions, and remaining geometric identifications.

## Interpretation and implementation

The method reduces to an orbit space or a fundamental domain, runs native CMGDB there, and reconstructs information about the original dynamics. Keep the source of each result explicit. A native quotient annotation, a cellular calculation, and an analytic original-space index calculation have different justifications.

Matching learned and explicit quotient graphs is numerical evidence. A sampled residual is not a uniform residual bound, and sampled learned image hulls are not validated enclosures. Transfer of attracting blocks and identification of Conley indices require the manuscript's separate hypotheses and arguments.

The package in `code/src/latent_equivariance/` contains grids, quotient graphs and group labels, homology, and sampled block checks. The numbered examples use the shared simulators, exact enclosures, and figure helpers in `code/examples/common/`. Start with a numbered `run.py` for computations and its README for the data contract.

Run the Python checks as described in [code/README.md](code/README.md#library-and-tests). For Lean, run `lake exe cache get` and `lake build` from `lean/`. A successful numerical run or formal build does not establish the mathematical hypotheses left outside those checks.
