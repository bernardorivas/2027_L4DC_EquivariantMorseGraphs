# An Equivariant Approach to Topological Dynamics

Code and Lean formalization for the paper *An Equivariant Approach to Topological Dynamics* (working title) by Bernardo Rivas (University of Toledo) and Kaito Iwasaki (University of Michigan), prepared for L4DC 2027. The manuscript is on Overleaf.

## The paper

For a map with a symmetry group $G$, the paper moves the computation of Morse graphs and Conley indices to the orbit space through equivariant outer approximations. Group labels on the quotient edges recover the Morse sets, their order and their stabilizers, and equivariant homology of index pairs recovers index information that the quotient loses. For learned latent dynamics, a bound on the residual modulo $G$ certifies the pullback of invariant attracting blocks through an equivariant encoder. Four examples compare explicit and learned models: rotations of the sphere, the Klein bottle, an O(3)-equivariant flow, and a spectral truncation of the Chafee-Infante equation. Selected set-level statements are checked in Lean 4. Agreement between an index computed from a learned model and the index of the true system is observed, not proved, unless the paper states otherwise.

## Contents

- `code/`: the Python package `latent_equivariance` and its tests. Setup and the list of modules are in `code/README.md`.
- `lean/`: the Lean 4 formalization on Mathlib v4.31.0. Layout, conventions and the weakened hypotheses are in `lean/README.md`.

## Correspondence with the paper

| Paper | Code (`code/src/latent_equivariance/`) | Lean (`lean/LatentEquivariant/`) |
|---|---|---|
| Section 3, equivariant grids and outer approximations (`prop:quotient-outer`, `prop:scc-orbits`, `lem:equivariant_outer`, `prop:lifted-order`) | `equivariant_grid.py`, `equivariant_graph.py` | `Orbit/`, `Combinatorial/` |
| Section 4, index reconstruction from the orbit space (`prop:z2_reconstruction`, `prop:sign-reconstruction`, `prop:rotational_reduction`) | `homology.py` | `Algebra/`, `Homology/` |
| Section 5, learning equivariant and latent dynamics (`thm:main_simple`, `prop:block-lattice`, `prop:equivariant_pullback`) | `certificates.py`, `learned.py` | `Dynamics/`, `Latent/` |

Appendix `app:lean` of the paper lists the statements checked in Lean.

## Quick start

Python 3.13, with CMGDB from the fork `github.com/bernardorivas/CMGDB`:

```bash
cd code
python3.13 -m venv .venv313
.venv313/bin/pip install -e ".[dev,nn]"
.venv313/bin/pip install cmgdb==1.5.3+fork.7 \
  --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.5.3%2Bfork.7
.venv313/bin/python -m pytest
```

Lean:

```bash
cd lean
lake exe cache get
lake build
```
