# An Equivariant Approach to Topological Dynamics

Code and Lean 4 formalization for the manuscript by Bernardo Rivas and Kaito Iwasaki, in preparation.

Start with the [Python setup](code/README.md#setup), then the [numbered examples](code/examples/README.md). Examples 1 and 2 include exact data, saved joint-learning checkpoints, checked replay, and fresh training. Example 2 uses native `CMGDB.BoxMapData` on encoded physical sources, compares three sampling densities, and computes numerical regions of attraction and physical preimages on a fixed $64\times64$ latent grid. Empty cells are not interpolated. Example 3 provides reference calculations and full-plant data generation.

- [Example 1](code/examples/example1/README.md): SO(2) flow on the sphere.
- [Example 2](code/examples/example2/README.md): O(3) flow.
- [Example 3](code/examples/example3/README.md): reduced attitude control.
- [Joining the project](ONBOARDING.md): manuscript reading order and interpretation of results.
- [Lean formalization](lean/README.md): formal statements and build commands.
