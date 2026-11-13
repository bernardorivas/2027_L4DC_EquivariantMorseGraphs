# An Equivariant Approach to Topological Dynamics

Code and Lean 4 formalization for the manuscript by Bernardo Rivas and Kaito Iwasaki, in preparation.

Start with the [Python setup](code/README.md#setup), then the [numbered examples](code/examples/README.md). All three examples include fixed inputs, saved learned models, checked replay, and training commands. Example 2 uses native `CMGDB.BoxMapData` on encoded physical sources at a fixed $64\times64$ latent grid, with empty cells left unavailable. Example 3 constructs upright and periodic coning controllers and learns task-aware sphere and interval models on the invariant sphere subsystem, with three seeds per controller and native CMGDB on 128 interval cells.

The [Example 3 presentation](code/examples/example3/README.md) pairs original and learned sphere preimages, interval cells, aggregate quotient graphs, and reconstructed sphere indices. Its presentation command uses saved inputs and exports PDF and SVG panels. The earlier full-state reconstruction experiment remains available separately.

[Read the two-page Example 3 presentation and view its SVG panels](code/examples/example3/presentation/README.md).

- [Example 1](code/examples/example1/README.md): SO(2) flow on the sphere.
- [Example 2](code/examples/example2/README.md): O(3) flow.
- [Example 3](code/examples/example3/README.md): reduced attitude control.
- [Joining the project](ONBOARDING.md): manuscript reading order and interpretation of results.
- [Lean formalization](lean/README.md): formal statements and build commands.
