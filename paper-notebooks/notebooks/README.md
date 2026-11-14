# Paper notebooks

Read these notebooks from top to bottom. Each introduces the physical system,
defines the learned maps, and explains how to interpret its figures. Saved
outputs are included.

- [Sphere flow — Section 5.1](01_sphere.ipynb)
- [O(3) flow — Section 5.2](02_o3_flow.ipynb)
- [Reduced attitude control — Section 5.3](03_attitude.ipynb)

For Examples 1 and 2, follow the [environment setup](../README.md#run-locally),
select `paper-notebooks/.venv`, and run the cells in order. Saved checkpoints
are loaded by default.

For Example 3, run `sh setup_attitude.sh` from `paper-notebooks/` to install the
pinned legacy source and replay environment. The notebook evaluates six saved fits without retraining and
compares ungrouped and grouped interval graphs, latent-sphere preimages, and
analytical thrust-axis indices. Its final cell runs the original replay check.

The [root-level notebooks](../../README.md#implementation-workflows) run the
current `morsegraphs/` implementation. Their settings and checkpoints should
not be substituted for the pinned experiments in these paper notebooks.
