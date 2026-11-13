# Worked examples of learned quotient dynamics

- [Sphere flow](01_sphere.ipynb) explains the height quotient, learned interval dynamics, and circle-fiber contribution to local indices.
- [O(3) flow](02_o3_flow.ipynb) explains positive-gain coordinates, two-dimensional quotient dynamics, and sphere-fiber contributions.

Read either notebook from top to bottom. Each introduces the system, derives a concrete quotient example, specifies the model and training loss, and explains how to read the figures and graph annotations. Saved outputs are included.

To reproduce the outputs, follow the [environment setup](../README.md#run-locally), select `code-kaito/.venv`, and run every cell. The default loads the bundled checkpoint without training. The numerical settings are explicit in the fitting cell. Figure exports retain PDF or SVG sources alongside the PNG notebook outputs.

The introductory figures are constructed in the notebook cells. The sphere example evaluates and projects the vector field, then integrates the height ODE with SciPy. The O(3) example evaluates its closed-form quotient flow and draws the phase portrait. Each cell saves PNG, PDF, and SVG files under `build/notebooks/`.
