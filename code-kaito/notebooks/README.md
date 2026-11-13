# Example notebooks — Section 5

- `01_sphere.ipynb` (§5.1): raw sphere states, an O(2)-invariant interval encoder, learned time-one map, and orbit decoder. CMGDB uses ten learned steps.
- `02_o3_flow.ipynb` (§5.2): raw scalar/vector states, an O(3)-invariant gain encoder, learned time-log(3) quotient map, and orbit decoder.

Use the `code-kaito/.venv` kernel and run the cells in order. The training cell loads the saved fit by default; `retrain=True` trains again. See the parent README for environment setup.

These notebooks retain their saved experiment configurations. Bernardo’s numbered workflows in `code/examples/` are separate experiments; their checkpoints and training settings are not interchangeable. The comparisons distinguish native graphs, aggregates, and reconstructed indices for specified lifts.
