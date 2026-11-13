# Example 3: Reduced attitude control

This is the physical system in Section 5.3. Run commands from `code/` after the [shared setup](../../README.md#setup). The available workflows are reference quotient calculations and full-plant data generation. A trained full-plant encoder, decoder, and latent map are not bundled.

## Reference calculation

```bash
python examples/example3/run.py replay --out outputs/example3-reference
```

This computes the two thrust-axis quotient graphs at zero angular-velocity error with native CMGDB, using $2^7$ interval cells and exact rational endpoint enclosures at time $\log2$. It saves JSON records, cell CSVs, and native DOT graphs. The nominal order is `S -> N`. The coning order is `S -> C`, `D -> C`, and `D -> N`, with heights `C=1/4` and `D=7/8`. The command checks these orders and their native quotient annotations. The original-space degree lists in the JSON records are analytic reference indices.

This is a calculation on the thrust-axis subsystem after restricting to zero angular-velocity error. The full plant also retains translation and heading, so its quotient is larger.

## Full-plant data

```bash
python examples/example3/run.py data --policy nominal --count 1024 --seed 2 --out outputs/example3-nominal-data
python examples/example3/run.py data --policy coning --count 1024 --seed 3 --out outputs/example3-coning-data
```

Each file has independent trajectories as its first axis and observation times as its second. The default has 33 observations at spacing `0.1` seconds. The simulator uses RK4 with step `0.001` seconds. Initial position, velocity, and body rates are uniform in balls of radii 1, 0.5, and 6, and attitude is Haar distributed. These initial conditions include angular-velocity transients.

`trajectories.npz` contains `states`, `observations`, and `times`. Physical states are `(p[3], velocity[3], quaternion[4], Omega_body[3])`, with scalar-first unit quaternions. Observations are `(p, velocity, R_column_1, R_column_2, R_column_3, Omega_body)` in dimension 18. The accompanying JSON records the settings. Split by trajectory before forming training pairs or rollout windows.

The physical simulator and EMLP representation interfaces are in [common/systems.py](../common/systems.py). Under a vertical world rotation the first five observation vectors rotate, and the three body-rate components are invariant scalars. Full-plant learning and its comparison with the reference graphs remain separate work.
