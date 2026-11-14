# Learning the three examples

The physical simulators and observation maps are in [systems.py](systems.py). The saved figure data reproduce the existing computations. New EMLP fits can use the interfaces below and be compared with those reference results.

## Data

Run these commands from `code/`. Each file contains independent trajectories as its first axis, observation times as its second axis, and physical states, observations, and times under the keys `states`, `observations`, and `times`.

```bash
python examples/systems.py generate sphere examples/output/sphere-train.npz --count 16384 --seed 0 --intervals 10
python examples/systems.py generate o3 examples/output/o3-train.npz --count 16384 --seed 1 --intervals 10
python examples/systems.py generate attitude examples/output/nominal-train.npz --count 16384 --seed 2 --policy nominal
python examples/systems.py generate attitude examples/output/coning-train.npz --count 16384 --seed 3 --policy coning
```

Generate validation and test files with different seeds. Split by trajectory before forming time pairs or rollout windows. Use observation arrays as training inputs and state arrays for physical diagnostics. The default observation spacings are one for the sphere, `log(3)` for O(3), and `0.1` seconds for the quadrotor. `--spacing` changes them. The attitude defaults produce 33 observations over 3.2 seconds, using fourth-order Runge--Kutta with step 0.001 seconds.

Sphere states are `(x,y,z)` on the unit sphere. O(3) states are `(a,v1,v2,v3)` in `[-1.5,1.5]` times the radius-1.5 ball. Their observations consist of three invariant scalars followed by two polar vectors, as defined in `observe`. Sphere observations have dimension 7 and O(3) observations dimension 9. The latent physical reference coordinates are `(z,x,y)` and `(a,v1,v2,v3)`, respectively.

The sphere step includes longitudinal rotation at rate `0.6`, so its symmetry is SO(2). `--corotating` removes that rotation and permits the larger O(2) symmetry used by the saved sphere fit. A one-step fit at spacing one must be iterated ten times for comparison with the sphere Morse graph. The displayed height-map panel uses time three.

The quadrotor state is `(p[3], velocity[3], quaternion[4], Omega_body[3])`, with scalar-first unit quaternions. Quaternions differing by sign represent the same attitude, so compare rotation matrices or observations when checking composition of the physical group action. The observations are `(p, velocity, R_column_1, R_column_2, R_column_3, Omega_body)` in dimension 18. Initial position, velocity, and body rates are uniform in balls of radii 1, 0.5, and 6, and the initial attitude is Haar distributed. The full plant is simulated, including angular-velocity transients and unregulated translation. Do not replace it with thrust-axis trajectories when testing whether a latent state is learned from the full observation.

## EMLP interfaces

The [upstream package](https://github.com/mfinzi/equivariant-MLP) supplies group and representation constructors and the [Objax EMLP implementation](https://emlp.readthedocs.io/en/latest/_modules/emlp/nn/objax.html). In an environment with EMLP installed:

```python
from examples.systems import emlp_representations, make_emlp_models

group, observation, latent = emlp_representations("o3")
encoder, decoder, latent_map = make_emlp_models("o3", width=128, layers=3)
```

The representation checks and small forward evaluations were tested with EMLP 1.0.3, Objax 1.8.0, JAX and JAXlib 0.4.23, NumPy 1.26.4, and SciPy 1.11.4 on Python 3.11. Keep that environment separate from the Python 3.13 CMGDB environment and exchange numerical data through the NPZ files.

Here `T(0)` denotes a scalar and `V` the base vector representation. `SO2eR3()` is SO(2) embedded as rotations about the vertical axis in three dimensions.

| System | Group | Observation | Latent candidates |
|---|---|---|---|
| Sphere | `SO(2)` | `3*T(0) + 2*V` | `T(0) + V`, normalized to the sphere |
| O(3) flow | `O(3)` | `3*T(0) + 2*V` | `T(0) + V` |
| Quadrotor | `SO2eR3()` | `5*V + 3*T(0)` | `V`, `2*V`, `3*V`, or the full observation representation |

For the quadrotor, the three components of body angular velocity are invariant scalars under a vertical world rotation. The five vectors are position, velocity, and the columns of the attitude matrix. The optional `latent` argument accepts `sphere`, `v1`, `v2`, `v3`, or `observation`. `sphere` and `v1` have the same linear representation. For the sphere candidate, normalize the outputs of the encoder and latent map by their Euclidean norms. Measure raw output norms and retain nonfinite or vanishing-output failures. The factory returns the unnormalized networks.

Use reconstruction loss for `D(E(x))`, prediction loss for `D(f**k(E(x)))`, and latent consistency loss comparing `f(E(x))` with `E(x_next)`. For the full quadrotor use horizons 1, 4, and 16, average errors within the position, velocity, rotation, and body-rate blocks, and then average the four block losses. Fit normalization statistics on training data only. Center vector blocks only by an invariant vector, and scale each irreducible rotation block by a shared factor so normalization commutes with the action. Match ordinary-network baselines by total parameter count and select checkpoints using validation prediction error. Keep held-out trajectories separate from model selection.

The intended quadrotor protocol uses three hidden layers of width 128, batches of 256, AdamW with weight decay `1e-6`, 20,000 updates, and a cosine learning rate from `1e-3` to `1e-5`, with five seeds. Other capacities and latent spaces can be reported with their failed fits and their actual parameter counts. The existing figure data do not establish the outcome of a new EMLP fit.

## Comparing with the quotient computations

```bash
python examples/check_systems.py
python examples/check_systems.py --emlp
python examples/systems.py attitude-baseline examples/output/attitude-reference --depth 7
```

The checks exercise state dimensions, flow composition, physical and observation equivariance, sphere and rotation constraints, the decay of angular-velocity error, and an independent ODE comparison of the reduced-attitude time map. `--emlp` also checks representation packing.

The reduced attitude reference on the invariant set of zero angular-velocity error uses height `z = q[2]`. At time `log(2)`, the nominal graph has the edge `S -> N`; the coning graph has `S -> C`, `D -> C`, and `D -> N`, with `C` at height `1/4` and `D` at height `7/8`. The baseline command computes the quotient graph and index annotations with CMGDB from exact rational endpoint enclosures. Its original-space degree lists are the analytic reference indices. This is the quotient of the thrust-axis factor after restricting to zero angular-velocity error. It is not the quotient of the full plant, even on that invariant set.

For a learned sphere latent, evaluate the latent map on one representative of each vertical-rotation orbit and take the resulting height. For a vector latent, the vertical-rotation quotient has coordinates `(sqrt(x*x+y*y), z)`. For several vectors, the quotient also needs their relative configurations. A sampled cell-image hull alone is not a validated outer approximation. A new architecture needs bounds for its actual learned map before using a block certificate or claiming certified agreement of Conley indices.
