# Example 3: Two learned reductions for attitude control

The separate [autonomous coning variant](periodic/README.md) constructs a single attracting coning orbit with two repelling poles. It includes a controller proof and a task-aware joint learning experiment on the invariant sphere subsystem. The original full-state experiment below and its saved inputs are preserved.

This experiment learns a sphere model of the full closed-loop plant, then learns an invariant interval encoder and interval dynamics for that sphere model. Both the nominal pointing and coning controllers have three retained seeds. Native CMGDB analyzes the learned scalar map with a regular box callback on 64 or 128 interval cells. The default is 64; `--cells 128` performs the finer computation on the same models.

The supplied sphere checkpoints are completed 20,000-update joint EMLP fits, verified in the shared Python 3.13 environment and reused. The supplied interval checkpoints are six new 20,000-update fits. These runs do not reproduce the analytic reference graph orders. The report separates physical model error from interval distillation error and retains every native component.

Run commands from `code/` after the [shared setup](../../README.md#setup).

```bash
python examples/example3/run.py replay --out outputs/example3-replay
python examples/example3/run.py replay --cells 128 --out outputs/example3-replay-128
python examples/example3/run.py train --stage interval --out outputs/example3-new-interval-fits
```

Replay loads both stages, recomputes diagnostics and graphs, checks the frozen references, and writes `results.pdf`, `native-graphs.pdf` for all six models, `reference-graphs.pdf` for the matched thrust-axis references, standalone panels in `figures/`, and raw native outputs. Graphs retain the native node IDs, annotations, edges, and layout as vector graphics. Interval cell panels use `CMGDB.PlotMorseSets1D`. Interval training reuses the bundled sphere models and trains new interval encoders, orbit decoders, and time maps. Add `--policy nominal` or `--policy coning` to select one controller; `--seeds 0 1` selects two seeds. Each command requires a new output directory.

To restart both learned stages:

```bash
python examples/example3/run.py train --stage both --out outputs/example3-all-new-fits
```

First-stage training is substantially more expensive than the interval fits. The editable [config.json](config.json) records both recipes. `--config FILE`, `--sphere-steps N`, and `--steps N` select another configuration or update counts. Checkpoints and model outcomes can differ in a fresh run; the actual results are retained.

## Physical system and symmetry

The physical state is $(p,v,R,\Omega)\in\mathbb R^3\times\mathbb R^3\times\mathrm{SO}(3)\times\mathbb R^3$. Its dimension is twelve. Simulations store scalar-first unit quaternions, giving thirteen stored state coordinates. Network observations are $(p,v,R_1,R_2,R_3,\Omega)\in\mathbb R^{18}$, with the columns of $R$ in order. Vertical rotations act by

$$Q\cdot(p,v,R,\Omega)=(Qp,Qv,QR,\Omega).$$

The observation representation is `5*V + 3*T(0)` for `SO2eR3`: the five world-frame vectors rotate, while the three body-rate components are invariant scalars. The latent sphere has the standard vertical SO(2) action. The interval action is trivial. For a thrust axis $q=Re_3$ and $z=q_3$, the controller induces

$$\dot q=a(z)(e_3-zq)+\omega e_3\times q+\eta\times q,\qquad \dot\eta=-2\eta,$$

with $\omega=\pi/\log2$. The nominal gain is $a(z)=1$. The coning gain is $a(z)=8(s(z)^2-1/4)$, where $s(z)=(z-2/3)/(1-2z/3)$. The full simulator, including translation and the moment feedback, is in [systems.py](../common/systems.py). Position is not regulated, and there is no actuator saturation.

On the invariant subsystem $\eta=0$, the analytic reference orders are `S -> N` for nominal pointing and `S -> C`, `D -> C`, `D -> N` for coning, with physical heights $C=1/4$ and $D=7/8$. These are interpretation references for the thrust-axis subsystem. The learned sphere coordinate is not prescribed to equal that thrust axis.

On the physical thrust-axis sphere, a repelling pole has tuple `(0,0,x-1)`, an attracting pole has `(x-1,0,0)`, an attracting periodic latitude has `(x-1,x-1,0)`, and a repelling periodic latitude has `(0,x-1,x-1)`. Disk and annulus index pairs give these groups; the flow induces identity index maps. These are analytic reference indices. Native interval tuples have only degrees 0 and 1: the quotient collapses the circle fibers, and a repelling endpoint has zero interval index. Sphere reconstruction is a separate step.

A different coning gain with a single attracting latitude, such as $a(z)=c-z$ with $-1<c<1$, would have two repelling poles and one attracting periodic orbit. That is the three-node sphere order `S -> C <- N`, with tuple `(x-1,x-1,0)` at `C`. The implemented coning gain also stabilizes the north pole, separated from `C` by the repelling latitude `D`; its reference order has four nodes.

## Learned maps

```text
       full plant X -------- physical F_0.1 --------> X
             |                                      |
             | learned equivariant E                | E
             v                                      v
            S^2 ----------- learned f ------------> S^2
             |                                      |
             | learned invariant pi                 | pi
             v                                      v
          [-1,1] ---------- learned f_bar --------> [-1,1]

       D: S^2 -> R^18       reconstructs observations
       r: [-1,1] -> [-1,1]  reconstructs latent orbit heights
```

The upper square is learned approximately from physical trajectories. The lower square is learned approximately from the frozen sphere map. The SO(2) action and its orbit geometry are specified. Learning `pi` means learning coordinates on that known orbit space. The full plant's SO(2) quotient is larger than an interval.

### Stage 1

The encoder, observation decoder, and sphere time map are upstream Objax EMLPs with three width-128 hidden layers. Encoder and time-map vector outputs are normalized. The latent representation is `V`; no physical thrust-axis target is supplied. There are 167,688 trainable parameters in total. All learned-model calculations use float32, with the shared pinned library versions.

The loss is reconstruction plus prediction averaged over horizons 1, 4, and 16, plus `0.1` times one-step latent consistency. Squared errors are averaged within the position, velocity, rotation-matrix, and body-rate blocks after training-only normalization. Each world-vector block has a common scale and may be centered only in its invariant vertical component. This preprocessing preserves SO(2) equivariance.

AdamW uses batches of 256, 20,000 updates, weight decay `1e-6`, and a cosine learning rate from `0.001` to `0.00001`. Validation uses one fixed window per trajectory, with start indices from seed `8181`. Checkpoints are selected by validation prediction, then reconstruction. Minibatches use seed `10900 + model_seed`. The reused checkpoints were selected before any new interval fit or graph analysis.

The implementation contracts the same EMLP bilinear operator without constructing its dense batch matrix and projects each linear weight once per rollout. Ordered parameters are unchanged. Each converted checkpoint was checked against its stored fingerprint and the upstream native backend; maximum differences are below `3e-6`. The earlier experiment remains separate. This package supports fresh training of the same sphere architecture, with finite-loss, gradient, and optimizer-state checks.

### Stage 2

An invariant encoder $\pi:S^2\to[-1,1]$, an orbit decoder $r$, and a scalar map $\bar f$ are trained jointly on pairs $(z,f(z))$ from the frozen sphere model. The split sizes are 16,384, 4,096, and 4,096, with seeds `2026111301`, `2026111302`, and `2026111303`. Heights and azimuths are sampled independently, one quarter of heights are near the poles, and both exact poles are included. These sources cover the latent sphere rather than being additional physical trajectories.

The coordinate map has the form $\pi(z)=h(z_3)$. On 65 fixed latitude knots, an invariant EMLP assigns positive increments `exp(0.35*tanh(network))`; their normalized cumulative sum gives endpoints `-1` and `1`, and linear interpolation defines $h$. Thus $h$ is a strictly increasing homeomorphism in real arithmetic, with segment slopes between $e^{-0.7}$ and $e^{0.7}$. The decoder uses an independently learned positive-increment coordinate map. The known orbit geometry and coordinate orientation are prescribed; the coordinate values are learned. Both networks have two hidden layers of 32 scalar channels.

The time head is

$$\bar f(u)=\operatorname{clip}_{[-1,1]}\bigl(u+2\tanh B(u)\bigr),$$

where $B$ is another two-layer, width-32 scalar EMLP. Its endpoint images and interior fixed-point count are learned. The three unit-weight losses are

$$\mathbb E|r(\pi(z))-z_3|^2,\qquad\mathbb E|r(\bar f(\pi(z)))-f(z)_3|^2,\qquad\mathbb E|\bar f(\pi(z))-\pi(f(z))|^2.$$

Adam uses batches of 512, 20,000 updates, and a cosine learning rate from `0.002` to `0.0001`. Initialization uses seed `1000 + model_seed`; minibatches use `1200 + model_seed`. The checkpoint minimizes the sum of all three validation losses, checked initially, after step one, every 100 updates, and at the final update. All three networks change in every retained fit.

## Data and regeneration

[data/metadata.json](data/metadata.json) records the six physical splits. For each controller, there are 16,384 training, 2,048 validation, and 4,096 test trajectories, each with 33 observations at times `0, 0.1, ..., 3.2`. The simulator uses RK4 with step `0.001`. Position, velocity, and body rates start uniformly in balls of radii 1, 0.5, and 6; attitude is Haar distributed. The split seeds are `2026111111`, `2026111112`, and `2026111113`. Both controllers use the same initial states. Split by trajectory before forming windows.

Each physical `.npz` contains `observations` of shape `(N,33,18)`, `initial_state` of shape `(N,13)`, and `times` of shape `(33,)`, all float64. Frozen numeric arrays are retained without numerical changes. The largest input file is below 100 MB.

```bash
python examples/example3/run.py data --out outputs/example3-data
python examples/example3/run.py train --data outputs/example3-data --stage both --out outputs/example3-regenerated-data-fit
```

`data --sizes TRAIN VALIDATION TEST` changes the split sizes, with at least eight trajectories per split. `--policy` selects the controller. The fixed sphere checkpoints retain their original normalization. Fresh sphere training fits normalization from the supplied training split only.

## Current results and their limits

| Controller | Seed | Test prediction block MSE | Physical sphere closure RMS | Interval consistency RMS | Nodes at 64 | Nodes at 128 |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 0 | 0.4512 | 0.4765 | 0.00985 | 16 | 16 |
| nominal | 1, selected | 0.4301 | 0.5966 | 0.00248 | 17 | 21 |
| nominal | 2 | 0.4511 | 0.4010 | 0.00242 | 40 | 44 |
| coning | 0 | 0.3299 | 0.5744 | 0.00985 | 3 | 2 |
| coning | 1 | 0.3339 | 0.6115 | 0.00898 | 7 | 5 |
| coning | 2, selected | 0.2992 | 0.4263 | 0.00575 | 6 | 5 |

Physical sphere closure measures $\|f(E(x))-E(F_{0.1}(x))\|$ on independent physical test starts. Interval consistency measures $|\bar f(\pi(z))-\pi(f(z))|$ on independent latent-sphere test points. They measure different approximation errors. The second stage fits the sphere model reasonably closely, while the first stage has substantial error relative to the plant. The models do not recover the analytic reference orders, and matching the lower square does not establish accuracy of the upper square.

All six scalar fits also violate exact pole compatibility: at least one endpoint maps into the interval interior, with the largest error about `0.00950`. An exact quotient of an SO(2)-equivariant sphere map must map pole endpoints to pole endpoints. The current generic scalar head imposes boundedness but allows this error. Replay records both the fitted endpoint images and the sphere-induced endpoint targets. These fits are approximate reduced models and do not satisfy this necessary condition for an exact sphere quotient. Increasing the grid resolution leaves that modeling defect unchanged.

Native CMGDB samples the scalar function at 257 points per requested cell and adds padding `2e-6`, clipped to the interval. This regular construction analyzes the learned interval model at time `0.1`. The learned coordinate covers the full interval on the latent sphere. These function hulls are numerical, not validated enclosures. Many extra finite-grid components have trivial native annotations and can arise from cell self-loops under weak drift. Every component is retained; their count is not a count of true invariant sets.

The runner also computes the physical thrust-axis reference at the same time step, grid, and padding. It solves the scalar height equation with DOP853 (`rtol=2e-13`, `atol=2e-14`) at cell vertices and uses monotonicity of a scalar flow for endpoint hulls. The nominal endpoint images agree with the closed formula to within `1e-12`. This is a numerical reference, not a validated enclosure. At 64 and 128 cells its nominal graph has 10 components, with one nonzero annotation and one sink. The coning reference has respectively 9 and 8 components, with three nonzero annotations and two sinks. Thus even the reference's short time map has more native components than its ideal orbit order. The learned nominal graphs have two sinks; the learned coning graphs have one. The mismatch remains after distinguishing trivial finite-grid components from the underlying reference orbits.

Intrinsic interval basins use native singleton Morse reachability. Physical support is measured separately from encoded training and validation observations at indices `0,1,4,16,32`. Physical labels exclude paths into cells without that support. Selected coning seed 2 visits 35 of 64 cells, or 70 of 128 cells, and no physical cells receive a safe sink label in either run. Missing samples are not proof that an orbit is physically impossible. `physical-diagnostics.npz` retains full-state sampled preimages under `pi o E`, including physical tilt and test basin labels. The main report shows native cells, latent sphere preimages, and learned-map drift.

Reports distinguish the learned latent sphere from the physical thrust axis. Native index annotations apply to the numerical interval map. No new sphere-index reconstruction or physical transfer certificate is claimed. Normalization, equivariance, finite rollout, coordinate-order, and inverse-coordinate diagnostics are recomputed on replay. Float32 metrics use relative tolerance `1e-4` and absolute tolerance `3e-6`; native cell sets, graph edges, basin arrays, and discrete fields must match exactly.

All fixed data, numeric checkpoints, and raw reference records are covered by [manifest.json](manifest.json). The original inputs are unchanged; `reference/cmgdb128/` adds the six finer-grid native references, and `reference/time01/` adds the matched thrust-axis reference records at both grids. Generated trajectories, pair caches, diagnostics, graphs, and reports go to the requested ignored output directory. `reference/` contains no executable checkpoint serialization.
