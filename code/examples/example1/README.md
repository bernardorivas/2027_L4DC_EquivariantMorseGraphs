# Example 1: SO(2) flow on the sphere

This folder provides joint learning from three physical sphere coordinates to a one-dimensional interval, followed by native CMGDB. It corresponds to the physical system in Section 5.1. The interval experiment and the older saved manuscript fit are separate reference models.

Run the commands from `code/` after the [shared setup](../../README.md#setup).

```bash
python examples/example1/run.py replay --out outputs/example1-replay
python examples/example1/run.py train --out outputs/example1-new-fit
```

`replay` loads the frozen weights, recomputes diagnostics and all nine learned graphs, checks the results against the reference, and produces a six-page comparison PDF. `train` fits all three seeds jointly on the fixed data, computes their graphs, and produces the same report. Both commands keep the input files unchanged and require a new output directory. To replay only the selected fit, add `--seeds 1`.

## Physical map and data

The state is $p=(x,y,z)\in S^2$. The group acts by rotations about the vertical axis. Training pairs are $(p,F_1(p))$, where $F_1$ is the time-one map of

$$\dot p=0.6\,e_3\times p+z(0.16-z^2)(e_3-zp).$$

Thus $\dot z=z(0.16-z^2)(1-z^2)$ and longitude advances at rate $0.6$. The comparison with the explicit quotient uses time ten, which is ten iterates of the learned time-one map.

The exact arrays used for the supplied fits are in [data/trajectories.npz](data/trajectories.npz), with [metadata](data/metadata.json). There are $2^{14}$ training starts, $2^{12}$ validation starts, and $2^{12}$ test starts. Each start supplies one time-one pair. The splits are independent and fixed across the model seeds.

For each split, three quarters of the starts are uniform on the sphere. The remaining quarter has uniformly sampled longitude and symmetric heights with $\log_{10}(1-|z|)$ uniform on $[-10,-2]$. The generator uses `numpy.random.default_rng(20261112)`, draws the splits sequentially, and shuffles their heights. Use the supplied arrays for comparison with the frozen experiment.

For `split` equal to `train`, `validation`, or `test`, the archive contains:

| Keys | Shape | Meaning |
|---|---|---|
| `split_p0`, `split_p1`, `split_p10` | `(count, 3)` | Physical states at times zero, one, and ten, ordered `(x,y,z)` |
| `split_w0`, `split_w1`, `split_w10` | `(count,)` | Log latitude $w=\operatorname{arctanh}(z)$ at those times |

The data generator in [training.py](training.py) integrates $\dot w=\tanh(w)(0.16-\tanh(w)^2)$ with DOP853, relative tolerance `2e-13`, and absolute tolerance `2e-14`. Physical states use radius $1/\cosh(w)$ and the rotating longitude. Stored log latitudes condition the loss near the poles. The training objective uses only time-zero and time-one data; time-ten states are rollout diagnostics.

## Learned maps

The learned maps are an invariant encoder $E:S^2\to[-1,1]$, a scalar orbit decoder $d$, and an interval time-one map $f$. The decoder height represents the latitude orbit of that height and radius $\sqrt{1-d(u)^2}$.

All three raw networks use upstream Objax EMLP with two hidden layers of 32 scalar channels. The group constructor is `SO2eR3()`. The encoder input is its three-dimensional vector representation `V`, ordered `(x,y,z)`, and its output is `T(0)`. The decoder and time map have scalar input and output representations. The hidden representation is `32*T(0)`; the upstream mixed-tensor choice is a different architecture.

For $M(u,s)=(u+\tanh s)/(1+u\tanh s)$ and raw networks $e,a,b$, the implemented heads are

$$E(p)=M(z,0.35\tanh e(p)),\qquad d(u)=M(u,0.35\tanh a(u)),\qquad f(u)=M(u,1.4\tanh b(u)).$$

The quotient geometry, height skip connection, and interval endpoints are prescribed. The encoder learns a bounded reparametrization of height. The interior fixed points and computed graph are outcomes of training. Reconstruction is modulo the rotation action.

## Joint objective and optimization

All three networks are optimized simultaneously. Reconstruction compares $d(E(p))$ with the orbit of $p$. Prediction compares $d(f(E(p)))$ with the orbit of $F_1(p)$. Latent consistency compares $f(E(p))$ with $E(F_1(p))$.

The reconstruction and prediction terms use the mean squared orbit distance $(z-z_*)^2+(r-r_*)^2$ plus `0.03` times squared log-latitude error. The consistency term uses squared latent-height error plus `0.08` times squared encoded log-latitude error. The ordering penalty is `0.1` times the mean squared positive part of `0.2 - slope`, using encoder secant slopes on 65 equally spaced heights in `[-0.999,0.999]`. The three main terms have unit weight. [training.py](training.py), function `objective`, evaluates these expressions before converting the log coordinates back to heights.

[config.json](config.json) records the architecture, head bounds, loss weights, and optimization settings. Adam uses batches of 512 sampled with replacement and 20,000 updates. The cosine learning rate decreases from `0.002` toward `0.0001`. NumPy and the Objax default generator are seeded with the model seed; the minibatch generator is `default_rng(seed + 100)`. Validation is evaluated initially, at step one, every 100 updates, and at the final step. The best checkpoint is saved by the complete validation objective. The report selects the seed with the lowest validation objective before inspecting CMGDB.

The reference environment is Python 3.13 on CPU with JAX float64 enabled. Numerical package versions are pinned in [requirements.txt](../../requirements.txt). The supplied fits and an independent seed 1 retraining agree on macOS arm64. Fresh optimization can differ across numerical platforms; checkpoint replay provides the fixed comparison.

For a short execution check, use a new directory and fewer steps:

```bash
python examples/example1/run.py train --seeds 1 --steps 20 --out outputs/example1-smoke
```

A short run is not expected to recover the reference graph. For a new experiment, edit `config.json` or pass a copy with `train --config CONFIG.json`. The effective settings and fitted checkpoint are saved with the output. Replay always uses the immutable configuration in `reference/joint/`.

## Expected results

The frozen joint experiment contains these outcomes:

| Seed | Best validation objective | Ten-step orbit RMS | Nodes at $2^6$ | At $2^7$ | At $2^8$ |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000312542 | 0.072153 | 15 | 13 | 12 |
| 1 | 0.0000519593 | 0.016355 | 5 | 5 | 6 |
| 2 | 0.000502196 | 0.083925 | 7 | 7 | 6 |

Seed 1 is selected, at step 19,700. Its reconstruction, one-step prediction, and ten-step prediction orbit RMS errors are `0.000722116`, `0.00481847`, and `0.0163549`. At $2^6$ and $2^7$ cells its native graph order and annotations match the explicit reference. At $2^8$ it has an extra node. The other seeds remain in the report.

## CMGDB, outputs, and older manuscript fit

The explicit time-ten quotient images use rational endpoint enclosures. Learned images of $f^{10}$ use 65 samples per cell and padding `1e-10`. These sampled hulls are numerical approximations, not validated image enclosures. The inverse encoder at interval boundaries gives the physical latitude bands; a sampled ordering check precedes this inversion. Agreement of quotient graphs does not establish a transfer certificate or equality of original-space Conley indices.

The output includes `sphere-emlp-comparison.pdf`, standalone panels in `figures/`, `selection.json`, and `verification.json`. Each `seedN/` contains numeric weights, effective configuration, validation history, diagnostics, and raw native results at all three grids. Replay checks diagnostics with absolute tolerance `1e-9` and relative tolerance `1e-6`, image endpoints within `2e-13`, and exact agreement of native cell sets, edges, and annotations. Fresh training reports agreement or disagreement with the frozen fit and retains the actual result.

The older manuscript panels use a different saved fit with a latent sphere. Their data and fitted image enclosures are in `reference/manuscript/`. To recompute those native graphs and redraw those panels:

```bash
python examples/example1/run.py manuscript --out outputs/example1-manuscript
```

This command reproduces the saved manuscript computation from its supplied enclosures. It does not refit that older model. Its native results are separate from the new joint interval experiment.
