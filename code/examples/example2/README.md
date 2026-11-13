# Example 2: O(3) flow

This folder trains an invariant encoder, an orbit decoder, and a two-dimensional time map jointly from the four physical coordinates of the system in Section 5.2. Native `CMGDB.BoxMapData` computes its quotient graph and numerical regions of attraction from encoded physical sources. The latent grid is fixed at $64\times64$ cells, with three physical sampling densities.

Run commands from `code/` after the [shared setup](../../README.md#setup).

```bash
python examples/example2/run.py replay --out outputs/example2-replay
python examples/example2/run.py train --out outputs/example2-new-fit
```

Replay loads the fixed data and all three saved checkpoints, recomputes encoder diagnostics, native data maps and basin labels, and checks them against the frozen reference. Training starts new joint fits with the same training data and recipe, followed by the same data-map construction. Both commands write `results.pdf`, `regions-of-attraction.pdf`, `sampling-study.pdf`, standalone panels in `figures/`, and raw native outputs. Reports show the data-map graphs and basins, sampling-density comparisons, and physical inverse images under the encoder. Each command requires a new output directory. To replay only the validation-selected fit, add `--seeds 1`.

The [attracting-region tolerance command](reference/tolerances/README.md) accepts a checkpoint with `run.py tolerances --model MODEL --out OUTPUT`. It regenerates physical source pairs and native BoxMapData, rebuilds the safe sink regions, and compares numerical tolerance margins with sampled residuals. The reference includes all three seeds and source densities.

## Physical system and learned maps

For $(a,v)\in X=[-3/2,3/2]\times B^3_{3/2}$,

$$\dot a=a(1-a^2),\qquad \dot v=(1-2a^2-\|v\|^2)v.$$

The action is $(a,v)\mapsto(a,Qv)$ for $Q\in O(3)$. Its orbit coordinates are $(a,r)=(a,\|v\|)$, with both signs of $a$ retained. The time step is $\tau=\log3$. CMGDB analyzes one learned step $f$ at this time step. Ten-step comparisons are forecasts over $10\log3$. Writing $D=1+8a^2$, the exact physical map and its quotient are

$$F_\tau(a,v)=\left(\frac{3a}{\sqrt D},\frac{3v}{\sqrt{D(D+8\|v\|^2)}}\right),\qquad \overline F_\tau(a,r)=\left(\frac{3a}{\sqrt D},\frac{3r}{\sqrt{D(D+8r^2)}}\right).$$

The learned maps are an O(3)-invariant encoder $E:X\to Z$, a time-$\tau$ map $f:Z\to Z$, and an orbit decoder $d:Z\to[-3/2,3/2]\times[0,3/2]$, where $Z=[-2,2]\times[0,2]$ has trivial group action. Decoder outputs are the scalar coordinate and radius of a physical orbit.

```text
           physical time-log(3) map
    x in X --------------------------> F_tau(x) in X
      |                                      |
      | learned invariant E                  | same E
      v                                      v
    u in Z -------- learned f --------> u_next in Z
      |                                      |
      | learned orbit decoder d              | same d
      v                                      v
 reconstructed (a,r)                   predicted (a,r)
```

All raw networks use upstream Objax EMLP with group `O(3)`. The encoder representation is `T(0)+V` to `2*T(0)`, with two hidden layers of `16*T(0)+16*V`, of total dimension 64. The decoder and dynamics have scalar input and output pairs and two hidden layers of `32*T(0)`. The configuration's `width=32` denotes these representation counts. The encoder receives `(a,v1,v2,v3)` directly.

For raw networks $e,A,B$, the heads are

$$E(x)=(a,\|v\|)\odot\exp(0.2\tanh e(x/1.5)),$$

$$d(u)=\operatorname{clip}_{[-1.5,1.5]\times[0,1.5]}\left(u\odot\exp(0.35\tanh A(u/2))\right),\qquad f(u)=2\tanh\left(\frac{u\odot\exp(3\tanh B(u/2))}{2}\right).$$

The products and nonlinearities are coordinatewise. The orbit geometry, multiplicative corrections, scalar sign preservation, and zero axes are prescribed. The encoder learns gain corrections around the known quotient coordinates. The learned gains determine the nonzero fixed points and dynamics. Sampled Jacobians and orbit-separation diagnostics do not establish global injectivity of the induced map on the orbit space.

## Fixed data and joint training

[data/trajectories.npz](data/trajectories.npz) contains the exact data used for the saved fits. Its keys are `train_x`, `validation_x`, and `test_x`, with shapes `(16384,11,4)`, `(4096,11,4)`, and `(4096,11,4)`. The second axis indexes times $0,\tau,\ldots,10\tau$; the state order is `(a,v1,v2,v3)`. [metadata.json](data/metadata.json) records the split seeds and data hash.

The split generators use independent seeds `20261112`, `20261113`, and `20261114`. Base starts have uniform $a\in[-1.5,1.5]$, uniform radius $r\in[0,1.5]$, and uniform direction on $S^2$. One eighth is moved near $a=0$, one eighth near $r=0$, and one quarter near the four reference equilibrium orbits. Logarithmic distances are sampled on the ranges in [training.py](training.py), function `sample`. A small additional group lies exactly on the axes. Each split ends with four equilibrium-orbit controls, so those controls are shared even though the random draws are independent. Radius-uniform sampling is the specified quotient-space sampling recipe.

Only the time-zero and time-one states train the model. If $y=F_\tau(x)$ and $\pi(x)=(a,\|v\|)$, the three equally weighted losses are

$$\mathbb E\|d(E(x))-\pi(x)\|^2,\qquad\mathbb E\|d(f(E(x)))-\pi(y)\|^2,\qquad\mathbb E\|f(E(x))-E(y)\|^2.$$

All parameters of $E,d,f$ are updated simultaneously by Adam. The default has 20,000 updates, batches of 512 sampled with replacement, and a cosine learning rate from `0.002` toward `0.0001`. NumPy and the Objax default generator use the model seed, and minibatches use `default_rng(seed + 200)`. Checkpoints are selected by the complete validation loss, evaluated initially, after step one, every 100 steps, and at the final step. The seed ranking is fixed by validation before graph analysis. The reference runs took about 24 minutes per seed on the reference machine; times depend on hardware and load. Use the shared pinned CPU/float64 environment. Replay provides the fixed checkpoint comparison, while fresh training reports its actual agreement or disagreement with that reference.

[config.json](config.json) contains the editable training and analysis settings. `train --config CONFIG.json` uses another configuration; `--steps N` overrides its update count. The effective settings are saved with the result. A short fit may have a different graph or lack the two-sink basin configuration. Such fits retain their actual graph and report unavailable basin labeling rather than substituting the reference result.

## Data-map construction

The graph construction uses a physical source sample independent of the training, validation, and test splits. On the section $x=(a,r,0,0)$, sample uniform grids of $257^2$, $513^2$, and $1025^2$ points in $[-1.5,1.5]\times[0,1.5]$, including the boundary and both axes. Each section point represents a complete O(3) orbit. Four physical equilibrium-orbit controls are appended to each grid. The grids are nested, and all three use the same fixed networks.

Form the pairs

$$z_i=E(x_i),\qquad w_i=f(E(x_i)).$$

Every source supplied to the data map lies in $E(X)$. The targets use the learned time map $f$, so this analyzes the learned dynamics. In particular, they are not replaced by $E(F_\tau(x_i))$. The physical samples provide denser coverage after training, without changing the fitted parameters or the validation ranking.

`CMGDB.BoxMapData(z, w, map_empty="outside", domain_padding=False, padding=False, ...)` supplies both scalar and batch callbacks directly to CMGDB. For each closed latent cell, its image is the coordinate hull of targets paired with sources inside that cell. A cell with no source samples maps outside the ambient rectangle. No empty-cell interpolation, expanded source query, or evaluation of $f$ at artificial latent cell corners is used. Image hulls can cover points between sampled targets; they remain numerical hulls rather than validated enclosures.

The largest grid supplies the primary basin panels. The report compares all three densities at the same $64\times64$ latent grid. The raw cell-graph edges can change with density even when Morse cells and basin labels agree. Saved checks verify that supported cells, sample counts, image hulls, and transition edges grow consistently for the nested source sets. An empty cell is unknown, not proof that it is outside $E(X)$.

```bash
python examples/example2/run.py replay --source-sizes 257 513 1025 --out outputs/example2-sampling-study
python examples/example2/run.py replay --seeds 1 --source-sizes 1025 --out outputs/example2-selected
```

Source sizes must be increasing distinct values of the form `2^k+1`; the latent grid remains $64\times64$. Outputs are organized as `seedN/S1025/K64/`, with raw data-map images, native graphs, occupancy, basin labels, physical preimages, and sampling summaries. `source-pairs.npz` stores the generated largest-grid pairs for independent inspection; it is a generated output rather than a bundled input.

## Regions of attraction

The physical system has sinks $A_-=(-1,0)$ and $A_+=(1,0)$, with basins $a<0$ and $a>0$. The quotient origin connects to the saddle $(0,1)$, which connects to both sinks.

For each native cell graph, `CMGDB.MorseSingletonReachability` identifies cells whose only reachable Morse node is one sink. `MorseReachabilityMasks` cross-checks these labels, and `MorseDirectedPathCells` gives the origin-to-saddle path cells. In learned coordinates, a supported cell contains at least one of the encoded physical source samples. The training splits do not determine this support mask. A sink label is assigned only if no unsupported cell is reachable. Forward invariance of these finite-graph sink labels is checked.

The learned computation is the data map on supported cells, with unavailable sources mapped outside the rectangle. The report distinguishes its complete supported phase view from its one-step image-cover view. Its native annotations belong to this data map.

Pale sink colors show the singleton basins; pale saddle color shows origin-to-saddle paths; dark colors show native Morse cells. White remains unresolved, including paths to unsupported cells. Gray marks unsupported cells. Physical panels use the sampled inverse images under $E$, evaluated on a $513\times513$ physical section; they do not use the decoder. Each physical $(a,r)$ represents its full O(3) orbit.

For the selected seed at all three physical sampling densities, the $64\times64$ grid has 2,400 supported cells: 1,160 labeled for each sink and 80 unresolved. No supported transition reaches an unsupported cell. These are latent-cell counts, not physical-volume estimates.

| Model seed | Nodes with $257^2$ physical sources | With $513^2$ | With $1025^2$ |
|---|---:|---:|---:|
| 0 | 5 | 5 | 5 |
| 1, selected | 4 | 4 | 4 |
| 2 | 5 | 5 | 4 |

All nine computations have the same supported-cell and signed-basin counts listed above. The minimum source count per supported cell grows from 6 to 15 to 54. For the selected seed, Morse cells, Morse graph edges, native annotations, and signed basin labels agree across densities. Additional raw cell-graph transitions are still added as sampling becomes denser. Seed 2's extra component merges at the densest sample; seed 0's extra component is retained. Nodes that receive no index annotation from CMGDB are displayed with their number alone; an absent annotation is not a zero index.

## Learning diagnostics

| Seed | Best validation | One-step orbit RMS | Ten-step orbit RMS |
|---|---:|---:|---:|
| 0 | 0.0000599135 | 0.00568354 | 0.0203858 |
| 1 | 0.0000535414 | 0.00531243 | 0.0179731 |
| 2 | 0.000101341 | 0.00738023 | 0.0174773 |

Seed 1 is selected at step 19,900. Its reconstruction orbit RMS is `0.0000992468`, sampled encoder invariance defect is about `6.7e-16`, and minimum sampled encoder singular value is `0.818909`. All three networks changed in every fit. Extra native components remain in the output.

## Generate training data

```bash
python examples/example2/run.py data --out outputs/example2-data
python examples/example2/run.py train --data outputs/example2-data/trajectories.npz --out outputs/example2-regenerated-data-fit
```

The data command regenerates the specified split recipe; `--sizes TRAIN VALIDATION TEST` changes the counts, with at least eight starts per split. The file format is the joint experiment's split arrays described above.

## Numerical status

Data-map boxes are hulls of sampled targets paired with encoded physical sources; they have no interpolation or padding. These hulls are not validated enclosures. Basin reachability is exact for the resulting finite graph; interpreting it as a region of attraction of the continuous map requires valid outer-approximation hypotheses. Support and physical preimages are sampled diagnostics. Native quotient annotations and original-space indices remain separate, and this experiment does not establish a transfer certificate.

Input manifests cover the dataset, numeric checkpoints, saved diagnostics, native graphs, and basin labels. `reference/data-map/` contains the data-map graph references, and `reference/joint/` supplies the checkpoints. Replay checks image endpoints within `2e-13`, exact native cell sets and graph edges, occupancy and basin arrays, and numerical diagnostics with relative tolerance `1e-6` and absolute tolerance `1e-9`. The cached physical-section coordinates are checked against 66,049 freshly encoded section points before the physical basin panels are drawn. Axis-root diagnostics search only the ranges attained by the encoded physical axes.
