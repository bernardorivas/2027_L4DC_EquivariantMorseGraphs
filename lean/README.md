# LatentEquivariant (Lean 4)

A Lean 4 formalization of set-level and algebraic statements of the paper, built on Mathlib
v4.31.0. Everything is stated for maps `f : X → X`, as CMGDB computes the Conley index of maps,
not for flows.

## Build

```sh
cd lean
lake exe cache get
lake build
```

`lake exe cache get` downloads the Mathlib build, so Mathlib is not compiled locally.

Checks (2026-11-09): `lake --wfail build` succeeds with no warnings. No source file contains `sorry`, `admit`, `axiom` or `native_decide`, and no line is longer than 100 characters. A `collectAxioms` audit of all 1984 declarations of the project permits only `propext`, `Classical.choice` and `Quot.sound`.

## Conventions

- The conditions in `IsIndexPair` are based on Szymczak, *The Conley index for discrete semidynamical systems*, Topology Appl. 66 (1995), Definition 3.4, but the existing predicate omits `L ⊆ N`. Geometric applications must impose this inclusion separately. For a map the exit set must contain every point of `N` that
  leaves `N` in one step, so it is a collar, not a thin piece of `∂N`. `IsIndexPair` is defined on
  any topological space; Szymczak's index pairs live in metric spaces.
- Attracting blocks, the tolerance `τ_g(N)` and the omega-limit set follow the latent-dynamics
  paper (Bailon et al. 2026, arXiv:2609.01509), Definition of an attracting block. The tolerance takes values in `ℝ≥0∞`, so it is `∞` when every radius
  works and `0` when none does.
- A group acts on cells by permutations without signs and preserves the incidence numbers. Under
  the grid condition (an element that maps a cell to itself fixes it pointwise) cells can be
  oriented along orbits so that this holds: `Algebra/Reorient.lean` proves that a signed action
  can be made unsigned if and only if the signs are a cocycle and stabilizers preserve
  orientations (`exists_reorient`, `exists_reorient_iff`). The model is weaker than the grid
  condition: it only says that a cell mapped to itself keeps its orientation. The other consequence used, that faces
  of fixed cells are fixed, is an explicit hypothesis where needed, and identifying orbit chains
  with the chains of `N/G` needs the grid condition itself, which is not formalized.
- Relative chains `C_*(N, L)` are the chains on the cells of `N` not in `L`.

## Layout

| File | Contents |
|---|---|
| `Dynamics/Basic.lean` | Solutions, `Inv`, isolating neighborhoods, index pairs, lifting solutions (also through a prescribed point), pullback of index pairs and of their isolated invariant set, attracting blocks, tolerance, `thm:main_simple` and its weakened forms |
| `Dynamics/AttractorLattice.lean` | The finite omega-image lattice, union and omega-intersection operations, the bounded surjective omega homomorphism, survivor order isomorphism, and nonempty Morse-piece criterion |
| `Dynamics/TrajectoryLimits.lean` | Alpha limits of full trajectories, compact forward/backward limit invariance and nonemptiness, and eventual-tail containment in maximal invariant sets |
| `Dynamics/Morse.lean` | Finite block labels and their two-sided stabilization, compact disjoint Morse pieces, the full Morse representation including strict connecting-orbit order, and the induced attractor-lattice order |
| `Order/FiniteLatticeQuotient.lean` | Surviving join-irreducibles of a finite bounded lattice epimorphism, their order isomorphism with target irreducibles, and predecessor-image identity |
| `Orbit/Quotient.lean` | Orbit map, invariant sets, `f/G`, `Inv(π K, f/G) = π Inv(K, f)`, index pairs descend to `X/G` |
| `Latent/Pullback.lean` | Proposition `prop:equivariant_pullback` (i)-(iii), with weakened forms |
| `Latent/QuotientMetric.lean` | `Z/G` as a pseudometric space with the quotient topology, a metric exactly when orbits are closed, compactness, the tolerance of `Z/G`, and (ii) proved the paper's way through `Z/G` |
| `Latent/ApproxEquivariant.lean` | A lemma, not yet in the paper, for an approximately equivariant latent map `f`: the defect `δ_N`, the saturation `G N`, `τ_f(G N) ≥ τ_f(N) / L - δ_N` for an action by `L`-Lipschitz maps (`τ_f(N) - δ_N` for isometries), `G N` an invariant attracting block when `δ_N < τ_f(N)`, `r̄(g x) ≤ r̄(x) + δ_N`, and the certificate for `E⁻¹(G N)` from `sup_{E⁻¹(N)} r̄ < τ_f(N)` alone, for equivariant `E` and `F` |
| `Combinatorial/OuterApprox.lean` | Lemma `lem:equivariant_outer` and the stabilizer saturation |
| `Combinatorial/MorseOrder.lean` | Recurrent SCCs, their reverse-reachability order and induced group action, the component-orbit order formula, and the explicit quotient Morse-poset order isomorphism |
| `Combinatorial/LiftedOrder.lean` | Proposition `prop:lifted-order`: the Morse sets as `⊔_P G/H_P`, the criterion `h M̂_P ≤ k M̂_Q` if and only if `k⁻¹ h ∈ W(Q, P)` and its strict form, `H_Q W(Q, P) H_P = W(Q, P)`, `W(P, P) = H_P`, `W(Q, P)` nonempty if and only if `P ≤ Q`, and `W(Q, P)` as the voltages of positive quotient walks |
| `Algebra/Chains.lean` | Cellular chains with prescribed incidence numbers |
| `Algebra/Isotypic.lean` | Averaging and involution splittings, nilpotency of the index map componentwise |
| `Algebra/OrbitChains.lean` | `π_#`, the orbit complex, the transfer at the chain level, isomorphisms for free actions and for stabilizer orders that are units, counterexamples |
| `Algebra/SignChains.lean` | The `±1` eigenspaces of an involution, representatives, the twisted complex, fixed faces, untwisting when `N = D ∪ gD` |
| `Algebra/ShiftEquivalence.lean` | Shift equivalence and conjugacy of endomorphisms, the trivial index, automorphisms, the Leray reduction (both forms), restriction to the components of an equivariant shift equivalence, and examples where the shift class of an eigenspace component is not determined by the shift class of the total map |
| `Algebra/Reorient.lean` | Signed cellular actions, the cocycle condition, reorientation along orbits and its converse, the reflection of a grid on `[-1, 1]` |
| `Homology/Basic.lean` | `CellComplex`, its Mathlib chain complex and homology, chain maps, relative complexes, the splitting `H ≅ H⁺ ⊕ H⁻` and the nilpotency criterion |
| `Homology/Reconstruction.lean` | Proposition `prop:z2_reconstruction` assembled: orbit and twisted complexes, `H_n ≃ H_n(orbit) × H_n(twisted)` (absolute and relative), naturality and nilpotency, untwisting, exactness of `0 → C^tw → C → C^orb → 0` over any ring |
| `Examples/Doubling.lean` | The doubling map on a mirror: index pairs, the thin pair fails, the pair on the orbit space |
| `Examples/Sharpness.lean` | Counterexamples bounding the weakenings: discontinuous `f`, discontinuous `g`, dense orbits |
| `Examples/ActionSharpness.lean` | (i) and (ii) of `prop:equivariant_pullback` with only `int N` invariant; counterexamples for actions that are not by homeomorphisms and for `int N` not invariant |
| `Examples/Cellular.lean` | Homology of the doubling example and of `-I` on a square, over any ring and over `ℤ`; where the formula of `prop:z2_reconstruction` holds and where it fails |

`Dynamics/Basic.lean` also proves finite generation and pullback properties of block sublattices, compact-forward omega identities, nested block-pair Morse identities, and compactness of the resulting pieces. `Orbit/Quotient.lean` adds quotient multivalued maps and invariance of full-solution sets. `Combinatorial/OuterApprox.lean` proves quotient outer-approximation descent, positive recurrence and component-orbit correspondence, voltage path lifting and stabilizer recovery, and the forward-invariant/attractor order correspondences with the correct omega-intersection meet. `Latent/Pullback.lean` transfers generator certificates to the whole generated preimage lattice and proves invariance of the resulting Morse pieces. The generated lattice adjoins only the empty set. Its top is formed from the checked generators.

## Weakened hypotheses

Each item is proved in Lean. The last column assesses each weakening and names the files with the
examples that limit it.

| | Paper item | Not needed | Lean | Verdict |
|---|---|---|---|---|
| W1 | `thm:main_simple` | continuity of `f`, `g`; compactness of `Z`; `N` an attracting block (`N` closed suffices); `X` compact (`E⁻¹(N)` compact suffices) | `isAttractingBlock_preimage_of_isClosed`, `isAttractingBlock_preimage_of_isCompact`, `image_preimage_subset_interior` | genuine; limits in `Sharpness` |
| W2 | `prop:equivariant_pullback` (ii) | equivariance of `g`; compactness of `G`; isometries and any topology on the action (`int N` invariant suffices; `N` need not be invariant); `N` an attracting block; `X` compact | `isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior`, `_of_isCompact`, `_of_isClosed`, `IsEquivariant.isAttractingBlock_preimage_of_residualMod_of_isClosed` | genuine; limits in `ActionSharpness` |
| W3 | `prop:equivariant_pullback` (i) | isometries, a metric and any topology on the action for the block part (`int N` invariant suffices); compactness for the tolerance | `IsAttractingBlock.quotient_of_isGInvariant_interior`, `image_quotientMap_subset_interior_iff`, `isAttractingBlock_quotient_of_isCompact_image`, `tolerance_quotientMap` | genuine; artifact for noncompact `G` with `N` compact; limit in `ActionSharpness` |
| W4 | `(N/G, L/G)` is an index pair | finiteness of `G` (compact `G`, or `X/G` metrizable) | `IsIndexPair.image_orbitMap` | artifact beyond compact `G` |
| W5 | Pullback of index pairs | invariance of `h⁻¹(S̄)` for the pair conditions; properness beyond `h⁻¹(N)`, `h⁻¹(L)` | `IsIndexPair.preimage_of_isCompact`, `IsIndexPair.Inv_preimage_of_lift` | genuine |
| W6 | `lem:equivariant_outer` | finiteness of `G`, grid condition | `exists_equivariant_outerApprox`, `exists_outerApprox_of_reps` | genuine, trivial |
| W7 | `prop:equivariant_pullback` (iii) | topology on `X` except for `ω`; compactness of `G` | `residualMod_smul`, `isGInvariant_omegaSet`, `iSup_residualMod_eq` | genuine |
| W8 | Transfer (reconstruction from the orbit space) | a field; `|G|` invertible (orbit sizes units suffice, and are necessary for `π_#`); stabilizer orders units for another isomorphism | `transferEquiv`, `bijective_push_domRestrict_iff`, `scaledOrbitEquiv_bdInvariant`, `orbitFunEquiv_bdInvariant` | genuine, chain level only |
| W9 | `prop:z2_reconstruction`, `-1` part | `2` invertible (`2` a non-zero-divisor suffices, sharp) | `antiChainsEquiv`, `isLeftRegular_two_iff_forall_antiChains`, `bd_comp_antiExtend` | genuine for the `-1` chains; the `+1` part and homology still need `2` invertible |
| W10 | `prop:z2_reconstruction`, splitting | a field, the grid condition, finite dimension | `homologyEquivEigen`, `isNilpotent_iff_homology_eigenspaces` | genuine |
| W11 | Shift equivalence (Conley index of maps) | a field and finite dimension for "trivial iff nilpotent"; a field for "shift equivalent automorphisms are conjugate"; finite dimension for the Leray classification (finite length and Artinian suffice) | `shiftEquivalent_iff_isNilpotent`, `shiftEquivalent_iff_isConjugate_of_isUnit`, `shiftEquivalent_iff_isConjugate_lerayReduction` | genuine; the Noetherian hypothesis cannot be dropped (hand check) |
| W12 | `Z/G` a compact metric space (latent dynamics) | compactness of `G` for a pseudometric with the quotient topology; compactness of `G` for a metric (closed orbits are necessary and sufficient) | `quotient_topology_eq`, `forall_isClosed_orbit_iff`, `isCompactMetric_of_isClosed_orbit` | genuine; for compact `Z`, closed orbits are equivalent to the paper's hypothesis; the finite-group instance is an artifact |
| W13 | `prop:z2_reconstruction` assembled | a field (`2` invertible suffices; `2` regular for `C⁻`; nothing for the choice of representatives) | `reconstructionEquiv`, `minusIso`, `twistedChainEquiv`, `push_eq_zero_iff_mem_range_antiExtend` | genuine; `2` invertible is needed (`twisted_formula_fails`) |
| W14 | Signed cubical actions (group actions on grids) | the grid condition, for the unsigned model (cocycle signs and orientation-preserving stabilizers are necessary and sufficient) | `exists_reorient`, `exists_reorient_iff` | genuine; weakens the model's hypothesis, not a paper statement |

## Not formalized

The index map of a topological index pair, independence of the pair and Mrozek's Theorem 2.9 need
relative singular homology and excision, which Mathlib v4.31 lacks; the linear algebra of shift
equivalence is in `Algebra/ShiftEquivalence.lean`. The identifications of the
orbit and twisted complexes with the cellular complexes of `N/G` (with local coefficients) need a
`G`-CW structure and local systems. Borel cohomology, the `G`-homotopy index, flows and the
example sections are out of scope.
