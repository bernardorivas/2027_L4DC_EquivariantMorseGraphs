import LatentEquivariant.Latent.Pullback

/-!
# The quotient pseudometric on an orbit space

For a compact group `G` acting continuously by isometries on a compact metric latent space `Z`,
the orbit space `Z/G` is a compact metric space with `d̄([a],[b]) = min_h d(a, h b)`, and for an
equivariant latent map Proposition `prop:equivariant_pullback` (ii) follows from Theorem
`thm:main_simple` on `Z/G`. `Latent/Pullback.lean` defines `d̄` (`quotientDist`) and the
tolerance measured with it (`quotientTolerance`) directly. This file shows that `d̄` is a
pseudometric that induces the quotient topology, so that `quotientTolerance` is the tolerance of
the pseudometric space `Z/G`, and it carries out that proof of (ii) on `Z/G`.

Throughout, `G` is an arbitrary group acting by isometries on a pseudometric space `Z`,
`π : Z → Z/G` is the orbit map `orbitMap G Z`, and `d̄` is defined by an infimum.

## Main statements

* `quotientDist_self`, `quotientDist_comm`, `quotientDist_triangle`, `quotientDist_nonneg`: `d̄` is
  a pseudometric.
* `isOpen_iff_quotientDist`, `quotient_topology_eq`: the quotient topology is the topology of `d̄`.
  Hence the instance `instPseudoMetricSpaceOrbitRelQuotient`: `Z/G` is a pseudometric space with
  distance `d̄` whose topology is the quotient topology, definitionally
  (`toTopologicalSpace_eq_instTopologicalSpaceQuotient`). Along the way,
  `preimage_orbitMap_ball` (`π⁻¹(B̄_ε(π z)) = ⋃_h B_ε(h z)`), `image_orbitMap_ball`
  (`π(B_ε(z)) = B̄_ε(π z)`) and `infDist_orbitMap_image` (`d̄(π z, π(A)) = d(z, A)` for invariant
  `A`, the first step of the paper's proof).
* `quotientDist_eq_zero_iff`, `forall_isClosed_orbit_iff`, `t0Space_iff_isClosed_orbit`: `d̄`
  separates points if and only if every orbit is closed. Then `quotientMetricSpace` is a metric
  space structure on `Z/G` over the same pseudometric structure. Orbits are closed for a compact
  group acting continuously on a Hausdorff space (`isClosed_orbit_of_compactSpace`) and for a
  finite group acting on a T1 space (`isClosed_orbit_of_finite`); for metric `Z`, `Z/G` is then a
  metric space by instance. The finite case is a special case of the compact one (a finite group
  is compact in the discrete topology, and every action of a discrete group is continuous); that
  `G` carries no topology is a feature of the encoding, not a weakening. `isLeast_orbitDist`,
  `isLeast_residualMod`: for a compact group acting continuously, the infima defining `d̄` and `r̄`
  are the minima of the paper. These proofs, and `isClosed_orbit_of_compactSpace`, use
  `ContinuousSMul G Z` only through the continuity of the orbit maps `h ↦ h • z`.
* `thickening_eq_quotientThickening`, `quotientTolerance_eq_tolerance`, `tolerance_quotientMap`,
  `IsAttractingBlock.quotient_and_tolerance_eq`: Proposition `prop:equivariant_pullback` (i) with
  the tolerance of the pseudometric space `Z/G`.
* `dist_quotientMap_comp_orbitMap`, `preimage_comp_orbitMap_image`,
  `IsGInvariant.isClosed_image_orbitMap`: the residual of the encoder `π ∘ E` for `ḡ` is `r̄`,
  `(π ∘ E)⁻¹(N/G) = E⁻¹(N)`, and `N/G` is closed. With these,
  `IsEquivariant.isAttractingBlock_preimage_of_residualMod_of_isClosed` and
  `IsEquivariant.isAttractingBlock_preimage_of_residualMod` prove Proposition
  `prop:equivariant_pullback` (ii) as the paper does, by Theorem `thm:main_simple` on `Z/G`.
* `compactSpace_orbitRelQuotient`, `isCompactMetric_of_isClosed_orbit`,
  `isCompactMetric_orbitRelQuotient`: for compact `Z` and closed orbits, `Z/G` is a compact metric
  space with metric `d̄` and the quotient topology, in particular for a compact group acting
  continuously by isometries on a compact metric space.

## Design

`instPseudoMetricSpaceOrbitRelQuotient` is a global instance, so that statements about `Z/G` use
`dist`, `Metric.thickening` and `tolerance` directly. This is safe for two reasons. First, Mathlib
has no `Dist`, extended distance, uniform or bornology instance on `Quotient s` or on
`MulAction.orbitRel.Quotient G Z`. The quotients that carry a metric are by other setoids
(`SeparationQuotient`, the additive `AddCircle`), except `M ⧸ S` for a subgroup `S` of a
multiplicative seminormed commutative group, whose setoid `QuotientGroup.leftRel S` unfolds to
`orbitRel S.op M`. Instance search does not reach this instance on `M ⧸ S`: it would need
`IsIsometricSMul S.op M`, which Mathlib does not provide, and in a test with such a local instance
it still returned Mathlib's quotient norm structure. Second, the instance is built with
`PseudoMetricSpace.replaceTopology`, so its topology is the existing quotient topology
`instTopologicalSpaceQuotient` up to unfolding instances and reducible definitions. So there is no
topology diamond, and the lemmas of `Orbit/Quotient.lean` and `Latent/Pullback.lean` about the
quotient topology (`continuous_orbitMap`, `IsAttractingBlock.quotient`) apply unchanged. A metric
structure needs a proof that orbits are closed, so `quotientMetricSpace` is a reducible definition,
to be used with `letI`. The instances `instMetricSpaceOrbitRelQuotientOfCompactSpace` and
`instMetricSpaceOrbitRelQuotientOfFinite` apply it for a compact group acting continuously and
for a finite group. Their pseudometric structure is the global one, and they agree with each
other, so they create no diamond.

## Not formalized

The remarks after the proof of Proposition `prop:equivariant_pullback` (the rotation of `S²` with a
wrong frequency, the size of the tolerance of grid blocks, the height on `S²` as an encoder with
zero residual) are not formalized. The paper's `min` is an infimum here; `isLeast_orbitDist` shows
that it is attained in the paper's setting, and no statement depends on it.
-/

namespace LatentEquivariant

open Set Topology MulAction
open scoped ENNReal

variable {G X Z : Type*} [Group G]

/-! ### Orbits and the orbit map -/

section Orbit

variable [MulAction G Z]

/-- For a `G`-invariant set `N`, `(π ∘ E)⁻¹(π(N)) = E⁻¹(N)`, a step of the proof of
Proposition `prop:equivariant_pullback` (ii). -/
theorem preimage_comp_orbitMap_image (E : X → Z) {N : Set Z} (hN : IsGInvariant G N) :
    (orbitMap G Z ∘ E) ⁻¹' (orbitMap G Z '' N) = E ⁻¹' N := by
  rw [preimage_comp, hN.preimage_image_orbitMap]

/-- The image `N/G = π(N)` of a closed `G`-invariant set is closed in the quotient topology, a
step of the proof of Proposition `prop:equivariant_pullback` (ii). -/
theorem IsGInvariant.isClosed_image_orbitMap [TopologicalSpace Z] {N : Set Z}
    (hNG : IsGInvariant G N) (hN : IsClosed N) : IsClosed (orbitMap G Z '' N) := by
  have hpre : IsClosed (orbitMap G Z ⁻¹' (orbitMap G Z '' N)) := by
    rwa [hNG.preimage_image_orbitMap]
  exact (isQuotientMap_quotient_mk' (s := orbitRel G Z)).isCoinducing.isClosed_preimage.mp hpre

/-- The orbits of a compact group acting continuously on a Hausdorff space are closed, the reason
why `d̄` is then a metric. -/
theorem isClosed_orbit_of_compactSpace [TopologicalSpace Z] [T2Space Z] [TopologicalSpace G]
    [CompactSpace G] [ContinuousSMul G Z] (z : Z) : IsClosed (orbit G z) :=
  (isCompact_range (continuous_id.smul continuous_const)).isClosed

/-- The orbits of a finite group acting on a T1 space are closed (for the
finite groups of CMGDB). -/
theorem isClosed_orbit_of_finite [TopologicalSpace Z] [T1Space Z] [Finite G] (z : Z) :
    IsClosed (orbit G z) :=
  (finite_range fun g : G ↦ g • z).isClosed

/-- `Z/G` is compact when `Z` is, for any action
(Mathlib's `Quotient.compactSpace`). -/
theorem compactSpace_orbitRelQuotient [TopologicalSpace Z] [CompactSpace Z] :
    CompactSpace (orbitRel.Quotient G Z) :=
  inferInstance

end Orbit

/-! ### The orbit distance -/

section OrbitDist

variable [PseudoMetricSpace Z] [MulAction G Z]

/-- The orbit distance `inf_h d(a, h b)` is nonnegative. -/
theorem orbitDist_nonneg (a b : Z) : 0 ≤ orbitDist G a b :=
  le_ciInf fun _ ↦ dist_nonneg

/-- `inf_h d(a, h b) ≤ d(a, k b)` for every `k ∈ G`. -/
theorem orbitDist_le_dist_smul (a b : Z) (k : G) : orbitDist G a b ≤ dist a (k • b) :=
  ciInf_le (bddBelow_range_dist a b) k

/-- `inf_h d(a, h b) ≤ d(a, b)`. -/
theorem orbitDist_le_dist (a b : Z) : orbitDist G a b ≤ dist a b := by
  simpa using orbitDist_le_dist_smul (G := G) a b 1

/-- `inf_h d(a, h b) < ε` if and only if `d(a, h b) < ε` for some `h ∈ G`. -/
theorem orbitDist_lt_iff {a b : Z} {ε : ℝ} :
    orbitDist G a b < ε ↔ ∃ h : G, dist a (h • b) < ε :=
  ciInf_lt_iff (bddBelow_range_dist a b)

/-- `inf_h d(a, h a) = 0`. -/
theorem orbitDist_self (a : Z) : orbitDist G a a = 0 :=
  le_antisymm ((orbitDist_le_dist a a).trans_eq (dist_self a)) (orbitDist_nonneg a a)

/-- `inf_h d(a, h b) = 0` if and only if `a` lies in the closure of the orbit of `b`. -/
theorem orbitDist_eq_zero_iff {a b : Z} : orbitDist G a b = 0 ↔ a ∈ closure (orbit G b) := by
  rw [Metric.mem_closure_iff]
  constructor
  · intro h ε hε
    obtain ⟨k, hk⟩ := orbitDist_lt_iff.mp (h.trans_lt hε)
    exact ⟨k • b, mem_orbit b k, hk⟩
  · intro h
    refine le_antisymm (le_of_forall_pos_lt_add fun ε hε ↦ ?_) (orbitDist_nonneg a b)
    obtain ⟨c, hc, hac⟩ := h ε hε
    obtain ⟨k, rfl⟩ := mem_orbit_iff.mp hc
    rw [zero_add]
    exact (orbitDist_le_dist_smul a b k).trans_lt hac

/-- For a compact group acting continuously, the infimum `inf_h d(a, h b)` is attained: it is
the minimum `min_h d(a, h b)`. -/
theorem isLeast_orbitDist [TopologicalSpace G] [CompactSpace G] [ContinuousSMul G Z] (a b : Z) :
    IsLeast (range fun h : G ↦ dist a (h • b)) (orbitDist G a b) := by
  have hc : Continuous fun h : G ↦ dist a (h • b) :=
    continuous_const.dist (continuous_id.smul continuous_const)
  obtain ⟨h₀, -, hmin⟩ := isCompact_univ.exists_isMinOn univ_nonempty hc.continuousOn
  have heq : orbitDist G a b = dist a (h₀ • b) :=
    le_antisymm (orbitDist_le_dist_smul a b h₀)
      (le_ciInf fun h ↦ isMinOn_iff.mp hmin h (mem_univ h))
  exact ⟨⟨h₀, heq.symm⟩, by rintro _ ⟨k, rfl⟩; exact orbitDist_le_dist_smul a b k⟩

/-- For a compact group acting continuously, the residual modulo `G` is the
minimum `r̄(x) = min_h d(g(E x), h E(f x))`. -/
theorem isLeast_residualMod [TopologicalSpace G] [CompactSpace G] [ContinuousSMul G Z]
    (f : X → X) (g : Z → Z) (E : X → Z) (x : X) :
    IsLeast (range fun h : G ↦ dist (g (E x)) (h • E (f x))) (residualMod G f g E x) :=
  isLeast_orbitDist _ _

end OrbitDist

/-! ### The quotient pseudometric and the quotient topology -/

section Pseudometric

variable [PseudoMetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z]

/-- For an isometric action, `inf_h d(a, h b) = inf_h d(b, h a)`. -/
theorem orbitDist_comm (a b : Z) : orbitDist G a b = orbitDist G b a := by
  unfold orbitDist
  calc ⨅ h : G, dist a (h • b) = ⨅ h : G, dist a (h⁻¹ • b) :=
        (inv_surjective.iInf_comp fun h : G ↦ dist a (h • b)).symm
    _ = ⨅ h : G, dist b (h • a) :=
        iInf_congr fun h ↦ by rw [dist_comm b, dist_smul_left_eq]

/-- For an isometric action, the orbit distance satisfies the triangle inequality. -/
theorem orbitDist_triangle (a b c : Z) :
    orbitDist G a c ≤ orbitDist G a b + orbitDist G b c := by
  have key : ∀ h k : G, orbitDist G a c ≤ dist a (h • b) + dist b (k • c) := fun h k ↦
    calc orbitDist G a c ≤ dist a ((h * k) • c) := orbitDist_le_dist_smul a c (h * k)
      _ ≤ dist a (h • b) + dist (h • b) ((h * k) • c) := dist_triangle _ _ _
      _ = dist a (h • b) + dist b (k • c) := by rw [mul_smul, dist_smul]
  have h1 : ∀ k : G, orbitDist G a c - dist b (k • c) ≤ orbitDist G a b := fun k ↦
    le_ciInf fun h ↦ by linarith [key h k]
  have h2 : orbitDist G a c - orbitDist G a b ≤ orbitDist G b c :=
    le_ciInf fun k ↦ by linarith [h1 k]
  linarith

/-- `d̄ ≥ 0`. -/
theorem quotientDist_nonneg (p q : orbitRel.Quotient G Z) : 0 ≤ quotientDist G p q := by
  obtain ⟨a, rfl⟩ := orbitMap_surjective p
  obtain ⟨b, rfl⟩ := orbitMap_surjective q
  exact orbitDist_nonneg a b

/-- `d̄(p, p) = 0`. -/
theorem quotientDist_self (p : orbitRel.Quotient G Z) : quotientDist G p p = 0 := by
  obtain ⟨a, rfl⟩ := orbitMap_surjective p
  exact orbitDist_self a

/-- `d̄` is symmetric. -/
theorem quotientDist_comm (p q : orbitRel.Quotient G Z) :
    quotientDist G p q = quotientDist G q p := by
  obtain ⟨a, rfl⟩ := orbitMap_surjective p
  obtain ⟨b, rfl⟩ := orbitMap_surjective q
  exact orbitDist_comm a b

/-- `d̄` satisfies the triangle inequality. -/
theorem quotientDist_triangle (p q r : orbitRel.Quotient G Z) :
    quotientDist G p r ≤ quotientDist G p q + quotientDist G q r := by
  obtain ⟨a, rfl⟩ := orbitMap_surjective p
  obtain ⟨b, rfl⟩ := orbitMap_surjective q
  obtain ⟨c, rfl⟩ := orbitMap_surjective r
  exact orbitDist_triangle a b c

variable (G Z) in
/-- The pseudometric structure of `d̄` on `Z/G` with the uniform structure,
and so the topology, generated by `d̄`. It is auxiliary: `instPseudoMetricSpaceOrbitRelQuotient`
replaces its topology by the quotient topology, which is the same topology by
`quotient_topology_eq`. -/
noncomputable abbrev quotientPseudoMetricAux : PseudoMetricSpace (orbitRel.Quotient G Z) where
  dist := quotientDist G
  dist_self := quotientDist_self
  dist_comm := quotientDist_comm
  dist_triangle := quotientDist_triangle

/-- The quotient topology of `Z/G` is the topology of `d̄`: a set is open in the quotient topology
if and only if it contains a `d̄`-ball around each of its points. The proof of Proposition
`prop:equivariant_pullback` (i) uses this without proof. If `s` is open, then `π⁻¹(s)` is open
(and `G`-invariant), so it contains a ball
`B_ε(z)`, whose image `π(B_ε(z)) = B̄_ε(π z)` lies in `s`; the converse holds because
`d̄(π z, π w) ≤ d(z, w)`. -/
theorem isOpen_iff_quotientDist {s : Set (orbitRel.Quotient G Z)} :
    IsOpen s ↔ ∀ p ∈ s, ∃ ε > 0, ∀ q, quotientDist G p q < ε → q ∈ s := by
  constructor
  · intro hs p hp
    obtain ⟨z, rfl⟩ := orbitMap_surjective p
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.mp (hs.preimage continuous_orbitMap) z hp
    refine ⟨ε, hε, fun q hq ↦ ?_⟩
    obtain ⟨w, rfl⟩ := orbitMap_surjective q
    obtain ⟨h, hh⟩ := orbitDist_lt_iff.mp hq
    have hmem : h • w ∈ orbitMap G Z ⁻¹' s := hball (Metric.mem_ball'.mpr hh)
    rwa [mem_preimage, orbitMap_smul] at hmem
  · intro hs
    refine (isQuotientMap_quotient_mk' (s := orbitRel G Z)).isCoinducing.isOpen_preimage.mp ?_
    refine Metric.isOpen_iff.mpr fun z hz ↦ ?_
    obtain ⟨ε, hε, hball⟩ := hs (orbitMap G Z z) hz
    exact ⟨ε, hε, fun w hw ↦
      hball (orbitMap G Z w) ((orbitDist_le_dist z w).trans_lt (Metric.mem_ball'.mp hw))⟩

variable (G Z) in
/-- The quotient topology of `Z/G` equals the topology generated by `d̄`. The proof of
Proposition `prop:equivariant_pullback` (i) uses this without proof. -/
theorem quotient_topology_eq :
    (instTopologicalSpaceQuotient : TopologicalSpace (orbitRel.Quotient G Z)) =
      (quotientPseudoMetricAux G Z).toUniformSpace.toTopologicalSpace := by
  refine TopologicalSpace.ext_iff.mpr fun s ↦ ?_
  rw [isOpen_iff_quotientDist, @Metric.isOpen_iff _ (quotientPseudoMetricAux G Z)]
  simp only [subset_def, @Metric.mem_ball' _ (quotientPseudoMetricAux G Z)]
  rfl

/-- `Z/G` is a pseudometric space with distance
`d̄([a],[b]) = inf_h d(a, h b)` (`quotientDist`), for any group acting by isometries. Its topology
is the quotient topology `instTopologicalSpaceQuotient`, definitionally: the structure is
`quotientPseudoMetricAux` with its topology replaced along `quotient_topology_eq`. -/
noncomputable instance instPseudoMetricSpaceOrbitRelQuotient :
    PseudoMetricSpace (orbitRel.Quotient G Z) :=
  (quotientPseudoMetricAux G Z).replaceTopology (quotient_topology_eq G Z)

/-- The distance of `Z/G` is `d̄`, by definition. -/
theorem dist_eq_quotientDist (p q : orbitRel.Quotient G Z) : dist p q = quotientDist G p q :=
  rfl

/-- `dist([a],[b]) = inf_h d(a, h b)`, by definition. -/
@[simp] theorem dist_orbitMap (a b : Z) :
    dist (orbitMap G Z a) (orbitMap G Z b) = orbitDist G a b :=
  rfl

/-- The topology of the pseudometric space `Z/G` is the quotient topology, by definition. -/
theorem toTopologicalSpace_eq_instTopologicalSpaceQuotient :
    (PseudoMetricSpace.toUniformSpace : UniformSpace (orbitRel.Quotient G Z)).toTopologicalSpace =
      instTopologicalSpaceQuotient :=
  rfl

/- The two topologies agree up to unfolding instances and reducible definitions, so instance
arguments unify. -/
example : (PseudoMetricSpace.toUniformSpace :
    UniformSpace (orbitRel.Quotient G Z)).toTopologicalSpace = instTopologicalSpaceQuotient := by
  with_reducible_and_instances rfl

/-- `π⁻¹(B̄_ε(π z)) = ⋃_h B_ε(h z)`. -/
theorem preimage_orbitMap_ball (z : Z) (ε : ℝ) :
    orbitMap G Z ⁻¹' Metric.ball (orbitMap G Z z) ε = ⋃ h : G, Metric.ball (h • z) ε := by
  ext w
  simp only [mem_preimage, Metric.mem_ball, mem_iUnion, dist_orbitMap]
  exact orbitDist_lt_iff

/-- `π(B_ε(z)) = B̄_ε(π z)`. -/
theorem image_orbitMap_ball (z : Z) (ε : ℝ) :
    orbitMap G Z '' Metric.ball z ε = Metric.ball (orbitMap G Z z) ε := by
  ext q
  obtain ⟨w, rfl⟩ := orbitMap_surjective q
  constructor
  · rintro ⟨v, hv, hvw⟩
    rw [← hvw, Metric.mem_ball, dist_orbitMap]
    exact (orbitDist_le_dist v z).trans_lt hv
  · intro hw
    rw [Metric.mem_ball, dist_orbitMap] at hw
    obtain ⟨h, hh⟩ := orbitDist_lt_iff.mp hw
    refine ⟨h⁻¹ • w, ?_, orbitMap_smul _ _⟩
    rw [Metric.mem_ball, dist_smul_left_eq, inv_inv]
    exact hh

/-- The first step of the proof of Proposition `prop:equivariant_pullback`: for a `G`-invariant
set `A`, `d̄([z], π(A)) = d(z, A)`. -/
theorem infDist_orbitMap_image {A : Set Z} (hA : IsGInvariant G A) (z : Z) :
    Metric.infDist (orbitMap G Z z) (orbitMap G Z '' A) = Metric.infDist z A := by
  rcases A.eq_empty_or_nonempty with rfl | hne
  · simp
  refine eq_of_forall_gt_iff fun r ↦ ?_
  rw [Metric.infDist_lt_iff (hne.image _), Metric.infDist_lt_iff hne, exists_mem_image]
  constructor
  · rintro ⟨a, ha, hlt⟩
    obtain ⟨h, hh⟩ := orbitDist_lt_iff.mp hlt
    exact ⟨h • a, hA h a ha, hh⟩
  · rintro ⟨a, ha, hlt⟩
    exact ⟨a, ha, (orbitDist_le_dist z a).trans_lt hlt⟩

end Pseudometric

/-! ### Closed orbits and the quotient metric -/

section Metric

variable [PseudoMetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z]

/-- If every orbit is closed, then `d̄(p, q) = 0` only for `p = q`: `d̄` is a metric. -/
theorem quotientDist_eq_zero_iff (hG : ∀ z : Z, IsClosed (orbit G z))
    {p q : orbitRel.Quotient G Z} : quotientDist G p q = 0 ↔ p = q := by
  obtain ⟨a, rfl⟩ := orbitMap_surjective p
  obtain ⟨b, rfl⟩ := orbitMap_surjective q
  rw [quotientDist_orbitMap, orbitDist_eq_zero_iff, (hG b).closure_eq, mem_orbit_iff,
    orbitMap_eq_iff]

/-- `d̄` separates points if and only if every orbit is closed: closed orbits are exactly what
is needed for `d̄` to be a metric. -/
theorem forall_isClosed_orbit_iff :
    (∀ z : Z, IsClosed (orbit G z)) ↔
      ∀ p q : orbitRel.Quotient G Z, quotientDist G p q = 0 → p = q := by
  refine ⟨fun hG p q ↦ (quotientDist_eq_zero_iff hG).mp, fun h z ↦
    isClosed_of_closure_subset fun a ha ↦ ?_⟩
  exact mem_orbit_iff.mpr (orbitMap_eq_iff.mp
    (h (orbitMap G Z a) (orbitMap G Z z) (orbitDist_eq_zero_iff.mpr ha)))

/-- The quotient topology of `Z/G` is T0 (for a pseudometric space, equivalently, `d̄` is a
metric) if and only if every orbit is closed (for an irrational rotation of
`S²`, `Z/G` is not T0). -/
theorem t0Space_iff_isClosed_orbit :
    T0Space (orbitRel.Quotient G Z) ↔ ∀ z : Z, IsClosed (orbit G z) := by
  rw [t0Space_iff_inseparable, forall_isClosed_orbit_iff]
  simp only [Metric.inseparable_iff, dist_eq_quotientDist]

variable (G Z) in
/-- If every orbit is closed, then `Z/G` with `d̄` is a metric space. Its
pseudometric structure, and so its topology, is that of `instPseudoMetricSpaceOrbitRelQuotient`
(`quotientMetricSpace_toPseudoMetricSpace`). A reducible definition, for `letI`. -/
noncomputable abbrev quotientMetricSpace (hG : ∀ z : Z, IsClosed (orbit G z)) :
    MetricSpace (orbitRel.Quotient G Z) where
  toPseudoMetricSpace := instPseudoMetricSpaceOrbitRelQuotient
  eq_of_dist_eq_zero h := (quotientDist_eq_zero_iff hG).mp h

/-- The metric structure `quotientMetricSpace` extends the pseudometric structure
`instPseudoMetricSpaceOrbitRelQuotient`, by definition. -/
theorem quotientMetricSpace_toPseudoMetricSpace (hG : ∀ z : Z, IsClosed (orbit G z)) :
    (quotientMetricSpace G Z hG).toPseudoMetricSpace = instPseudoMetricSpaceOrbitRelQuotient :=
  rfl

/-- With closed orbits in place of a compact group: for compact `Z`, `Z/G`
is compact and `d̄` separates points, so `Z/G` is a compact metric space (`quotientMetricSpace`)
with the quotient topology. -/
theorem isCompactMetric_of_isClosed_orbit [CompactSpace Z] (hG : ∀ z : Z, IsClosed (orbit G z)) :
    CompactSpace (orbitRel.Quotient G Z) ∧
      ∀ p q : orbitRel.Quotient G Z, dist p q = 0 → p = q :=
  ⟨inferInstance, fun _ _ ↦ (quotientDist_eq_zero_iff hG).mp⟩

/- Instance check: with closed orbits and `Z` compact, `quotientMetricSpace` makes `Z/G` a compact
metric space, and so a Hausdorff space for the quotient topology. -/
example [CompactSpace Z] (hG : ∀ z : Z, IsClosed (orbit G z)) :
    letI := quotientMetricSpace G Z hG
    CompactSpace (orbitRel.Quotient G Z) ∧ T2Space (orbitRel.Quotient G Z) :=
  letI := quotientMetricSpace G Z hG
  ⟨inferInstance, inferInstance⟩

end Metric

section MetricInstances

variable [MetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z]

/-- For a compact group acting continuously by isometries on a metric space,
`Z/G` with `d̄` is a metric space. -/
noncomputable instance instMetricSpaceOrbitRelQuotientOfCompactSpace [TopologicalSpace G]
    [CompactSpace G] [ContinuousSMul G Z] : MetricSpace (orbitRel.Quotient G Z) :=
  quotientMetricSpace G Z isClosed_orbit_of_compactSpace

/-- For a finite group acting by isometries on a metric space, `Z/G` with `d̄` is a metric
space (for the finite groups of CMGDB). -/
noncomputable instance instMetricSpaceOrbitRelQuotientOfFinite [Finite G] :
    MetricSpace (orbitRel.Quotient G Z) :=
  quotientMetricSpace G Z isClosed_orbit_of_finite

/- The metric instances extend the global pseudometric instance, and agree with each other, up
to unfolding instances and reducible definitions. -/
example [TopologicalSpace G] [CompactSpace G] [ContinuousSMul G Z] :
    (MetricSpace.toPseudoMetricSpace : PseudoMetricSpace (orbitRel.Quotient G Z)) =
      instPseudoMetricSpaceOrbitRelQuotient := by
  with_reducible_and_instances rfl

example [Finite G] [TopologicalSpace G] [CompactSpace G] [ContinuousSMul G Z] :
    instMetricSpaceOrbitRelQuotientOfFinite =
      instMetricSpaceOrbitRelQuotientOfCompactSpace (G := G) (Z := Z) := by
  with_reducible_and_instances rfl

/- Instance check: for a finite group acting by isometries on a compact metric space, `Z/G` is a
compact metric space. -/
example [Finite G] [CompactSpace Z] :
    CompactSpace (orbitRel.Quotient G Z) ∧ T2Space (orbitRel.Quotient G Z) :=
  ⟨inferInstance, inferInstance⟩

/-- For a compact group acting continuously by isometries on a compact
metric space `Z`, `Z/G` is a compact metric space (by the instances
`instMetricSpaceOrbitRelQuotientOfCompactSpace` and `Quotient.compactSpace`) whose metric is
`d̄([a],[b]) = min_h d(a, h b)` and whose topology is the quotient topology. -/
theorem isCompactMetric_orbitRelQuotient [CompactSpace Z] [TopologicalSpace G] [CompactSpace G]
    [ContinuousSMul G Z] :
    CompactSpace (orbitRel.Quotient G Z) ∧
      (∀ a b : Z,
        IsLeast (range fun h : G ↦ dist a (h • b)) (dist (orbitMap G Z a) (orbitMap G Z b))) ∧
      ∀ p q : orbitRel.Quotient G Z, dist p q = 0 → p = q :=
  ⟨inferInstance, isLeast_orbitDist, fun _ _ ↦ eq_of_dist_eq_zero⟩

end MetricInstances

/-! ### Proposition `prop:equivariant_pullback` (i): the tolerance on `Z/G` -/

section Tolerance

variable [PseudoMetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z]

/-- `quotientThickening` is the open `ν`-neighborhood of the pseudometric space `Z/G`
(Proposition `prop:equivariant_pullback` (i)). -/
theorem thickening_eq_quotientThickening (ν : ℝ) (A : Set (orbitRel.Quotient G Z)) :
    Metric.thickening ν A = quotientThickening G ν A := by
  ext q
  rw [Metric.mem_thickening_iff]
  rfl

/-- `quotientTolerance` is the tolerance `τ` of the pseudometric space `Z/G` (Proposition
`prop:equivariant_pullback` (i)). -/
theorem quotientTolerance_eq_tolerance (gbar : orbitRel.Quotient G Z → orbitRel.Quotient G Z)
    (Abar : Set (orbitRel.Quotient G Z)) :
    quotientTolerance G gbar Abar = tolerance gbar Abar := by
  simp only [quotientTolerance, tolerance, thickening_eq_quotientThickening]

/-- The proof of Proposition `prop:equivariant_pullback` (i): `B_ν(A) = π⁻¹(B̄_ν(π(A)))` for a
`G`-invariant set `A`, with `B̄_ν` the open `ν`-neighborhood of `Z/G`. -/
theorem preimage_orbitMap_thickening {A : Set Z} (hA : IsGInvariant G A) (ν : ℝ) :
    orbitMap G Z ⁻¹' Metric.thickening ν (orbitMap G Z '' A) = Metric.thickening ν A := by
  rw [thickening_eq_quotientThickening, preimage_quotientThickening hA]

/-- Proposition `prop:equivariant_pullback` (i), tolerance, with the tolerance of the pseudometric
space `Z/G`: `τ_ḡ(N/G) = τ_g(N)` for equivariant `g` and `G`-invariant `N`. -/
theorem tolerance_quotientMap {g : Z → Z} (hg : IsEquivariant G g) {N : Set Z}
    (hN : IsGInvariant G N) :
    tolerance (quotientMap g hg) (orbitMap G Z '' N) = tolerance g N := by
  rw [← quotientTolerance_eq_tolerance, quotientTolerance_eq hg hN]

/-- Proposition `prop:equivariant_pullback` (i) on the pseudometric space `Z/G`: if `N` is a
`G`-invariant attracting block for the equivariant map `g`, then `N/G` is an attracting block for
`ḡ` and `τ_ḡ(N/G) = τ_g(N)`. -/
theorem IsAttractingBlock.quotient_and_tolerance_eq {g : Z → Z} (hg : IsEquivariant G g)
    {N : Set Z} (hN : IsGInvariant G N) (h : IsAttractingBlock g N) :
    IsAttractingBlock (quotientMap g hg) (orbitMap G Z '' N) ∧
      tolerance (quotientMap g hg) (orbitMap G Z '' N) = tolerance g N :=
  ⟨h.quotient hg hN, tolerance_quotientMap hg hN⟩

end Tolerance

/-! ### Proposition `prop:equivariant_pullback` (ii) through `Z/G` -/

section Pullback

variable [PseudoMetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z] {f : X → X} {g : Z → Z}
  {E : X → Z}

/-- The proof of Proposition `prop:equivariant_pullback` (ii): the residual of `f`, `ḡ` and the
encoder `π ∘ E` is the residual modulo `G`,
`d̄(ḡ(π(E x)), π(E(f x))) = inf_h d(g(E x), h E(f x)) = r̄(x)`. Neither `f` nor `E` needs to be
equivariant. -/
theorem dist_quotientMap_comp_orbitMap (hg : IsEquivariant G g) (x : X) :
    dist (quotientMap g hg ((orbitMap G Z ∘ E) x)) ((orbitMap G Z ∘ E) (f x)) =
      residualMod G f g E x :=
  rfl

/-- Proposition `prop:equivariant_pullback` (ii), by the paper's proof: Theorem `thm:main_simple`
(`isAttractingBlock_preimage_of_isClosed`) applied on `Z/G` to `f`, `ḡ` and the encoder `π ∘ E`.
If `N` is closed and `G`-invariant and `r̄(x) < τ_g(N)` for every `x ∈ E⁻¹(N)`, then `E⁻¹(N)` is
an attracting block for `f`. `isAttractingBlock_preimage_of_residualMod_of_isClosed` proves the
same conclusion in `Z`, without equivariance of `g` and for any action by homeomorphisms. In the
paper's setting, `Z` compact and `E⁻¹(N)` nonempty, "`N` closed" is no weaker than "`N` an
attracting block": the bound makes `τ_g(N) > 0`, so `g(N) ⊆ int N`
(`image_subset_interior_of_tolerance_pos`), and a closed subset of a compact space is compact. -/
theorem IsEquivariant.isAttractingBlock_preimage_of_residualMod_of_isClosed [TopologicalSpace X]
    [CompactSpace X] (hg : IsEquivariant G g) (hE : Continuous E) {N : Set Z} (hN : IsClosed N)
    (hNG : IsGInvariant G N)
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) := by
  have key := isAttractingBlock_preimage_of_isClosed (f := f) (g := quotientMap g hg)
    (continuous_orbitMap.comp hE) (hNG.isClosed_image_orbitMap hN) fun x hx ↦ by
      rw [preimage_comp_orbitMap_image E hNG] at hx
      rw [dist_quotientMap_comp_orbitMap hg x, tolerance_quotientMap hg hNG]
      exact hres x hx
  rwa [preimage_comp_orbitMap_image E hNG] at key

end Pullback

section PullbackPaper

variable [MetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z] [TopologicalSpace G]
  [CompactSpace G] [ContinuousSMul G Z] {f : X → X} {g : Z → Z} {E : X → Z}

/-- Proposition `prop:equivariant_pullback` (ii), literally by the paper's proof: for a compact
group acting continuously by isometries, `Z/G` is a metric space, (i) makes `N/G` an attracting
block for `ḡ` with `τ_ḡ(N/G) = τ_g(N)`, and Theorem `thm:main_simple`
(`isAttractingBlock_preimage`) on `Z/G` gives that `E⁻¹(N) = (π ∘ E)⁻¹(N/G)` is an attracting
block for `f` if `sup_{x ∈ E⁻¹(N)} r̄(x) < τ_g(N)`. Neither the compactness of `G` nor the
metric on `Z` is needed: `IsEquivariant.isAttractingBlock_preimage_of_residualMod_of_isClosed`
assumes `N` closed instead of an attracting block, and a pointwise bound. See
`isAttractingBlock_preimage_of_residualMod` for the proof in `Z`, without equivariance of `g`. -/
theorem IsEquivariant.isAttractingBlock_preimage_of_residualMod [TopologicalSpace X]
    [CompactSpace X] (hg : IsEquivariant G g) (hE : Continuous E) {N : Set Z}
    (hN : IsAttractingBlock g N) (hNG : IsGInvariant G N)
    (hres : ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) := by
  have key := isAttractingBlock_preimage (f := f) (continuous_orbitMap.comp hE)
    (hN.quotient hg hNG) (by
      rw [preimage_comp_orbitMap_image E hNG, tolerance_quotientMap hg hNG]
      exact hres)
  rwa [preimage_comp_orbitMap_image E hNG] at key

end PullbackPaper

end LatentEquivariant
