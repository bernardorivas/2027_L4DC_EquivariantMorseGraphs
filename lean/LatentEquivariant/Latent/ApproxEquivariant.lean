import LatentEquivariant.Latent.Pullback

/-!
# Approximately equivariant latent maps

The draft lemma formalized here concerns latent maps that are approximately equivariant.
A group `G` acts on the latent space `Z`, `f : Z → Z` is the latent map, `E : X → Z` the
encoder, and `F : X → X` the true map. The latent map need not be equivariant. Its defect on `N`
is `δ_N = sup_{z ∈ N} sup_{g ∈ G} d(f(g z), g f(z))` (`equivDefect`), and `G N = ⋃_g g N` is the
saturation of `N`.

* `tolerance_div_sub_equivDefect_le`: if every `g` acts by an `L`-Lipschitz map with `L ≠ 0`,
  then `τ_f(G N) ≥ τ_f(N) / L - δ_N`. For an isometric action this is
  `tolerance_sub_equivDefect_le`, `τ_f(G N) ≥ τ_f(N) - δ_N`. No hypothesis on `f`, `N` or `G`.
* `isAttractingBlock_saturation`: for finite `G`, compact `N` and `δ_N < τ_f(N)`, the saturation
  `G N` is an attracting block for `f`. It is `G`-invariant for every `N`
  (`isGInvariant_saturation`). `isAttractingBlock_saturation_of_lipschitz` is the Lipschitz form,
  with `δ_N < τ_f(N) / L`.
* `residualMod_smul_le`: for equivariant `E` and `F` and an isometric action,
  `r̄(g x) ≤ r̄(x) + δ_N` whenever `E x ∈ N`.
* `isAttractingBlock_preimage_saturation`: for equivariant `E` and `F`, the bound
  `sup_{E⁻¹(N)} r̄ < τ_f(N)` already makes `E⁻¹(G N)` an attracting block for `F`. The defect
  does not enter, the action need not be isometric, and `N` need not be `G`-invariant. The draft
  certificate `sup_{E⁻¹(N)} r̄ + 2 δ_N < τ_f(N)` is the weaker
  `isAttractingBlock_preimage_saturation_of_defect`.

So the defect matters for the latent statement that `G N` is an attracting block for `f`, and
not for the certificate upstairs. Compare `isAttractingBlock_preimage_of_residualMod`
(Proposition `prop:equivariant_pullback` (ii)), which needs `N` to be `G`-invariant and no
equivariance of `E` or `F`.
-/

namespace LatentEquivariant

open Set Topology Pointwise MulAction
open scoped ENNReal NNReal

variable {G X Z : Type*} [Group G]

section Saturation

variable [MulAction G Z]

/-- The saturation `G N = ⋃_g g N` of `N`. -/
def saturation (G : Type*) [Group G] [MulAction G Z] (N : Set Z) : Set Z :=
  ⋃ g : G, g • N

theorem mem_saturation {N : Set Z} {w : Z} : w ∈ saturation G N ↔ ∃ g : G, ∃ z ∈ N, g • z = w := by
  simp only [saturation, mem_iUnion, mem_smul_set]

theorem subset_saturation (N : Set Z) : N ⊆ saturation G N :=
  fun z hz ↦ mem_saturation.mpr ⟨1, z, hz, one_smul G z⟩

theorem isGInvariant_saturation (N : Set Z) : IsGInvariant G (saturation G N) := by
  intro g w hw
  obtain ⟨k, z, hz, rfl⟩ := mem_saturation.mp hw
  exact mem_saturation.mpr ⟨g * k, z, hz, mul_smul g k z⟩

theorem IsGInvariant.saturation_eq {N : Set Z} (hN : IsGInvariant G N) : saturation G N = N := by
  refine Subset.antisymm (fun w hw ↦ ?_) (subset_saturation N)
  obtain ⟨g, z, hz, rfl⟩ := mem_saturation.mp hw
  exact hN g z hz

theorem isCompact_saturation [TopologicalSpace Z] [ContinuousConstSMul G Z] [Finite G]
    {N : Set Z} (hN : IsCompact N) : IsCompact (saturation G N) :=
  isCompact_iUnion fun g ↦ hN.smul g

theorem isClosed_saturation [TopologicalSpace Z] [ContinuousConstSMul G Z] [Finite G]
    {N : Set Z} (hN : IsClosed N) : IsClosed (saturation G N) :=
  isClosed_iUnion_of_finite fun g ↦ hN.smul g

end Saturation

section Defect

variable [PseudoMetricSpace Z] [MulAction G Z]

/-- The equivariance defect `δ_N = sup_{z ∈ N} sup_{g ∈ G} d(f(g z), g f(z))` of `f` on `N`. -/
noncomputable def equivDefect (G : Type*) [Group G] [MulAction G Z] (f : Z → Z) (N : Set Z) :
    ℝ≥0∞ :=
  ⨆ z ∈ N, ⨆ g : G, ENNReal.ofReal (dist (f (g • z)) (g • f z))

variable {f : Z → Z} {N : Set Z}

theorem le_equivDefect {z : Z} (hz : z ∈ N) (g : G) :
    ENNReal.ofReal (dist (f (g • z)) (g • f z)) ≤ equivDefect G f N :=
  le_iSup₂_of_le (f := fun z (_ : z ∈ N) ↦ ⨆ g : G, ENNReal.ofReal (dist (f (g • z)) (g • f z)))
    z hz (le_iSup (fun g : G ↦ ENNReal.ofReal (dist (f (g • z)) (g • f z))) g)

theorem dist_le_toReal_equivDefect (hδ : equivDefect G f N ≠ ⊤) {z : Z} (hz : z ∈ N) (g : G) :
    dist (f (g • z)) (g • f z) ≤ (equivDefect G f N).toReal :=
  (ENNReal.ofReal_le_iff_le_toReal hδ).mp (le_equivDefect hz g)

/-- An equivariant map has defect zero. -/
theorem equivDefect_eq_zero (hf : IsEquivariant G f) (N : Set Z) : equivDefect G f N = 0 := by
  simp [equivDefect, hf _ _]

/-- The tolerance of the saturation, for an action by `L`-Lipschitz maps:
`τ_f(G N) ≥ τ_f(N) / L - δ_N`. No hypothesis on `f`, `N` or `G` is needed. -/
theorem tolerance_div_sub_equivDefect_le {L : ℝ≥0} (hL0 : L ≠ 0)
    (hL : ∀ g : G, LipschitzWith L fun z : Z ↦ g • z) (f : Z → Z) (N : Set Z) :
    tolerance f N / L - equivDefect G f N ≤ tolerance f (saturation G N) := by
  rw [tsub_le_iff_right, ENNReal.div_le_iff (ENNReal.coe_ne_zero.mpr hL0) ENNReal.coe_ne_top]
  rcases eq_or_ne (equivDefect G f N) ⊤ with hδ | hδ
  · simp [hδ, ENNReal.top_mul (ENNReal.coe_ne_zero.mpr hL0)]
  set d := (equivDefect G f N).toReal with hd
  have hLpos : (0 : ℝ) < L := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hL0)
  have hdδ : ENNReal.ofReal d = equivDefect G f N := ENNReal.ofReal_toReal hδ
  refine iSup₂_le fun ν ⟨_, hνN⟩ ↦ ?_
  rcases le_or_gt ν (d * L) with hle | hlt
  · calc ENNReal.ofReal ν ≤ ENNReal.ofReal (d * L) := ENNReal.ofReal_le_ofReal hle
      _ = equivDefect G f N * L := by
          rw [ENNReal.ofReal_mul ENNReal.toReal_nonneg, hdδ, ENNReal.ofReal_coe_nnreal]
      _ ≤ _ := by gcongr; exact le_add_self
  set μ := ν / L - d with hμ
  have hμpos : 0 < μ := by
    rw [hμ, sub_pos, lt_div_iff₀ hLpos]
    exact hlt
  have hsub : Metric.thickening μ (f '' saturation G N) ⊆ saturation G N := by
    intro w hw
    obtain ⟨_, ⟨y, hy, rfl⟩, hwy⟩ := Metric.mem_thickening_iff.mp hw
    obtain ⟨g, z, hz, rfl⟩ := mem_saturation.mp hy
    have hclose : dist w (g • f z) < μ + d :=
      calc dist w (g • f z) ≤ dist w (f (g • z)) + dist (f (g • z)) (g • f z) :=
            dist_triangle _ _ _
        _ < μ + d := add_lt_add_of_lt_of_le hwy (dist_le_toReal_equivDefect hδ hz g)
    have hback : dist (g⁻¹ • w) (f z) < ν :=
      calc dist (g⁻¹ • w) (f z) = dist (g⁻¹ • w) (g⁻¹ • g • f z) := by rw [inv_smul_smul]
        _ ≤ L * dist w (g • f z) := (hL g⁻¹).dist_le_mul _ _
        _ < L * (μ + d) := mul_lt_mul_of_pos_left hclose hLpos
        _ = ν := by rw [hμ, sub_add_cancel, mul_div_cancel₀ _ hLpos.ne']
    have hmem : g⁻¹ • w ∈ N :=
      hνN (Metric.mem_thickening_iff.mpr ⟨f z, mem_image_of_mem f hz, hback⟩)
    exact mem_saturation.mpr ⟨g, g⁻¹ • w, hmem, smul_inv_smul g w⟩
  have hτ : ENNReal.ofReal μ ≤ tolerance f (saturation G N) :=
    le_iSup₂_of_le (f := fun (ν : ℝ) (_ : 0 < ν ∧ Metric.thickening ν (f '' saturation G N) ⊆
      saturation G N) ↦ ENNReal.ofReal ν) μ ⟨hμpos, hsub⟩ le_rfl
  calc ENNReal.ofReal ν = (ENNReal.ofReal μ + ENNReal.ofReal d) * L := by
        rw [← ENNReal.ofReal_add hμpos.le ENNReal.toReal_nonneg, ← ENNReal.ofReal_coe_nnreal,
          ← ENNReal.ofReal_mul (by positivity), hμ, sub_add_cancel, div_mul_cancel₀ _ hLpos.ne']
    _ ≤ (tolerance f (saturation G N) + equivDefect G f N) * L := by rw [hdδ]; gcongr

/-- For an action by `L`-Lipschitz maps with `L ≠ 0`, a finite group and a compact `N`: if
`δ_N < τ_f(N) / L`, then `G N` is an attracting block for `f`. -/
theorem isAttractingBlock_saturation_of_lipschitz [Finite G] {L : ℝ≥0} (hL0 : L ≠ 0)
    (hL : ∀ g : G, LipschitzWith L fun z : Z ↦ g • z) (hN : IsCompact N)
    (hδ : equivDefect G f N < tolerance f N / L) : IsAttractingBlock f (saturation G N) := by
  have : ContinuousConstSMul G Z := ⟨fun g ↦ (hL g).continuous⟩
  exact ⟨isCompact_saturation hN, image_subset_interior_of_tolerance_pos <|
    (tsub_pos_of_lt hδ).trans_le (tolerance_div_sub_equivDefect_le hL0 hL f N)⟩

variable [IsIsometricSMul G Z]

/-- The draft lemma, tolerance: for an isometric action, `τ_f(G N) ≥ τ_f(N) - δ_N`. -/
theorem tolerance_sub_equivDefect_le (f : Z → Z) (N : Set Z) :
    tolerance f N - equivDefect G f N ≤ tolerance f (saturation G N) := by
  simpa using tolerance_div_sub_equivDefect_le one_ne_zero
    (fun g : G ↦ (isometry_smul Z g).lipschitz) f N

/-- The draft lemma: for an isometric action of a finite group and a compact `N` with
`δ_N < τ_f(N)`, the saturation `G N` is a `G`-invariant attracting block for `f`, with
`τ_f(G N) ≥ τ_f(N) - δ_N` (`tolerance_sub_equivDefect_le`). -/
theorem isAttractingBlock_saturation [Finite G] (hN : IsCompact N)
    (hδ : equivDefect G f N < tolerance f N) :
    IsAttractingBlock f (saturation G N) ∧ IsGInvariant G (saturation G N) :=
  ⟨⟨isCompact_saturation hN, image_subset_interior_of_tolerance_pos <|
    (tsub_pos_of_lt hδ).trans_le (tolerance_sub_equivDefect_le f N)⟩, isGInvariant_saturation N⟩

end Defect

section Residual

variable [PseudoMetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z] [MulAction G X]
  {F : X → X} {f : Z → Z} {E : X → Z}

omit [IsIsometricSMul G Z] [MulAction G X] in
theorem residualMod_nonneg (x : X) : 0 ≤ residualMod G F f E x :=
  le_ciInf fun _ ↦ dist_nonneg

/-- The draft lemma, residuals: for equivariant `E` and `F`, `r̄(g x) ≤ r̄(x) + δ_N` whenever
`E x ∈ N`. -/
theorem residualMod_smul_le (hF : IsEquivariant G F) (hE : IsEquivariant G E) {N : Set Z}
    {x : X} (hx : E x ∈ N) (g : G) :
    ENNReal.ofReal (residualMod G F f E (g • x)) ≤
      ENNReal.ofReal (residualMod G F f E x) + equivDefect G f N := by
  set D := dist (f (g • E x)) (g • f (E x))
  have key : residualMod G F f E (g • x) ≤ residualMod G F f E x + D := by
    rw [← sub_le_iff_le_add]
    refine le_ciInf fun k ↦ ?_
    rw [sub_le_iff_le_add']
    calc residualMod G F f E (g • x)
        ≤ dist (f (E (g • x))) ((g * k * g⁻¹) • E (F (g • x))) :=
          ciInf_le (bddBelow_range_dist (G := G) _ _) _
      _ = dist (f (g • E x)) (g • k • E (F x)) := by
          rw [hE, hF, hE, mul_smul, mul_smul, inv_smul_smul]
      _ ≤ D + dist (g • f (E x)) (g • k • E (F x)) := dist_triangle _ _ _
      _ = D + dist (f (E x)) (k • E (F x)) := by rw [dist_smul]
  calc ENNReal.ofReal (residualMod G F f E (g • x))
      ≤ ENNReal.ofReal (residualMod G F f E x + D) := ENNReal.ofReal_le_ofReal key
    _ = ENNReal.ofReal (residualMod G F f E x) + ENNReal.ofReal D :=
        ENNReal.ofReal_add (residualMod_nonneg x) dist_nonneg
    _ ≤ _ := by gcongr; exact le_equivDefect hx g

end Residual

section Certificate

variable [PseudoMetricSpace Z] [MulAction G Z] [ContinuousConstSMul G Z] [MulAction G X]
  [TopologicalSpace X] {F : X → X} {f : Z → Z} {E : X → Z}

/-- For equivariant `E` and `F`, if `r̄(x) < τ_f(N)` for every `x ∈ E⁻¹(N)`, then
`F(E⁻¹(G N)) ⊆ int E⁻¹(G N)`. The defect of `f` does not enter, and neither an isometric action
nor `G`-invariance of `N` is needed. -/
theorem image_preimage_saturation_subset_interior (hEc : Continuous E) (hF : IsEquivariant G F)
    (hE : IsEquivariant G E) {N : Set Z}
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G F f E x) < tolerance f N) :
    F '' (E ⁻¹' saturation G N) ⊆ interior (E ⁻¹' saturation G N) := by
  rintro _ ⟨x, hx, rfl⟩
  obtain ⟨g, z, hz, hgz⟩ := mem_saturation.mp hx
  have hy : E (g⁻¹ • x) ∈ N := by rwa [hE, ← hgz, inv_smul_smul]
  obtain ⟨ν, hν, hνN⟩ := exists_lt_of_lt_tolerance (hres _ hy)
  obtain ⟨k, hk⟩ := exists_dist_lt_of_residualMod_lt hν
  have hkF : k • E (F (g⁻¹ • x)) ∈ interior N :=
    hνN (Metric.mem_thickening_iff.mpr ⟨_, mem_image_of_mem f hy, by rwa [dist_comm]⟩)
  have hEF : E (F x) = (g * k⁻¹) • k • E (F (g⁻¹ • x)) := by
    rw [smul_smul, inv_mul_cancel_right, ← hE, ← hF, smul_inv_smul]
  have hmem : E (F x) ∈ interior (saturation G N) := by
    rw [hEF]
    exact (isGInvariant_saturation N).interior _ _ (interior_mono (subset_saturation N) hkF)
  exact preimage_interior_subset_interior_preimage hEc hmem

/-- For equivariant `E` and `F` and a compact `E⁻¹(G N)`: if `r̄(x) < τ_f(N)` for every
`x ∈ E⁻¹(N)`, then `E⁻¹(G N)` is an attracting block for `F`. -/
theorem isAttractingBlock_preimage_saturation_of_isCompact (hEc : Continuous E)
    (hF : IsEquivariant G F) (hE : IsEquivariant G E) {N : Set Z}
    (hK : IsCompact (E ⁻¹' saturation G N))
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G F f E x) < tolerance f N) :
    IsAttractingBlock F (E ⁻¹' saturation G N) :=
  ⟨hK, image_preimage_saturation_subset_interior hEc hF hE hres⟩

/-- The certificate for an approximately equivariant `f`, sharp form: for a finite group,
compact `X`, equivariant `E` and `F` and a closed `N`, if `sup_{x ∈ E⁻¹(N)} r̄(x) < τ_f(N)`, then
`E⁻¹(G N)` is an attracting block for `F`. -/
theorem isAttractingBlock_preimage_saturation [CompactSpace X] [Finite G] (hEc : Continuous E)
    (hF : IsEquivariant G F) (hE : IsEquivariant G E) {N : Set Z} (hN : IsClosed N)
    (hres : ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G F f E x) < tolerance f N) :
    IsAttractingBlock F (E ⁻¹' saturation G N) :=
  isAttractingBlock_preimage_saturation_of_isCompact hEc hF hE
    ((isClosed_saturation hN).preimage hEc).isCompact fun x hx ↦
      (le_iSup₂ (f := fun x (_ : x ∈ E ⁻¹' N) ↦ ENNReal.ofReal (residualMod G F f E x))
        x hx).trans_lt hres

/-- The certificate of the draft lemma, `sup_{x ∈ E⁻¹(N)} r̄(x) + 2 δ_N < τ_f(N)`. It is implied
by the sharp form `isAttractingBlock_preimage_saturation`, whose hypothesis is weaker. -/
theorem isAttractingBlock_preimage_saturation_of_defect [CompactSpace X] [Finite G]
    (hEc : Continuous E) (hF : IsEquivariant G F) (hE : IsEquivariant G E) {N : Set Z}
    (hN : IsClosed N)
    (hres : (⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G F f E x)) + 2 * equivDefect G f N <
      tolerance f N) :
    IsAttractingBlock F (E ⁻¹' saturation G N) :=
  isAttractingBlock_preimage_saturation hEc hF hE hN (le_self_add.trans_lt hres)

end Certificate

end LatentEquivariant
