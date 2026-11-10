import LatentEquivariant.Dynamics.Morse
import LatentEquivariant.Orbit.Quotient

/-!
# The equivariant pullback of attracting blocks

This is Proposition `prop:equivariant_pullback`. A group `G` acts
on the latent space `Z`, `g : Z → Z` is the latent map, `E : X → Z` the encoder, `f : X → X` the
map upstairs. The residual is `r(x) = d(g(E x), E(f x))` and the residual modulo `G` is
`r̄(x) = inf_h d(g(E x), h E(f x))`.

* (i) `image_quotientMap_subset_interior_iff`, `IsAttractingBlock.quotient`,
  `quotientTolerance_eq`: `N/G` is an attracting block for `g/G`, with the same tolerance, the
  tolerance downstairs taken for the quotient distance `d̄([a],[b]) = inf_h d(a, h b)`. The
  attracting-block part needs only an action by homeomorphisms and no metric
  (`isAttractingBlock_quotient_of_isCompact_image` assumes `N/G` compact instead of `N`); the
  tolerance identity needs an isometric action and no compactness.
* (ii) `isAttractingBlock_preimage_of_residualMod`: if `r̄ < τ_g(N)` on `E⁻¹(N)`, then `E⁻¹(N)`
  is an attracting block for `f`. The proof works in `Z` and needs only that `N` is
  `G`-invariant and that `G` acts by homeomorphisms: `g` need not be equivariant (the paper
  assumes it), and neither do `f` and `E`. `isAttractingBlock_preimage_of_residualMod_of_isCompact`
  and `isAttractingBlock_preimage_of_residualMod_of_isClosed` drop the hypothesis that `N` is an
  attracting block for `g`, as in `isAttractingBlock_preimage_of_isClosed`.
* (iii) `residual_smul`, `residualMod_smul`, `isGInvariant_omegaSet`, `iSup_residual_eq`,
  `iSup_residualMod_eq`: for equivariant `f`, `E`, `g` and an isometric action, `r` and `r̄` are
  constant on orbits, their suprema over `E⁻¹(N)` are the suprema over any subset of `E⁻¹(N)`
  that meets every orbit in it, and `ω(E⁻¹(N), f)` is invariant. Only the last needs `G` to act
  on `X` by homeomorphisms.

All statements hold for an arbitrary group. In the paper, compactness of `G` also makes the
orbits closed, so that `d̄` is a metric and `Z/G` is a compact metric space, which its proof of
(ii) uses. With orbits that are not closed `d̄` is only a pseudometric
(`Sharpness.quotientDist_eq_zero`), and (i) is then a statement about it. The proof of (ii) here
does not use `Z/G`. `quotientTolerance` is defined from `d̄` directly; `d̄` is not packaged here
as a pseudometric structure on `Z/G`.
-/

namespace LatentEquivariant

open Set Topology Pointwise MulAction
open scoped ENNReal

variable {G X Z : Type*} [Group G]

section Residual

variable [PseudoMetricSpace Z] [MulAction G Z]

/-- The residual `r(x) = d(g(E x), E(f x))`. -/
def residual (f : X → X) (g : Z → Z) (E : X → Z) (x : X) : ℝ :=
  dist (g (E x)) (E (f x))

/-- The residual modulo `G`, `r̄(x) = inf_h d(g(E x), h E(f x))`. -/
noncomputable def residualMod (G : Type*) [Group G] [MulAction G Z] (f : X → X) (g : Z → Z)
    (E : X → Z) (x : X) : ℝ :=
  ⨅ h : G, dist (g (E x)) (h • E (f x))

variable {f : X → X} {g : Z → Z} {E : X → Z}

theorem bddBelow_range_dist (a b : Z) : BddBelow (range fun h : G ↦ dist a (h • b)) :=
  ⟨0, by rintro _ ⟨h, rfl⟩; exact dist_nonneg⟩

theorem residualMod_le_residual (x : X) : residualMod G f g E x ≤ residual f g E x := by
  have := ciInf_le (bddBelow_range_dist (G := G) (g (E x)) (E (f x))) 1
  simpa [residualMod, residual] using this

theorem exists_dist_lt_of_residualMod_lt {x : X} {ν : ℝ} (h : residualMod G f g E x < ν) :
    ∃ k : G, dist (g (E x)) (k • E (f x)) < ν :=
  exists_lt_of_ciInf_lt h

end Residual

section PullbackUpstairs

variable [PseudoMetricSpace Z] [MulAction G Z] [ContinuousConstSMul G Z] [TopologicalSpace X]
  {f : X → X} {g : Z → Z} {E : X → Z}

/-- Proposition `prop:equivariant_pullback` (ii) with compactness of `E⁻¹(N)` in place of
compactness of `X`: if `N` is `G`-invariant and `r̄(x) < τ_g(N)` for every `x ∈ E⁻¹(N)`, then
`E⁻¹(N)` is an attracting block for `f`. No equivariance of `g`, `f` or `E`, no isometric action,
no compactness of `G` and no attracting-block hypothesis on `N` are needed. -/
theorem isAttractingBlock_preimage_of_residualMod_of_isCompact (hE : Continuous E) {N : Set Z}
    (hK : IsCompact (E ⁻¹' N)) (hNG : IsGInvariant G N)
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) := by
  refine ⟨hK, ?_⟩
  rintro _ ⟨x, hx, rfl⟩
  obtain ⟨ν, hν, hνN⟩ := exists_lt_of_lt_tolerance (hres x hx)
  obtain ⟨k, hk⟩ := exists_dist_lt_of_residualMod_lt hν
  have hkfx : k • E (f x) ∈ interior N := by
    apply hνN
    rw [Metric.mem_thickening_iff]
    exact ⟨g (E x), mem_image_of_mem g hx, by rwa [dist_comm]⟩
  have hfx : E (f x) ∈ interior N := by
    simpa using hNG.interior k⁻¹ _ hkfx
  exact preimage_interior_subset_interior_preimage hE hfx

/-- Proposition `prop:equivariant_pullback` (ii) with `X` compact and `N` only closed and
`G`-invariant. -/
theorem isAttractingBlock_preimage_of_residualMod_of_isClosed [CompactSpace X]
    (hE : Continuous E) {N : Set Z} (hN : IsClosed N) (hNG : IsGInvariant G N)
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) :=
  isAttractingBlock_preimage_of_residualMod_of_isCompact hE (hN.preimage hE).isCompact hNG hres

/-- `r̄` is upper semicontinuous, so under the paper's continuity hypotheses the pointwise bound
gives the bound on the supremum, which is attained. -/
theorem iSup_residualMod_lt_of_forall [CompactSpace X] (hf : Continuous f) (hg : Continuous g)
    (hE : Continuous E) {N : Set Z} (hN : IsClosed N) (hne : (E ⁻¹' N).Nonempty)
    {τ : ℝ≥0∞} (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < τ) :
    ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < τ := by
  have husc : UpperSemicontinuous (residualMod G f g E) := by
    show UpperSemicontinuous fun x ↦ ⨅ h : G, dist (g (E x)) (h • E (f x))
    refine upperSemicontinuous_ciInf
      (fun x ↦ bddBelow_range_dist (G := G) (g (E x)) (E (f x))) fun h ↦ ?_
    exact ((hg.comp hE).dist ((continuous_const_smul h).comp (hE.comp hf))).upperSemicontinuous
  obtain ⟨x₀, hx₀, hmax⟩ :=
    (husc.upperSemicontinuousOn _).exists_isMaxOn hne (hN.preimage hE).isCompact
  exact lt_of_le_of_lt
    (iSup₂_le fun x hx ↦ ENNReal.ofReal_le_ofReal (isMaxOn_iff.mp hmax x hx)) (hres x₀ hx₀)

end PullbackUpstairs

section PullbackPaper

variable [MetricSpace Z] [MulAction G Z] [ContinuousConstSMul G Z] [TopologicalSpace X]
  [CompactSpace X] {f : X → X} {g : Z → Z} {E : X → Z}

/-- Proposition `prop:equivariant_pullback` (ii), pointwise form: if `N` is a `G`-invariant
attracting block for `g` and `r̄(x) < τ_g(N)` for every `x ∈ E⁻¹(N)`, then `E⁻¹(N)` is an
attracting block for `f`. No equivariance of `g`, `f` or `E` is assumed. -/
theorem isAttractingBlock_preimage_of_residualMod_of_forall (hE : Continuous E) {N : Set Z}
    (hN : IsAttractingBlock g N) (hNG : IsGInvariant G N)
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) :=
  isAttractingBlock_preimage_of_residualMod_of_isClosed hE hN.1.isClosed hNG hres

/-- Proposition `prop:equivariant_pullback` (ii): if `N` is a `G`-invariant attracting block for
`g` and `sup_{x ∈ E⁻¹(N)} r̄(x) < τ_g(N)`, then `E⁻¹(N)` is an attracting block for `f`. -/
theorem isAttractingBlock_preimage_of_residualMod (hE : Continuous E) {N : Set Z}
    (hN : IsAttractingBlock g N) (hNG : IsGInvariant G N)
    (hres : ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) :=
  isAttractingBlock_preimage_of_residualMod_of_forall hE hN hNG fun x hx ↦
    lt_of_le_of_lt (le_iSup₂ (f := fun x (_ : x ∈ E ⁻¹' N) ↦
      ENNReal.ofReal (residualMod G f g E x)) x hx) hres

end PullbackPaper

section Quotient

variable [PseudoMetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z]

/-- The orbit distance `inf_h d(a, h b)`. -/
noncomputable def orbitDist (G : Type*) [Group G] [MulAction G Z] (a b : Z) : ℝ :=
  ⨅ h : G, dist a (h • b)

theorem dist_smul_left_eq (k : G) (a b : Z) : dist (k • a) b = dist a (k⁻¹ • b) := by
  rw [← dist_smul k a (k⁻¹ • b), smul_inv_smul]

theorem orbitDist_smul_left (k : G) (a b : Z) : orbitDist G (k • a) b = orbitDist G a b := by
  simp only [orbitDist, dist_smul_left_eq, smul_smul]
  have hsurj : Function.Surjective fun h : G ↦ k⁻¹ * h := fun h ↦ ⟨k * h, by group⟩
  exact hsurj.iInf_comp fun h ↦ dist a (h • b)

omit [IsIsometricSMul G Z] in
theorem orbitDist_smul_right (k : G) (a b : Z) : orbitDist G a (k • b) = orbitDist G a b := by
  simp only [orbitDist, smul_smul]
  have hsurj : Function.Surjective fun h : G ↦ h * k := fun h ↦ ⟨h * k⁻¹, by group⟩
  exact hsurj.iInf_comp fun h ↦ dist a (h • b)

/-- The quotient distance `d̄([a],[b]) = inf_h d(a, h b)` on `Z/G`. -/
noncomputable def quotientDist (G : Type*) [Group G] [MulAction G Z] [IsIsometricSMul G Z] :
    orbitRel.Quotient G Z → orbitRel.Quotient G Z → ℝ :=
  Quotient.lift₂ (orbitDist G) fun a₁ b₁ a₂ b₂ ha hb ↦ by
    obtain ⟨k, rfl⟩ := mem_orbit_iff.mp (orbitRel_apply.mp ha)
    obtain ⟨l, rfl⟩ := mem_orbit_iff.mp (orbitRel_apply.mp hb)
    rw [orbitDist_smul_left, orbitDist_smul_right]

theorem quotientDist_orbitMap (a b : Z) :
    quotientDist G (orbitMap G Z a) (orbitMap G Z b) = orbitDist G a b :=
  rfl

/-- The open `ν`-neighborhood of `A` for the quotient distance. -/
def quotientThickening (G : Type*) [Group G] [MulAction G Z] [IsIsometricSMul G Z] (ν : ℝ)
    (A : Set (orbitRel.Quotient G Z)) : Set (orbitRel.Quotient G Z) :=
  {q | ∃ p ∈ A, quotientDist G q p < ν}

/-- The tolerance of `Ā` for `ḡ`, computed with the quotient distance: the supremum of the
`ν > 0` such that the `ν`-neighborhood of `ḡ(Ā)` for `d̄` lies in `Ā`. -/
noncomputable def quotientTolerance (G : Type*) [Group G] [MulAction G Z] [IsIsometricSMul G Z]
    (gbar : orbitRel.Quotient G Z → orbitRel.Quotient G Z) (Abar : Set (orbitRel.Quotient G Z)) :
    ℝ≥0∞ :=
  ⨆ (ν : ℝ) (_ : 0 < ν ∧ quotientThickening G ν (gbar '' Abar) ⊆ Abar), ENNReal.ofReal ν

/-- For a `G`-invariant set `A`, `π⁻¹(B̄_ν(π(A))) = B_ν(A)`. -/
theorem preimage_quotientThickening {A : Set Z} (hA : IsGInvariant G A) (ν : ℝ) :
    orbitMap G Z ⁻¹' quotientThickening G ν (orbitMap G Z '' A) = Metric.thickening ν A := by
  ext z
  simp only [quotientThickening, mem_preimage, mem_setOf_eq, exists_mem_image,
    quotientDist_orbitMap, orbitDist, Metric.mem_thickening_iff]
  constructor
  · rintro ⟨a, ha, hza⟩
    obtain ⟨k, hk⟩ := exists_lt_of_ciInf_lt hza
    exact ⟨k • a, hA k a ha, hk⟩
  · rintro ⟨a, ha, hza⟩
    refine ⟨a, ha, lt_of_le_of_lt ?_ hza⟩
    simpa using ciInf_le (bddBelow_range_dist (G := G) z a) 1

variable {g : Z → Z}

/-- Proposition `prop:equivariant_pullback` (i), tolerance: `τ_g(N) = τ_ḡ(N/G)`. -/
theorem quotientTolerance_eq (hg : IsEquivariant G g) {N : Set Z} (hN : IsGInvariant G N) :
    quotientTolerance G (quotientMap g hg) (orbitMap G Z '' N) = tolerance g N := by
  have key : ∀ ν : ℝ, quotientThickening G ν (quotientMap g hg '' (orbitMap G Z '' N)) ⊆
      orbitMap G Z '' N ↔ Metric.thickening ν (g '' N) ⊆ N := by
    intro ν
    rw [image_quotientMap_image, ← orbitMap_surjective.preimage_subset_preimage_iff,
      preimage_quotientThickening (hN.image hg), hN.preimage_image_orbitMap]
  simp only [quotientTolerance, tolerance, key]

end Quotient

section QuotientBlock

variable [TopologicalSpace Z] [MulAction G Z] [ContinuousConstSMul G Z] {g : Z → Z}

/-- Proposition `prop:equivariant_pullback` (i), the inclusion: `ḡ(N/G) ⊆ int(N/G)` if and only if
`g(N) ⊆ int N`, for `G`-invariant `N`. Only an action by homeomorphisms is used. -/
theorem image_quotientMap_subset_interior_iff (hg : IsEquivariant G g) {N : Set Z}
    (hN : IsGInvariant G N) :
    quotientMap g hg '' (orbitMap G Z '' N) ⊆ interior (orbitMap G Z '' N) ↔
      g '' N ⊆ interior N := by
  rw [image_quotientMap_image, ← hN.image_interior]
  constructor
  · intro h x hx
    have hx' : x ∈ orbitMap G Z ⁻¹' (orbitMap G Z '' interior N) := h (mem_image_of_mem _ hx)
    rwa [hN.interior.preimage_image_orbitMap] at hx'
  · exact image_mono

/-- Proposition `prop:equivariant_pullback` (i): `N/G` is an attracting block for `ḡ`. -/
theorem IsAttractingBlock.quotient (hg : IsEquivariant G g) {N : Set Z} (hN : IsGInvariant G N)
    (h : IsAttractingBlock g N) : IsAttractingBlock (quotientMap g hg) (orbitMap G Z '' N) :=
  ⟨h.1.image continuous_orbitMap, (image_quotientMap_subset_interior_iff hg hN).mpr h.2⟩

/-- Proposition `prop:equivariant_pullback` (i) with compactness of `N/G` in place of compactness
of `N`, which makes it meaningful for proper actions of noncompact groups, such as deck groups,
where a compact invariant set is empty. -/
theorem isAttractingBlock_quotient_of_isCompact_image (hg : IsEquivariant G g) {N : Set Z}
    (hN : IsGInvariant G N) (hc : IsCompact (orbitMap G Z '' N)) (h : g '' N ⊆ interior N) :
    IsAttractingBlock (quotientMap g hg) (orbitMap G Z '' N) :=
  ⟨hc, (image_quotientMap_subset_interior_iff hg hN).mpr h⟩

end QuotientBlock

section OnOrbits

variable [MulAction G X] {f : X → X} {g : Z → Z} {E : X → Z}

theorem IsEquivariant.iterate {f : X → X} (hf : IsEquivariant G f) (n : ℕ) :
    IsEquivariant G f^[n] := by
  induction n with
  | zero => intro k x; rfl
  | succ n ih =>
    intro k x
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ih, hf]

theorem isGInvariant_iUnion {ι : Sort*} {s : ι → Set X} (hs : ∀ i, IsGInvariant G (s i)) :
    IsGInvariant G (⋃ i, s i) := by
  intro k x hx
  obtain ⟨i, hi⟩ := mem_iUnion.mp hx
  exact mem_iUnion.mpr ⟨i, hs i k x hi⟩

theorem isGInvariant_iInter {ι : Sort*} {s : ι → Set X} (hs : ∀ i, IsGInvariant G (s i)) :
    IsGInvariant G (⋂ i, s i) := by
  intro k x hx
  exact mem_iInter.mpr fun i ↦ hs i k x (mem_iInter.mp hx i)

/-- Proposition `prop:equivariant_pullback` (iii): `ω(A, f)` is `G`-invariant when `A` is and `f`
is equivariant. -/
theorem isGInvariant_omegaSet [TopologicalSpace X] [ContinuousConstSMul G X]
    (hf : IsEquivariant G f) {A : Set X} (hA : IsGInvariant G A) :
    IsGInvariant G (omegaSet f A) :=
  isGInvariant_iInter fun _ ↦ IsGInvariant.closure <| isGInvariant_iUnion fun k ↦
    isGInvariant_iUnion fun _ ↦ hA.image (hf.iterate k)

/-- A function constant on orbits has the same supremum over `A` as over any subset `T ⊆ A` that
meets the orbit of every point of `A`. -/
theorem iSup_eq_of_forall_exists_smul_mem {φ : X → ℝ≥0∞} (hφ : ∀ (k : G) x, φ (k • x) = φ x)
    {A T : Set X} (hTA : T ⊆ A) (hT : ∀ x ∈ A, ∃ k : G, k • x ∈ T) :
    ⨆ x ∈ A, φ x = ⨆ x ∈ T, φ x := by
  refine le_antisymm (iSup₂_le fun x hx ↦ ?_) (biSup_mono hTA)
  obtain ⟨k, hk⟩ := hT x hx
  rw [← hφ k x]
  exact le_iSup₂ (f := fun x (_ : x ∈ T) ↦ φ x) (k • x) hk

variable [PseudoMetricSpace Z] [MulAction G Z] [IsIsometricSMul G Z]

/-- Proposition `prop:equivariant_pullback` (iii): the residual is constant on orbits. -/
theorem residual_smul (hf : IsEquivariant G f) (hg : IsEquivariant G g)
    (hE : IsEquivariant G E) (k : G) (x : X) : residual f g E (k • x) = residual f g E x := by
  simp only [residual]
  rw [hf, hE, hE, hg, dist_smul]

/-- Proposition `prop:equivariant_pullback` (iii): the residual modulo `G` is constant on
orbits. -/
theorem residualMod_smul (hf : IsEquivariant G f) (hg : IsEquivariant G g)
    (hE : IsEquivariant G E) (k : G) (x : X) :
    residualMod G f g E (k • x) = residualMod G f g E x := by
  simp only [residualMod]
  rw [hf, hE, hE, hg]
  have hsurj : Function.Surjective fun h : G ↦ k⁻¹ * h * k := fun h ↦
    ⟨k * h * k⁻¹, by group⟩
  calc ⨅ h : G, dist (k • g (E x)) (h • k • E (f x))
      = ⨅ h : G, dist (g (E x)) ((k⁻¹ * h * k) • E (f x)) := by
        congr 1
        ext h
        rw [← dist_smul k⁻¹, inv_smul_smul, smul_smul, smul_smul]
    _ = ⨅ h : G, dist (g (E x)) (h • E (f x)) :=
        hsurj.iInf_comp fun h ↦ dist (g (E x)) (h • E (f x))

/-- Proposition `prop:equivariant_pullback` (iii), suprema: the supremum of `r` over `E⁻¹(N)` is
its supremum over any subset of `E⁻¹(N)` that meets every orbit in `E⁻¹(N)`. The paper says "any
set that meets every orbit in `E⁻¹(N)`"; the set must lie in `E⁻¹(N)`, since a larger set can
have a larger supremum. -/
theorem iSup_residual_eq (hf : IsEquivariant G f) (hg : IsEquivariant G g)
    (hE : IsEquivariant G E) {N : Set Z} {T : Set X} (hTN : T ⊆ E ⁻¹' N)
    (hT : ∀ x ∈ E ⁻¹' N, ∃ k : G, k • x ∈ T) :
    ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (residual f g E x) =
      ⨆ x ∈ T, ENNReal.ofReal (residual f g E x) :=
  iSup_eq_of_forall_exists_smul_mem (fun k x ↦ by rw [residual_smul hf hg hE]) hTN hT

/-- Proposition `prop:equivariant_pullback` (iii), suprema, for the residual modulo `G`. -/
theorem iSup_residualMod_eq (hf : IsEquivariant G f) (hg : IsEquivariant G g)
    (hE : IsEquivariant G E) {N : Set Z} {T : Set X} (hTN : T ⊆ E ⁻¹' N)
    (hT : ∀ x ∈ E ⁻¹' N, ∃ k : G, k • x ∈ T) :
    ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) =
      ⨆ x ∈ T, ENNReal.ofReal (residualMod G f g E x) :=
  iSup_eq_of_forall_exists_smul_mem (fun k x ↦ by rw [residualMod_smul hf hg hE]) hTN hT

end OnOrbits

end LatentEquivariant

namespace LatentEquivariant

open Set Topology Pointwise MulAction

section PullbackGenerators

variable {G X Z : Type*} [Group G] [TopologicalSpace X] [T2Space X] [CompactSpace X]
  [PseudoMetricSpace Z] [MulAction G Z] [ContinuousConstSMul G Z]
  {F : X → X} {f : Z → Z} {E : X → Z}

/-- Uniform residual bounds on generators transfer their entire generated block sublattice. -/
theorem preimage_generatedSublattice_blocks_of_residualMod (hE : Continuous E)
    {S : Set (Set Z)} (hclosed : ∀ N ∈ S, IsClosed N)
    (hinv : ∀ N ∈ S, IsGInvariant G N)
    (hres : ∀ N ∈ S, ∀ x ∈ E ⁻¹' N,
      ENNReal.ofReal (residualMod G F f E x) < tolerance f N) :
    preimageSublattice E (generatedSublattice (insert ∅ S)) ≤ attractingBlockSublattice F := by
  apply (Sublattice.map_le_iff_le_comap).mpr
  apply generatedSublattice_le
  intro N hN
  rcases hN with rfl | hN
  · change IsAttractingBlock F (E ⁻¹' ∅)
    simpa using (isAttractingBlock_empty (f := F))
  · exact isAttractingBlock_preimage_of_residualMod_of_isClosed hE (hclosed N hN)
      (hinv N hN) (hres N hN)

variable [MulAction G X]

omit [TopologicalSpace X] [T2Space X] [CompactSpace X] [PseudoMetricSpace Z]
  [ContinuousConstSMul G Z] in
theorem preimage_generatedSublattice_invariant (hE : IsEquivariant G E)
    {S : Set (Set Z)} (hinv : ∀ N ∈ S, IsGInvariant G N) :
    preimageSublattice E (generatedSublattice (insert ∅ S)) ≤ invariantSetSublattice G X := by
  apply (Sublattice.map_le_iff_le_comap).mpr
  apply generatedSublattice_le
  intro N hN
  rcases hN with rfl | hN
  · change IsGInvariant G (E ⁻¹' ∅)
    simpa using (isGInvariant_empty (G := G) (X := X))
  · exact (hinv N hN).preimage hE

end PullbackGenerators

end LatentEquivariant

namespace LatentEquivariant

open Set Topology

section InvariantMorseBlocks

variable {G X : Type*} [Group G] [TopologicalSpace X] [MulAction G X]
  [ContinuousConstSMul G X] {F : X → X} (L : Sublattice (Set X))
  [Fintype L] [BoundedOrder L]

/-- Equivariant dynamics and invariant lattice blocks give invariant Morse pieces. -/
theorem isGInvariant_morseBlock (hF : IsEquivariant G F)
    (hL : ∀ N ∈ L, IsGInvariant G N) (p : L) : IsGInvariant G (morseBlock L F p) :=
  ((hL p p.property).diff
    (hL ((lowerJoin p : L) : Set X) (lowerJoin p).property)).closure.Inv hF

end InvariantMorseBlocks

end LatentEquivariant
