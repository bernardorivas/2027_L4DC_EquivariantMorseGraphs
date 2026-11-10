import LatentEquivariant.Latent.Pullback

/-!
# The action by homeomorphisms in `prop:equivariant_pullback`

`Latent/Pullback.lean` proves the attracting-block part of (i) and part (ii) of Proposition
`prop:equivariant_pullback` for an action by homeomorphisms (`ContinuousConstSMul`) of any group,
with no metric condition on the action. This file shows where that hypothesis enters, and that it
cannot be dropped.

* `isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior` and
  `IsAttractingBlock.quotient_of_isGInvariant_interior`: both proofs use the action only through
  the `G`-invariance of `int N`, which follows from the invariance of `N` for an action by
  homeomorphisms (`IsGInvariant.interior`). With that hypothesis instead, (ii) and the
  attracting-block part of (i) hold for any action of any group, and `N` itself need not be
  invariant. The hypothesis also holds for some discontinuous actions
  (`isGInvariant_interior_Icc`).
* `equivariant_pullback_ii_needs_continuous_action`: (ii) fails for an action that is not by
  homeomorphisms. The group of order two `Flip` acts on `ℝ` by the involution `σ` that swaps `1/2`
  and `2` and fixes all other points, which is discontinuous (`not_continuous_swapHalfTwo`). Take
  `g = 0`, `N = [-1, 1] ∪ {2}`, `X = [-3, 3]`, `E` the inclusion and `f` the constant map `2`.
  Then `g` is equivariant, `N` is a `σ`-invariant attracting block of `g` with `τ_g(N) = 1`, and
  `r̄ = 1/2` everywhere. All hypotheses of `isAttractingBlock_preimage_of_residualMod_of_forall`
  hold except `ContinuousConstSMul`, and also the bound on the supremum of `r̄` in the paper, but
  `f(E⁻¹(N)) = {2}` does not lie in the interior of `E⁻¹(N) = [-1, 1] ∪ {2}`. Here
  `int N = (-1, 1)` is not invariant (`not_isGInvariant_interior_blockN`), and the residual
  `r = 2` exceeds the tolerance (`residual_eq`), so `thm:main_simple` does not apply.
  `not_forall_isAttractingBlock_preimage_of_residualMod` states that the theorem without
  `ContinuousConstSMul` is false.
* `equivariant_pullback_i_needs_continuous_action`: (i) fails for an action that is not by
  homeomorphisms. `SeqFlip` acts on `ℝ` by the involution that swaps `1/n` and `2 + 1/n` for
  `n ≥ 2` and fixes all other points. For `g = 0`, the set
  `N = [-1/2, 1/2] ∪ {2} ∪ {2 + 1/n | n ≥ 2}` is an invariant attracting block of `g`, but the
  orbit `[0]` is not an interior point of `N/G` in the quotient topology
  (`orbitMap_zero_notMem_interior`). So `ḡ(N/G) ⊄ int(N/G)`, and `N/G` is not an attracting block
  of `ḡ`. `not_forall_isAttractingBlock_quotient` states that `IsAttractingBlock.quotient` without
  `ContinuousConstSMul` is false.
* `equivariant_pullback_ii_needs_invariant_interior`: without the equivariance of `g`, which
  `Latent/Pullback.lean` does not assume, (ii) needs an invariance hypothesis on `N` (of `N`, or
  of `int N`), even for an isometric action. The group of order two `NegFlip` acts on `ℝ` by
  `x ↦ -x`. Take `X = ℝ`, `E = id`, `f` the constant map `-2`, `g` the constant map `2` and
  `N = [1, 3]`. Then `N` is an attracting block of `g`, `E⁻¹(N)` is compact, and
  `r̄ = 0 < τ_g(N)` everywhere, since the generator maps `-2` to `2`. But `f(1) = -2` does not
  lie in `int E⁻¹(N) = (1, 3)`. Here `g` is not equivariant, `N` is not invariant, and
  `int N = (1, 3)` is not invariant (`not_isGInvariant_interior_Icc_negFlip`). So the invariance
  of `int N` cannot be dropped from
  `isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior`, and a continuous
  isometric action does not replace it.
  `not_forall_isAttractingBlock_preimage_of_residualMod_of_isCompact` states that
  `isAttractingBlock_preimage_of_residualMod_of_isCompact` without the invariance of `N` is false,
  even for an isometric action. Since `X = ℝ` is not compact, this does not cover
  `isAttractingBlock_preimage_of_residualMod_of_forall`; restricting `f` and `E` to `X = [-3, 3]`
  would, which is not formalized.

So the weakenings of `Latent/Pullback.lean` to an action by homeomorphisms are sharp in the sense
that the invariance of `N` alone does not suffice; the invariance of `int N` does.

The counterexamples for `Flip` and `SeqFlip` take `Z = ℝ`, which is not compact as in the paper;
the Lean theorems they bound assume no compactness of `Z`. The involutions map `[-3, 3]` to
itself and the sets lie in `(-3, 3)`, so the same examples work on `Z = [-3, 3]`; this is not
formalized.

Not formalized: the tolerance identity `τ_g(N) = τ_ḡ(N/G)` of (i) for actions that are not
isometric (the quotient distance of `Latent/Pullback.lean` is defined only for isometric
actions), and the sharpness of the action hypothesis in part (iii), for instance of
`ContinuousConstSMul G X` in `isGInvariant_omegaSet`; part (iii) itself is in
`Latent/Pullback.lean`.
-/

namespace LatentEquivariant

open Set Topology MulAction Filter
open scoped ENNReal

section Interior

variable {G X Z : Type*} [Group G]

/-- Proposition `prop:equivariant_pullback` (ii) with the `G`-invariance of `int N` in place of
an action by homeomorphisms and the invariance of `N`: if `E⁻¹(N)` is compact and
`r̄(x) < τ_g(N)` for every `x ∈ E⁻¹(N)`, then `E⁻¹(N)` is an attracting block for `f`. No
topological or metric condition on the action is needed. For an action by homeomorphisms and
invariant `N` the hypothesis holds by `IsGInvariant.interior`, which gives
`isAttractingBlock_preimage_of_residualMod_of_isCompact`. -/
theorem isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior [PseudoMetricSpace Z]
    [MulAction G Z] [TopologicalSpace X] {f : X → X} {g : Z → Z} {E : X → Z} (hE : Continuous E)
    {N : Set Z} (hK : IsCompact (E ⁻¹' N)) (hint : IsGInvariant G (interior N))
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
    simpa using hint k⁻¹ _ hkfx
  exact preimage_interior_subset_interior_preimage hE hfx

/-- Proposition `prop:equivariant_pullback` (i), the attracting-block part, with the
`G`-invariance of `int N` in place of an action by homeomorphisms and the invariance of `N`: then
`π(int N)` is open in the quotient topology, so `N/G` is an attracting block for `ḡ`. For an
action by homeomorphisms and invariant `N` the hypothesis holds by `IsGInvariant.interior`, which
gives `IsAttractingBlock.quotient`. -/
theorem IsAttractingBlock.quotient_of_isGInvariant_interior [TopologicalSpace Z] [MulAction G Z]
    {g : Z → Z} (hg : IsEquivariant G g) {N : Set Z} (hint : IsGInvariant G (interior N))
    (h : IsAttractingBlock g N) : IsAttractingBlock (quotientMap g hg) (orbitMap G Z '' N) := by
  refine ⟨h.1.image continuous_orbitMap, ?_⟩
  rw [image_quotientMap_image]
  have hpre : IsOpen (orbitMap G Z ⁻¹' (orbitMap G Z '' interior N)) := by
    rw [hint.preimage_image_orbitMap]
    exact isOpen_interior
  have hopen : IsOpen (orbitMap G Z '' interior N) :=
    (isQuotientMap_quotient_mk' (s := orbitRel G Z)).isCoinducing.isOpen_preimage.mp hpre
  exact (image_mono h.2).trans (interior_maximal (image_mono interior_subset) hopen)

end Interior

namespace ActionSharpness

/-! ### Groups of order two acting by involutions -/

/-- The two elements of `ZMod 2`. -/
theorem zmod_two_cases (a : ZMod 2) : a = 0 ∨ a = 1 := by
  revert a
  decide

/-- The action of the group of order two `Multiplicative (ZMod 2)` on `α` through an involution
`s`: the generator `ofAdd 1` acts by `s`. No continuity is assumed. -/
@[reducible] def involutionAction {α : Type*} (s : α → α) (hs : Function.Involutive s) :
    MulAction (Multiplicative (ZMod 2)) α where
  smul a x := if Multiplicative.toAdd a = 0 then x else s x
  one_smul _ := if_pos rfl
  mul_smul a b x := by
    change (if Multiplicative.toAdd a + Multiplicative.toAdd b = 0 then x else s x) =
      if Multiplicative.toAdd a = 0 then (if Multiplicative.toAdd b = 0 then x else s x)
        else s (if Multiplicative.toAdd b = 0 then x else s x)
    rcases zmod_two_cases (Multiplicative.toAdd a) with ha | ha <;>
      rcases zmod_two_cases (Multiplicative.toAdd b) with hb | hb <;>
      simp only [ha, hb, add_zero, zero_add, if_true, show (1 : ZMod 2) + 1 = 0 by decide,
        show (1 : ZMod 2) ≠ 0 by decide, if_false, hs x]

/-- A group of order two, `Multiplicative (ZMod 2)`, as a new type so that its action on `ℝ` by
`swapHalfTwo` does not clash with other actions. -/
def Flip : Type := Multiplicative (ZMod 2)

/-- The group structure of `Flip`. -/
instance : Group Flip := inferInstanceAs (Group (Multiplicative (ZMod 2)))

/-- The generator of `Flip`. -/
def Flip.gen : Flip := Multiplicative.ofAdd (1 : ZMod 2)

/-- `Flip = {1, gen}`. -/
theorem Flip.eq_one_or (h : Flip) : h = 1 ∨ h = Flip.gen := by
  rcases zmod_two_cases (Multiplicative.toAdd (h : Multiplicative (ZMod 2))) with h0 | h1
  · exact Or.inl h0
  · exact Or.inr h1

/-- The involution `σ` of `ℝ` that swaps `1/2` and `2` and fixes all other points. -/
noncomputable def swapHalfTwo : ℝ → ℝ := Equiv.swap (1 / 2 : ℝ) 2

/-- `σ` is an involution. -/
theorem swapHalfTwo_involutive : Function.Involutive swapHalfTwo :=
  fun x ↦ Equiv.swap_apply_self _ _ x

/-- `σ` is not continuous: it is the identity on a punctured neighborhood of `1/2`, but
`σ(1/2) = 2`. -/
theorem not_continuous_swapHalfTwo : ¬ Continuous swapHalfTwo := by
  intro h
  have hU : swapHalfTwo ⁻¹' Ioo 1 3 ∈ 𝓝 (1 / 2 : ℝ) := by
    apply h.continuousAt.preimage_mem_nhds
    rw [swapHalfTwo, Equiv.swap_apply_left]
    exact Ioo_mem_nhds (by norm_num) (by norm_num)
  obtain ⟨δ, hδ, hball⟩ := Metric.mem_nhds_iff.mp hU
  have hm0 : 0 < min (δ / 2) (1 / 4) := lt_min (by linarith) (by norm_num)
  have hmδ : min (δ / 2) (1 / 4) < δ := lt_of_le_of_lt (min_le_left _ _) (by linarith)
  have hm1 : min (δ / 2) (1 / 4) ≤ 1 / 4 := min_le_right _ _
  have hmem : 1 / 2 + min (δ / 2) (1 / 4) ∈ Metric.ball (1 / 2 : ℝ) δ := by
    rw [Metric.mem_ball, Real.dist_eq, add_sub_cancel_left, abs_of_pos hm0]
    exact hmδ
  have h1 := hball hmem
  rw [mem_preimage, swapHalfTwo, Equiv.swap_apply_of_ne_of_ne (fun h ↦ by linarith)
    (fun h ↦ by linarith)] at h1
  linarith [h1.1]

/-- `Flip` acts on `ℝ` by `σ`. The action is not by homeomorphisms
(`not_continuousConstSMul_flip`). -/
noncomputable instance : MulAction Flip ℝ := involutionAction swapHalfTwo swapHalfTwo_involutive

/-- The generator acts by `σ`. -/
theorem Flip.gen_smul (x : ℝ) : Flip.gen • x = swapHalfTwo x :=
  if_neg (by decide)

/-- `0` is fixed. -/
theorem Flip.smul_zero (h : Flip) : h • (0 : ℝ) = 0 := by
  rcases h.eq_one_or with rfl | rfl
  · exact one_smul _ _
  · rw [Flip.gen_smul, swapHalfTwo, Equiv.swap_apply_of_ne_of_ne (by norm_num) (by norm_num)]

/-- The action of `Flip` is not by homeomorphisms, since `σ` is not continuous. -/
theorem not_continuousConstSMul_flip : ¬ ContinuousConstSMul Flip ℝ := fun h ↦
  not_continuous_swapHalfTwo (by
    have := h.continuous_const_smul Flip.gen
    simpa only [Flip.gen_smul] using this)

/-! ### Part (ii) -/

/-- The block `N = [-1, 1] ∪ {2}`. -/
def blockN : Set ℝ := Icc (-1) 1 ∪ {2}

/-- The latent map `g = 0`. -/
def zeroMap : ℝ → ℝ := fun _ ↦ 0

/-- The compact interval `[-3, 3]`, the space `X` upstairs. -/
abbrev Box : Type := Icc (-3 : ℝ) 3

/-- The encoder `E`: the inclusion `[-3, 3] → ℝ`. -/
def incl : Box → ℝ := Subtype.val

/-- The map `f` upstairs: the constant map `2` on `[-3, 3]`. -/
def constTwo : Box → Box := fun _ ↦ ⟨2, by norm_num, by norm_num⟩

/-- `0 ∈ N`. -/
theorem zero_mem_blockN : (0 : ℝ) ∈ blockN := Or.inl ⟨by norm_num, by norm_num⟩

/-- The image of a nonempty set under `g = 0` is `{0}`. -/
theorem image_zeroMap {A : Set ℝ} (hA : A.Nonempty) : zeroMap '' A = {0} :=
  hA.image_const 0

/-- `N` is an attracting block of `g = 0`: `g(N) = {0} ⊆ (-1, 1) ⊆ int N`. -/
theorem isAttractingBlock_blockN : IsAttractingBlock zeroMap blockN := by
  refine ⟨isCompact_Icc.union isCompact_singleton, ?_⟩
  rw [image_zeroMap ⟨0, zero_mem_blockN⟩, singleton_subset_iff]
  refine interior_mono subset_union_left ?_
  rw [interior_Icc]
  constructor <;> norm_num

/-- `N` is `σ`-invariant: `σ` swaps `1/2 ∈ [-1, 1]` and `2`. -/
theorem isGInvariant_blockN : IsGInvariant Flip blockN := by
  intro h x hx
  rcases h.eq_one_or with rfl | rfl
  · rwa [one_smul]
  · rw [Flip.gen_smul, swapHalfTwo, Equiv.swap_apply_def]
    split_ifs with h1 h2
    · exact Or.inr rfl
    · exact Or.inl ⟨by norm_num, by norm_num⟩
    · exact hx

/-- `g = 0` is equivariant, since `σ(0) = 0`. -/
theorem isEquivariant_zeroMap : IsEquivariant Flip zeroMap := fun h _ ↦ (h.smul_zero).symm

/-- The tolerance of `N` for `g = 0` is `τ_g(N) = 1`: the `ν`-neighborhood `(-ν, ν)` of
`g(N) = {0}` lies in `N` if and only if `ν ≤ 1`. -/
theorem tolerance_blockN : tolerance zeroMap blockN = 1 := by
  rw [tolerance, image_zeroMap ⟨0, zero_mem_blockN⟩]
  apply le_antisymm
  · refine iSup₂_le fun ν hν ↦ ?_
    obtain ⟨hν0, hsub⟩ := hν
    rw [ENNReal.ofReal_le_one]
    by_contra hν1
    rw [not_le] at hν1
    have ht1 : 1 < min ((1 + ν) / 2) (3 / 2) := lt_min (by linarith) (by norm_num)
    have ht2 : min ((1 + ν) / 2) (3 / 2) ≤ 3 / 2 := min_le_right _ _
    have htν : min ((1 + ν) / 2) (3 / 2) < ν := lt_of_le_of_lt (min_le_left _ _) (by linarith)
    have hmem : min ((1 + ν) / 2) (3 / 2) ∈ Metric.thickening ν ({0} : Set ℝ) := by
      rw [Metric.thickening_singleton, Metric.mem_ball, Real.dist_eq, sub_zero,
        abs_of_pos (by linarith)]
      exact htν
    rcases hsub hmem with h | h
    · linarith [h.2]
    · have : min ((1 + ν) / 2) (3 / 2) = 2 := h
      linarith
  · refine le_iSup₂_of_le (1 : ℝ) ⟨one_pos, ?_⟩ (by simp)
    intro y hy
    rw [Metric.thickening_singleton, Metric.mem_ball, Real.dist_eq, sub_zero, abs_lt] at hy
    exact Or.inl ⟨hy.1.le, hy.2.le⟩

/-- The residual modulo `G` is `r̄(x) = min(d(0, 2), d(0, σ 2)) = 1/2` for every `x`. -/
theorem residualMod_eq (x : Box) : residualMod Flip constTwo zeroMap incl x = 1 / 2 := by
  have hgen : dist (zeroMap (incl x)) (Flip.gen • incl (constTwo x)) = 1 / 2 := by
    rw [Flip.gen_smul]
    simp only [zeroMap, incl, constTwo, swapHalfTwo, Equiv.swap_apply_right, Real.dist_eq]
    norm_num
  apply le_antisymm
  · rw [← hgen]
    exact ciInf_le (bddBelow_range_dist (G := Flip) _ _) Flip.gen
  · refine le_ciInf fun h ↦ ?_
    rcases h.eq_one_or with rfl | rfl
    · rw [one_smul]
      simp only [zeroMap, incl, constTwo, Real.dist_eq]
      norm_num
    · rw [hgen]

/-- The residual itself is `r(x) = d(0, 2) = 2 > τ_g(N)`, so Theorem `thm:main_simple` does not
apply, only the residual modulo `G` is small. -/
theorem residual_eq (x : Box) : residual constTwo zeroMap incl x = 2 := by
  simp only [residual, zeroMap, incl, constTwo, Real.dist_eq]
  norm_num

/-- The pointwise hypothesis of (ii): `r̄(x) = 1/2 < 1 = τ_g(N)`. -/
theorem residualMod_lt_tolerance (x : Box) :
    ENNReal.ofReal (residualMod Flip constTwo zeroMap incl x) < tolerance zeroMap blockN := by
  rw [residualMod_eq, tolerance_blockN, ENNReal.ofReal_lt_one]
  norm_num

/-- The hypothesis of (ii) as stated in the paper: `sup_{x ∈ E⁻¹(N)} r̄(x) < τ_g(N)`. -/
theorem iSup_residualMod_lt_tolerance :
    ⨆ x ∈ incl ⁻¹' blockN, ENNReal.ofReal (residualMod Flip constTwo zeroMap incl x) <
      tolerance zeroMap blockN := by
  refine lt_of_le_of_lt (iSup₂_le fun x _ ↦ le_of_eq (congrArg ENNReal.ofReal (residualMod_eq x)))
    ?_
  rw [tolerance_blockN, ENNReal.ofReal_lt_one]
  norm_num

/-- The conclusion of (ii) fails: `f(E⁻¹(N)) = {2}`, and `2` is not an interior point of
`E⁻¹(N) = [-1, 1] ∪ {2}` in `[-3, 3]`. -/
theorem not_isAttractingBlock_preimage : ¬ IsAttractingBlock constTwo (incl ⁻¹' blockN) := by
  rintro ⟨-, hsub⟩
  have h0 : (⟨0, by norm_num, by norm_num⟩ : Box) ∈ incl ⁻¹' blockN := zero_mem_blockN
  have h2 := hsub ⟨_, h0, rfl⟩
  rw [mem_interior_iff_mem_nhds, Metric.mem_nhds_iff] at h2
  obtain ⟨δ, hδ, hball⟩ := h2
  have hm0 : 0 < min (δ / 2) (1 / 2) := lt_min (by linarith) (by norm_num)
  have hmδ : min (δ / 2) (1 / 2) < δ := lt_of_le_of_lt (min_le_left _ _) (by linarith)
  have hm1 : min (δ / 2) (1 / 2) ≤ 1 / 2 := min_le_right _ _
  have ht : 2 + min (δ / 2) (1 / 2) ∈ Icc (-3 : ℝ) 3 := ⟨by linarith, by linarith⟩
  have hmem : (⟨2 + min (δ / 2) (1 / 2), ht⟩ : Box) ∈
      Metric.ball (constTwo ⟨0, by norm_num, by norm_num⟩) δ := by
    rw [Metric.mem_ball, Subtype.dist_eq]
    simp only [constTwo, Real.dist_eq, add_sub_cancel_left, abs_of_pos hm0]
    exact hmδ
  rcases hball hmem with h | h
  · have : 2 + min (δ / 2) (1 / 2) ≤ 1 := h.2
    linarith
  · have : 2 + min (δ / 2) (1 / 2) = 2 := h
    linarith

/-- The interior `int N = (-1, 1)` is not invariant, since `σ(1/2) = 2`. By
`isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior` this is the hypothesis that
fails. -/
theorem not_isGInvariant_interior_blockN : ¬ IsGInvariant Flip (interior blockN) := fun h ↦
  not_isAttractingBlock_preimage
    (isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior (G := Flip)
      continuous_subtype_val
      ((isClosed_Icc.union isClosed_singleton).preimage continuous_subtype_val).isCompact h
      fun x _ ↦ residualMod_lt_tolerance x)

/-- **Proposition `prop:equivariant_pullback` (ii) needs an action by homeomorphisms.** For the
discontinuous action of `Flip` on `ℝ` by `σ`, the maps `f`, `g`, `E` are continuous, `g` is
equivariant, `N` is a `G`-invariant attracting block of `g`, and `r̄ < τ_g(N)` on `E⁻¹(N)`,
pointwise and for the supremum, on the compact space `X = [-3, 3]`: every hypothesis of
`isAttractingBlock_preimage_of_residualMod_of_forall` except `ContinuousConstSMul`, which fails.
Yet `E⁻¹(N)` is not an attracting block for `f`. -/
theorem equivariant_pullback_ii_needs_continuous_action :
    CompactSpace Box ∧ Continuous constTwo ∧ Continuous zeroMap ∧ Continuous incl ∧
      IsEquivariant Flip zeroMap ∧ IsAttractingBlock zeroMap blockN ∧
      IsGInvariant Flip blockN ∧
      (∀ x ∈ incl ⁻¹' blockN,
        ENNReal.ofReal (residualMod Flip constTwo zeroMap incl x) < tolerance zeroMap blockN) ∧
      ⨆ x ∈ incl ⁻¹' blockN, ENNReal.ofReal (residualMod Flip constTwo zeroMap incl x) <
        tolerance zeroMap blockN ∧
      ¬ IsAttractingBlock constTwo (incl ⁻¹' blockN) ∧ ¬ ContinuousConstSMul Flip ℝ :=
  ⟨inferInstance, continuous_const, continuous_const, continuous_subtype_val,
    isEquivariant_zeroMap, isAttractingBlock_blockN, isGInvariant_blockN,
    fun x _ ↦ residualMod_lt_tolerance x, iSup_residualMod_lt_tolerance,
    not_isAttractingBlock_preimage, not_continuousConstSMul_flip⟩

/-- The hypothesis `ContinuousConstSMul` cannot be dropped from
`isAttractingBlock_preimage_of_residualMod_of_forall`, Proposition `prop:equivariant_pullback`
(ii): the statement without it is false, by `equivariant_pullback_ii_needs_continuous_action`. -/
theorem not_forall_isAttractingBlock_preimage_of_residualMod :
    ¬ ∀ (G X Z : Type) [Group G] [MetricSpace Z] [MulAction G Z] [TopologicalSpace X]
      [CompactSpace X] (f : X → X) (g : Z → Z) (E : X → Z), Continuous E → ∀ N : Set Z,
      IsAttractingBlock g N → IsGInvariant G N →
      (∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) →
      IsAttractingBlock f (E ⁻¹' N) := fun h ↦
  not_isAttractingBlock_preimage (h Flip Box ℝ constTwo zeroMap incl continuous_subtype_val
    blockN isAttractingBlock_blockN isGInvariant_blockN fun x _ ↦ residualMod_lt_tolerance x)

/-- The invariance of `int N`, the hypothesis of
`isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior` and
`IsAttractingBlock.quotient_of_isGInvariant_interior`, does not need an action by homeomorphisms:
for the discontinuous action of `Flip`, the interior `(-1/4, 1/4)` of `[-1/4, 1/4]` is
invariant, since `σ` fixes it pointwise. -/
theorem isGInvariant_interior_Icc : IsGInvariant Flip (interior (Icc (-1 / 4 : ℝ) (1 / 4))) := by
  rw [interior_Icc]
  intro h x hx
  rcases h.eq_one_or with rfl | rfl
  · rwa [one_smul]
  · rw [Flip.gen_smul, swapHalfTwo, Equiv.swap_apply_of_ne_of_ne
      (fun h ↦ by linarith [hx.2]) (fun h ↦ by linarith [hx.2])]
    exact hx

/-! ### Part (i) -/

/-- A second group of order two, acting on `ℝ` by `seqSwap`. -/
def SeqFlip : Type := Multiplicative (ZMod 2)

/-- The group structure of `SeqFlip`. -/
instance : Group SeqFlip := inferInstanceAs (Group (Multiplicative (ZMod 2)))

/-- The generator of `SeqFlip`. -/
def SeqFlip.gen : SeqFlip := Multiplicative.ofAdd (1 : ZMod 2)

/-- `SeqFlip = {1, gen}`. -/
theorem SeqFlip.eq_one_or (h : SeqFlip) : h = 1 ∨ h = SeqFlip.gen := by
  rcases zmod_two_cases (Multiplicative.toAdd (h : Multiplicative (ZMod 2))) with h0 | h1
  · exact Or.inl h0
  · exact Or.inr h1

/-- The points `1/n` for `n ≥ 2`, written `1/(n + 2)` for `n : ℕ`. -/
def seqPts : Set ℝ := range fun n : ℕ ↦ 1 / ((n : ℝ) + 2)

/-- The points `1/n` are positive. -/
theorem pos_of_mem_seqPts {x : ℝ} (hx : x ∈ seqPts) : 0 < x := by
  obtain ⟨n, rfl⟩ := hx
  positivity

/-- The points `1/n`, `n ≥ 2`, are at most `1/2`. -/
theorem le_half_of_mem_seqPts {x : ℝ} (hx : x ∈ seqPts) : x ≤ 1 / 2 := by
  obtain ⟨n, rfl⟩ := hx
  exact one_div_le_one_div_of_le (by norm_num) (by linarith [n.cast_nonneg (α := ℝ)])

open scoped Classical in
/-- The involution of `ℝ` that swaps `1/n` and `2 + 1/n` for every `n ≥ 2` and fixes all other
points. -/
noncomputable def seqSwap (x : ℝ) : ℝ :=
  if x ∈ seqPts then x + 2 else if x - 2 ∈ seqPts then x - 2 else x

/-- `seqSwap (1/n) = 1/n + 2`. -/
theorem seqSwap_of_mem {x : ℝ} (hx : x ∈ seqPts) : seqSwap x = x + 2 := by
  rw [seqSwap, if_pos hx]

/-- `seqSwap (1/n + 2) = 1/n`. -/
theorem seqSwap_of_sub_mem {x : ℝ} (hx : x - 2 ∈ seqPts) : seqSwap x = x - 2 := by
  have hx' : x ∉ seqPts := fun h ↦ by
    have := le_half_of_mem_seqPts h
    have := pos_of_mem_seqPts hx
    linarith
  rw [seqSwap, if_neg hx', if_pos hx]

/-- `seqSwap` fixes all other points. -/
theorem seqSwap_of_notMem {x : ℝ} (h1 : x ∉ seqPts) (h2 : x - 2 ∉ seqPts) : seqSwap x = x := by
  rw [seqSwap, if_neg h1, if_neg h2]

/-- `seqSwap` is an involution. -/
theorem seqSwap_involutive : Function.Involutive seqSwap := by
  intro x
  by_cases h1 : x ∈ seqPts
  · rw [seqSwap_of_mem h1, seqSwap_of_sub_mem (by simpa using h1)]
    ring
  · by_cases h2 : x - 2 ∈ seqPts
    · rw [seqSwap_of_sub_mem h2, seqSwap_of_mem h2]
      ring
    · rw [seqSwap_of_notMem h1 h2, seqSwap_of_notMem h1 h2]

/-- `SeqFlip` acts on `ℝ` by `seqSwap`. The action is not by homeomorphisms
(`not_continuousConstSMul_seqFlip`). -/
noncomputable instance : MulAction SeqFlip ℝ := involutionAction seqSwap seqSwap_involutive

/-- The generator acts by `seqSwap`. -/
theorem SeqFlip.gen_smul (x : ℝ) : SeqFlip.gen • x = seqSwap x :=
  if_neg (by decide)

/-- The block `N = [-1/2, 1/2] ∪ {2} ∪ {2 + 1/n | n ≥ 2}`. -/
def seqBlock : Set ℝ :=
  Icc (-1 / 2) (1 / 2) ∪ insert 2 (range fun n : ℕ ↦ 1 / ((n : ℝ) + 2) + 2)

/-- `0 ∈ N`. -/
theorem zero_mem_seqBlock : (0 : ℝ) ∈ seqBlock := Or.inl ⟨by norm_num, by norm_num⟩

/-- `N` is compact: `{2} ∪ {2 + 1/n | n ≥ 2}` is a convergent sequence with its limit. -/
theorem isCompact_seqBlock : IsCompact seqBlock := by
  refine isCompact_Icc.union (Tendsto.isCompact_insert_range ?_)
  have h1 : Tendsto (fun n : ℕ ↦ 1 / ((n : ℝ) + 2)) atTop (𝓝 0) := by
    have := (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).comp (tendsto_add_atTop_nat 1)
    refine this.congr fun n ↦ ?_
    simp only [Function.comp_apply, Nat.cast_add, Nat.cast_one]
    ring_nf
  simpa using h1.add_const 2

/-- `N` is invariant: `seqSwap` swaps `1/n ∈ [-1/2, 1/2]` and `2 + 1/n`. -/
theorem isGInvariant_seqBlock : IsGInvariant SeqFlip seqBlock := by
  intro h x hx
  rcases h.eq_one_or with rfl | rfl
  · rwa [one_smul]
  · rw [SeqFlip.gen_smul]
    by_cases h1 : x ∈ seqPts
    · rw [seqSwap_of_mem h1]
      obtain ⟨n, rfl⟩ := h1
      exact Or.inr (Or.inr ⟨n, rfl⟩)
    · by_cases h2 : x - 2 ∈ seqPts
      · rw [seqSwap_of_sub_mem h2]
        exact Or.inl ⟨by linarith [pos_of_mem_seqPts h2], le_half_of_mem_seqPts h2⟩
      · rwa [seqSwap_of_notMem h1 h2]

/-- `0` is fixed. -/
theorem SeqFlip.smul_zero (h : SeqFlip) : h • (0 : ℝ) = 0 := by
  rcases h.eq_one_or with rfl | rfl
  · exact one_smul _ _
  · rw [SeqFlip.gen_smul, seqSwap_of_notMem (fun h ↦ (pos_of_mem_seqPts h).ne' rfl)
      (fun h ↦ by linarith [pos_of_mem_seqPts h])]

/-- `g = 0` is equivariant for `SeqFlip`. -/
theorem isEquivariant_zeroMap_seq : IsEquivariant SeqFlip zeroMap :=
  fun h _ ↦ (h.smul_zero).symm

/-- `N` is an attracting block of `g = 0`. -/
theorem isAttractingBlock_seqBlock : IsAttractingBlock zeroMap seqBlock := by
  refine ⟨isCompact_seqBlock, ?_⟩
  rw [image_zeroMap ⟨0, zero_mem_seqBlock⟩, singleton_subset_iff]
  refine interior_mono subset_union_left ?_
  rw [interior_Icc]
  constructor <;> norm_num

/-- The orbit `[0]` is not an interior point of `N/G` in the quotient topology. An open
neighborhood of `[0]` pulls back to an open invariant subset `V ⊆ N` containing `0`, so `V`
contains `1/n` for large `n`, hence `2 + 1/n`, and with it irrational points near `2 + 1/n`,
which are not in `N`. -/
theorem orbitMap_zero_notMem_interior :
    orbitMap SeqFlip ℝ 0 ∉ interior (orbitMap SeqFlip ℝ '' seqBlock) := by
  intro h0
  set V := orbitMap SeqFlip ℝ ⁻¹' interior (orbitMap SeqFlip ℝ '' seqBlock) with hV
  have hVopen : IsOpen V := isOpen_interior.preimage continuous_orbitMap
  have hVN : V ⊆ seqBlock := fun x hx ↦ by
    have h1 : orbitMap SeqFlip ℝ x ∈ orbitMap SeqFlip ℝ '' seqBlock := interior_subset hx
    have h2 : x ∈ orbitMap SeqFlip ℝ ⁻¹' (orbitMap SeqFlip ℝ '' seqBlock) := h1
    rwa [isGInvariant_seqBlock.preimage_image_orbitMap] at h2
  have hVinv : ∀ x ∈ V, SeqFlip.gen • x ∈ V := fun x hx ↦ by
    rw [hV, mem_preimage, orbitMap_smul]
    exact hx
  obtain ⟨δ, hδ, hball⟩ := Metric.isOpen_iff.mp hVopen 0 h0
  obtain ⟨n, hn⟩ := exists_nat_one_div_lt hδ
  have hy : 1 / ((n : ℝ) + 2) ∈ seqPts := ⟨n, rfl⟩
  have hypos : 0 < 1 / ((n : ℝ) + 2) := pos_of_mem_seqPts hy
  have hyδ : 1 / ((n : ℝ) + 2) < δ :=
    lt_trans (one_div_lt_one_div_of_lt (by positivity) (by linarith)) hn
  have hyV : 1 / ((n : ℝ) + 2) ∈ V := hball (by
    rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_of_pos hypos]
    exact hyδ)
  have hy2V : 1 / ((n : ℝ) + 2) + 2 ∈ V := by
    have := hVinv _ hyV
    rwa [SeqFlip.gen_smul, seqSwap_of_mem hy] at this
  obtain ⟨ε, hε, hballε⟩ := Metric.isOpen_iff.mp hVopen _ hy2V
  obtain ⟨r, hr, hr1, hr2⟩ := exists_irrational_btwn
    (show 1 / ((n : ℝ) + 2) + 2 < 1 / ((n : ℝ) + 2) + 2 + ε by linarith)
  have hrN : r ∈ seqBlock := hVN (hballε (by
    rw [Metric.mem_ball, Real.dist_eq, abs_lt]
    constructor <;> linarith))
  rcases hrN with hrI | hr2' | ⟨m, hm⟩
  · linarith [hrI.2]
  · have : r = 2 := hr2'
    linarith
  · apply hr
    refine ⟨1 / ((m : ℚ) + 2) + 2, ?_⟩
    rw [← hm]
    push_cast
    ring

/-- The conclusion of (i) fails: `ḡ(N/G) = {[0]} ⊄ int(N/G)`. -/
theorem not_image_quotientMap_subset_interior :
    ¬ quotientMap zeroMap isEquivariant_zeroMap_seq '' (orbitMap SeqFlip ℝ '' seqBlock) ⊆
      interior (orbitMap SeqFlip ℝ '' seqBlock) := fun h ↦
  orbitMap_zero_notMem_interior (h ⟨orbitMap SeqFlip ℝ 0, ⟨0, zero_mem_seqBlock, rfl⟩, rfl⟩)

/-- `N/G` is not an attracting block of `ḡ`. -/
theorem not_isAttractingBlock_quotient :
    ¬ IsAttractingBlock (quotientMap zeroMap isEquivariant_zeroMap_seq)
      (orbitMap SeqFlip ℝ '' seqBlock) := fun h ↦
  not_image_quotientMap_subset_interior h.2

/-- The action of `SeqFlip` is not by homeomorphisms: otherwise `IsAttractingBlock.quotient`
would make `N/G` an attracting block. -/
theorem not_continuousConstSMul_seqFlip : ¬ ContinuousConstSMul SeqFlip ℝ := by
  intro h
  exact not_isAttractingBlock_quotient (IsAttractingBlock.quotient isEquivariant_zeroMap_seq
    isGInvariant_seqBlock isAttractingBlock_seqBlock)

/-- The interior of `N` is not invariant: by `IsAttractingBlock.quotient_of_isGInvariant_interior`
this is the hypothesis that fails. Directly, `seqSwap` maps `1/3 ∈ (-1/2, 1/2)` to `2 + 1/3`. -/
theorem not_isGInvariant_interior_seqBlock : ¬ IsGInvariant SeqFlip (interior seqBlock) :=
  fun h ↦ not_isAttractingBlock_quotient
    (IsAttractingBlock.quotient_of_isGInvariant_interior isEquivariant_zeroMap_seq h
      isAttractingBlock_seqBlock)

/-- **Proposition `prop:equivariant_pullback` (i) needs an action by homeomorphisms.** For the
action of `SeqFlip` on `ℝ` by `seqSwap`, the map `g = 0` is continuous and equivariant and `N` is
a `G`-invariant attracting block of `g`, but `N/G` is not an attracting block of `ḡ`: the only
hypothesis of `IsAttractingBlock.quotient` that fails is `ContinuousConstSMul`. -/
theorem equivariant_pullback_i_needs_continuous_action :
    Continuous zeroMap ∧ IsAttractingBlock zeroMap seqBlock ∧ IsGInvariant SeqFlip seqBlock ∧
      ¬ IsAttractingBlock (quotientMap zeroMap isEquivariant_zeroMap_seq)
        (orbitMap SeqFlip ℝ '' seqBlock) ∧
      ¬ ContinuousConstSMul SeqFlip ℝ :=
  ⟨continuous_const, isAttractingBlock_seqBlock, isGInvariant_seqBlock,
    not_isAttractingBlock_quotient, not_continuousConstSMul_seqFlip⟩

/-- The hypothesis `ContinuousConstSMul` cannot be dropped from `IsAttractingBlock.quotient`,
Proposition `prop:equivariant_pullback` (i): the statement without it is false, by
`equivariant_pullback_i_needs_continuous_action`. -/
theorem not_forall_isAttractingBlock_quotient :
    ¬ ∀ (G Z : Type) [Group G] [TopologicalSpace Z] [MulAction G Z] (g : Z → Z)
      (hg : IsEquivariant G g) (N : Set Z), IsGInvariant G N → IsAttractingBlock g N →
      IsAttractingBlock (quotientMap g hg) (orbitMap G Z '' N) := fun h ↦
  not_isAttractingBlock_quotient (h SeqFlip ℝ zeroMap isEquivariant_zeroMap_seq seqBlock
    isGInvariant_seqBlock isAttractingBlock_seqBlock)

/-! ### Part (ii) without invariance of the interior -/

/-- A third group of order two, acting on `ℝ` by negation. -/
def NegFlip : Type := Multiplicative (ZMod 2)

/-- The group structure of `NegFlip`. -/
instance : Group NegFlip := inferInstanceAs (Group (Multiplicative (ZMod 2)))

/-- The generator of `NegFlip`. -/
def NegFlip.gen : NegFlip := Multiplicative.ofAdd (1 : ZMod 2)

/-- `NegFlip = {1, gen}`. -/
theorem NegFlip.eq_one_or (h : NegFlip) : h = 1 ∨ h = NegFlip.gen := by
  rcases zmod_two_cases (Multiplicative.toAdd (h : Multiplicative (ZMod 2))) with h0 | h1
  · exact Or.inl h0
  · exact Or.inr h1

/-- `NegFlip` acts on `ℝ` by `x ↦ -x`. The action is isometric, hence by homeomorphisms. -/
instance : MulAction NegFlip ℝ := involutionAction Neg.neg neg_involutive

/-- The generator acts by `x ↦ -x`. -/
theorem NegFlip.gen_smul (x : ℝ) : NegFlip.gen • x = -x :=
  if_neg (by decide)

/-- The action of `NegFlip` is isometric, so it is by homeomorphisms
(`IsIsometricSMul.to_continuousConstSMul`). -/
instance : IsIsometricSMul NegFlip ℝ :=
  ⟨fun h ↦ by
    rcases h.eq_one_or with rfl | rfl
    · simp only [one_smul]
      exact isometry_id
    · simpa only [NegFlip.gen_smul] using isometry_neg⟩

/-- The latent map `g`: the constant map `2`. It is not equivariant (`not_isEquivariant_twoMap`). -/
def twoMap : ℝ → ℝ := fun _ ↦ 2

/-- The map `f` upstairs, on `X = ℝ`: the constant map `-2`. -/
def negTwoMap : ℝ → ℝ := fun _ ↦ -2

/-- The image of a nonempty set under `g = 2` is `{2}`. -/
theorem image_twoMap {A : Set ℝ} (hA : A.Nonempty) : twoMap '' A = {2} :=
  hA.image_const 2

/-- `N = [1, 3]` is an attracting block of `g = 2`: `g(N) = {2} ⊆ (1, 3) = int N`. -/
theorem isAttractingBlock_Icc_twoMap : IsAttractingBlock twoMap (Icc 1 3) := by
  refine ⟨isCompact_Icc, ?_⟩
  rw [image_twoMap (nonempty_Icc.mpr (by norm_num)), singleton_subset_iff, interior_Icc]
  constructor <;> norm_num

/-- The tolerance is positive, `τ_g(N) > 0`, since `g` is continuous and `N` is an attracting
block of `g` (`tolerance_pos_of_isAttractingBlock`). In fact `τ_g(N) = 1`; only positivity is
used. -/
theorem tolerance_twoMap_pos : 0 < tolerance twoMap (Icc 1 3) :=
  tolerance_pos_of_isAttractingBlock continuous_const.continuousOn isAttractingBlock_Icc_twoMap

/-- For the encoder `E = id`, `E⁻¹(N) = [1, 3]` is compact. -/
theorem isCompact_preimage_id_Icc : IsCompact (id ⁻¹' Icc (1 : ℝ) 3) :=
  isCompact_Icc

/-- The residual modulo `G` vanishes, `r̄(x) = min(d(2, -2), d(2, 2)) = 0` for every `x`: the
generator maps `E(f x) = -2` to `g(E x) = 2`. -/
theorem residualMod_negFlip_eq_zero (x : ℝ) : residualMod NegFlip negTwoMap twoMap id x = 0 := by
  have hgen : dist (twoMap (id x)) (NegFlip.gen • id (negTwoMap x)) = 0 := by
    show dist (2 : ℝ) (NegFlip.gen • (-2 : ℝ)) = 0
    rw [NegFlip.gen_smul, neg_neg, dist_self]
  exact le_antisymm
    ((ciInf_le (bddBelow_range_dist (G := NegFlip) _ _) NegFlip.gen).trans_eq hgen)
    (le_ciInf fun _ ↦ dist_nonneg)

/-- The pointwise hypothesis of (ii): `r̄(x) = 0 < τ_g(N)`. -/
theorem residualMod_negFlip_lt_tolerance (x : ℝ) :
    ENNReal.ofReal (residualMod NegFlip negTwoMap twoMap id x) < tolerance twoMap (Icc 1 3) := by
  rw [residualMod_negFlip_eq_zero, ENNReal.ofReal_zero]
  exact tolerance_twoMap_pos

/-- The hypothesis of (ii) as stated in the paper: `sup_{x ∈ E⁻¹(N)} r̄(x) = 0 < τ_g(N)`. -/
theorem iSup_residualMod_negFlip_lt_tolerance :
    ⨆ x ∈ id ⁻¹' Icc (1 : ℝ) 3, ENNReal.ofReal (residualMod NegFlip negTwoMap twoMap id x) <
      tolerance twoMap (Icc 1 3) := by
  refine lt_of_le_of_lt (iSup₂_le fun x _ ↦ ?_) tolerance_twoMap_pos
  rw [residualMod_negFlip_eq_zero, ENNReal.ofReal_zero]

/-- The conclusion of (ii) fails: `f(1) = -2` does not lie in `int E⁻¹(N) = (1, 3)`. -/
theorem not_isAttractingBlock_preimage_Icc :
    ¬ IsAttractingBlock negTwoMap (id ⁻¹' Icc 1 3) := by
  rintro ⟨-, hsub⟩
  rw [preimage_id, interior_Icc] at hsub
  have h := hsub (mem_image_of_mem negTwoMap (left_mem_Icc.mpr (by norm_num : (1 : ℝ) ≤ 3)))
  have : (1 : ℝ) < -2 := h.1
  norm_num at this

/-- The interior `int N = (1, 3)` is not invariant, since the generator maps `2` to `-2`. By
`isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior` this is the hypothesis that
fails. -/
theorem not_isGInvariant_interior_Icc_negFlip :
    ¬ IsGInvariant NegFlip (interior (Icc (1 : ℝ) 3)) := fun h ↦
  not_isAttractingBlock_preimage_Icc
    (isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior (G := NegFlip)
      continuous_id isCompact_preimage_id_Icc h fun x _ ↦ residualMod_negFlip_lt_tolerance x)

/-- `N = [1, 3]` is not invariant either: for an action by homeomorphisms the interior of an
invariant set is invariant (`IsGInvariant.interior`). -/
theorem not_isGInvariant_Icc_negFlip : ¬ IsGInvariant NegFlip (Icc (1 : ℝ) 3) := fun h ↦
  not_isGInvariant_interior_Icc_negFlip h.interior

/-- `g = 2` is not equivariant: `g(-0) = 2 ≠ -2 = -g(0)`. -/
theorem not_isEquivariant_twoMap : ¬ IsEquivariant NegFlip twoMap := fun h ↦ by
  have := h NegFlip.gen 0
  rw [NegFlip.gen_smul, NegFlip.gen_smul] at this
  norm_num [twoMap] at this

/-- **Proposition `prop:equivariant_pullback` (ii) needs the invariance of `int N`.** For the
isometric action of `NegFlip` on `ℝ` by `x ↦ -x`, the maps `f = -2`, `g = 2` and `E = id` are
continuous, `N = [1, 3]` is an attracting block of `g`, `E⁻¹(N)` is compact, and
`r̄ = 0 < τ_g(N)` on `E⁻¹(N)`, pointwise and for the supremum: every hypothesis of
`isAttractingBlock_preimage_of_residualMod_of_isCompact` except the invariance of `N`. Yet
`E⁻¹(N)` is not an attracting block for `f`. Neither `N` nor `int N = (1, 3)` is invariant, and
`g` is not equivariant. So in the generality of
`isAttractingBlock_preimage_of_residualMod_of_isGInvariant_interior`, which assumes nothing on
`g` or on the action, the invariance of `int N` is needed; for `g` not equivariant, a continuous
isometric action does not replace it. -/
theorem equivariant_pullback_ii_needs_invariant_interior :
    IsIsometricSMul NegFlip ℝ ∧ ContinuousConstSMul NegFlip ℝ ∧ Continuous negTwoMap ∧
      Continuous twoMap ∧ Continuous (id : ℝ → ℝ) ∧ IsAttractingBlock twoMap (Icc 1 3) ∧
      IsCompact (id ⁻¹' Icc (1 : ℝ) 3) ∧
      (∀ x ∈ id ⁻¹' Icc (1 : ℝ) 3,
        ENNReal.ofReal (residualMod NegFlip negTwoMap twoMap id x) < tolerance twoMap (Icc 1 3)) ∧
      ⨆ x ∈ id ⁻¹' Icc (1 : ℝ) 3, ENNReal.ofReal (residualMod NegFlip negTwoMap twoMap id x) <
        tolerance twoMap (Icc 1 3) ∧
      ¬ IsAttractingBlock negTwoMap (id ⁻¹' Icc 1 3) ∧ ¬ IsGInvariant NegFlip (Icc (1 : ℝ) 3) ∧
      ¬ IsGInvariant NegFlip (interior (Icc (1 : ℝ) 3)) ∧ ¬ IsEquivariant NegFlip twoMap :=
  ⟨inferInstance, inferInstance, continuous_const, continuous_const, continuous_id,
    isAttractingBlock_Icc_twoMap, isCompact_preimage_id_Icc,
    fun x _ ↦ residualMod_negFlip_lt_tolerance x, iSup_residualMod_negFlip_lt_tolerance,
    not_isAttractingBlock_preimage_Icc, not_isGInvariant_Icc_negFlip,
    not_isGInvariant_interior_Icc_negFlip, not_isEquivariant_twoMap⟩

/-- The invariance of `N` cannot be dropped from
`isAttractingBlock_preimage_of_residualMod_of_isCompact`, Proposition `prop:equivariant_pullback`
(ii), which does not assume `g` equivariant: the statement without it is false, even with
`IsIsometricSMul G Z` added, by `equivariant_pullback_ii_needs_invariant_interior`. -/
theorem not_forall_isAttractingBlock_preimage_of_residualMod_of_isCompact :
    ¬ ∀ (G X Z : Type) [Group G] [PseudoMetricSpace Z] [MulAction G Z] [ContinuousConstSMul G Z]
      [IsIsometricSMul G Z] [TopologicalSpace X] (f : X → X) (g : Z → Z) (E : X → Z),
      Continuous E → ∀ N : Set Z, IsCompact (E ⁻¹' N) →
      (∀ x ∈ E ⁻¹' N, ENNReal.ofReal (residualMod G f g E x) < tolerance g N) →
      IsAttractingBlock f (E ⁻¹' N) := fun h ↦
  not_isAttractingBlock_preimage_Icc (h NegFlip ℝ ℝ negTwoMap twoMap id continuous_id (Icc 1 3)
    isCompact_preimage_id_Icc fun x _ ↦ residualMod_negFlip_lt_tolerance x)

end ActionSharpness

end LatentEquivariant
