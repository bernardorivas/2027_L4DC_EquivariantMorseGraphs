import LatentEquivariant.Latent.Pullback

/-!
# Limits of the weakened hypotheses

Counterexamples that bound the weakenings recorded in `lean/README.md`.

* `main_simple_discontinuous`: Theorem `thm:main_simple` of the latent-dynamics paper holds
  without continuity of `f` (`isAttractingBlock_preimage_of_forall`), but then its conclusion has
  no dynamical content. For a discontinuous `f` on the compact space `ℕ ∪ {∞}`, the whole space is
  an attracting block of `f` whose maximal invariant set is empty, so the corollary of that paper
  that `Inv(E⁻¹(N), f)` is nonempty fails.
* `isAttractingBlock_gapMap`, `tolerance_gapMap`: without continuity of `g`, an attracting block
  can have tolerance `0`, and then `thm:main_simple` says nothing. For continuous `g` the
  tolerance of an attracting block is positive (`tolerance_pos_of_isAttractingBlock`).
* `quotientDist_eq_zero`: for a group whose orbits are not closed, the quotient distance
  `d̄([a],[b]) = inf_h d(a, h b)` vanishes between distinct orbits, so it is only a pseudometric.
  The example is `ℚ` acting on `ℝ` by translations, an isometric action.
-/

namespace LatentEquivariant.Sharpness

open Set Topology OnePoint MulAction
open scoped ENNReal

/-! ### A discontinuous map upstairs -/

/-- `n ↦ n + 1` on `ℕ ∪ {∞}` with `∞ ↦ 0`. It is not continuous at `∞`. -/
def shiftInfty : OnePoint ℕ → OnePoint ℕ
  | OnePoint.infty => ((0 : ℕ) : OnePoint ℕ)
  | (n : ℕ) => ((n + 1 : ℕ) : OnePoint ℕ)

theorem shiftInfty_ne_infty (x : OnePoint ℕ) : shiftInfty x ≠ OnePoint.infty := by
  cases x <;> simp [shiftInfty]

/-- `shiftInfty` has no full solution: going back along a solution would decrease a natural
number forever. -/
theorem Inv_shiftInfty_univ : Inv shiftInfty (univ : Set (OnePoint ℕ)) = ∅ := by
  refine eq_empty_iff_forall_notMem.mpr ?_
  rintro x ⟨γ, hγ, -, -⟩
  have hne : ∀ m : ℤ, γ m ≠ OnePoint.infty := fun m ↦ by
    have h := hγ (m - 1)
    rw [sub_add_cancel] at h
    rw [← h]
    exact shiftInfty_ne_infty _
  choose a ha using fun m ↦ ne_infty_iff_exists.mp (hne m)
  have hstep : ∀ m : ℤ, a (m + 1) = a m + 1 := fun m ↦ by
    have h := hγ m
    rw [← ha m, ← ha (m + 1)] at h
    have h' : ((a m + 1 : ℕ) : OnePoint ℕ) = a (m + 1) := h
    exact (coe_injective h').symm
  have key : ∀ k : ℕ, a 0 = a (-(k : ℤ)) + k := fun k ↦ by
    induction k with
    | zero => simp
    | succ k ih =>
      have h := hstep (-((k + 1 : ℕ) : ℤ))
      have e : -((k + 1 : ℕ) : ℤ) + 1 = -(k : ℤ) := by push_cast; ring
      rw [e] at h
      omega
  have := key (a 0 + 1)
  omega

theorem not_continuous_shiftInfty : ¬ Continuous shiftInfty := by
  intro h
  have hopen : IsOpen ({((0 : ℕ) : OnePoint ℕ)} : Set (OnePoint ℕ)) := by
    have := (isOpen_image_coe (X := ℕ) (s := {0})).mpr (isOpen_discrete _)
    simpa using this
  have hpre : shiftInfty ⁻¹' {((0 : ℕ) : OnePoint ℕ)} = {OnePoint.infty} := by
    ext x
    cases x <;> simp [shiftInfty]
  have h1 := hopen.preimage h
  rw [hpre, isOpen_iff_of_mem' (mem_singleton _)] at h1
  have h2 : ((↑) ⁻¹' ({OnePoint.infty} : Set (OnePoint ℕ)) : Set ℕ)ᶜ = univ := by
    ext n
    simp
  rw [h2] at h1
  exact noncompact_univ ℕ h1.1

theorem tolerance_id_univ_pos : (0 : ℝ≥0∞) < tolerance (id : Unit → Unit) univ := by
  refine lt_of_lt_of_le (ENNReal.ofReal_pos.mpr one_pos) ?_
  unfold tolerance
  exact le_iSup₂_of_le (1 : ℝ) ⟨one_pos, subset_univ _⟩ le_rfl

/-- `thm:main_simple` without continuity of `f` applies, but its conclusion has no dynamical
content: with `Z` a point, `g = id` and `E` constant, the hypotheses of
`isAttractingBlock_preimage_of_forall` hold for the discontinuous `shiftInfty`, and `E⁻¹(N)` is a
nonempty attracting block whose maximal invariant set is empty. -/
theorem main_simple_discontinuous :
    IsAttractingBlock shiftInfty ((fun _ : OnePoint ℕ ↦ ()) ⁻¹' (univ : Set Unit)) ∧
      ((fun _ : OnePoint ℕ ↦ ()) ⁻¹' (univ : Set Unit)).Nonempty ∧
      Inv shiftInfty ((fun _ : OnePoint ℕ ↦ ()) ⁻¹' (univ : Set Unit)) = ∅ := by
  refine ⟨isAttractingBlock_preimage_of_forall (g := id) continuous_const
    ⟨isCompact_univ, by simp⟩ fun x _ ↦ ?_, ⟨OnePoint.infty, trivial⟩,
    by simpa using Inv_shiftInfty_univ⟩
  simpa using tolerance_id_univ_pos

/-! ### A discontinuous latent map -/

/-- `g = id` on `(0, 1)` and `g = 1/2` elsewhere. -/
noncomputable def gapMap (x : ℝ) : ℝ :=
  if x ∈ Ioo 0 1 then x else 1 / 2

/-- `[0, 1]` is an attracting block for the discontinuous map `gapMap`: its image is `(0, 1)`. -/
theorem isAttractingBlock_gapMap : IsAttractingBlock gapMap (Icc 0 1) := by
  refine ⟨isCompact_Icc, ?_⟩
  rw [interior_Icc]
  rintro _ ⟨x, -, rfl⟩
  unfold gapMap
  split_ifs with h
  · exact h
  · constructor <;> norm_num

/-- The tolerance of `[0, 1]` for `gapMap` is `0`: every neighborhood of `gapMap([0, 1]) = (0, 1)`
leaves `[0, 1]`. So without continuity of `g`, `thm:main_simple` can be vacuous. -/
theorem tolerance_gapMap : tolerance gapMap (Icc 0 1) = 0 := by
  apply le_antisymm _ zero_le
  unfold tolerance
  refine iSup₂_le fun ν h ↦ ?_
  obtain ⟨hν, hsub⟩ := h
  exfalso
  set m := min (ν / 4) (1 / 2) with hm
  have hm0 : 0 < m := lt_min (by linarith) (by norm_num)
  have hmν : m ≤ ν / 4 := min_le_left _ _
  have hm1 : m ≤ 1 / 2 := min_le_right _ _
  have hmI : m ∈ Ioo (0 : ℝ) 1 := ⟨hm0, by linarith⟩
  have hmem : -(ν / 2) ∈ Metric.thickening ν (gapMap '' Icc 0 1) := by
    rw [Metric.mem_thickening_iff]
    refine ⟨gapMap m, ⟨m, ⟨hm0.le, by linarith⟩, rfl⟩, ?_⟩
    rw [gapMap, if_pos hmI, Real.dist_eq, abs_lt]
    constructor <;> linarith
  have h0 := (hsub hmem).1
  linarith

/-! ### A group with dense orbits -/

/-- `ℚ`, as a multiplicative group, acting on `ℝ` by translations. -/
def RatShift : Type := Multiplicative ℚ

instance : Group RatShift := inferInstanceAs (Group (Multiplicative ℚ))

noncomputable instance : MulAction RatShift ℝ where
  smul q x := x + ((Multiplicative.toAdd (q : Multiplicative ℚ) : ℚ) : ℝ)
  one_smul x := by
    show x + ((Multiplicative.toAdd (1 : Multiplicative ℚ) : ℚ) : ℝ) = x
    simp
  mul_smul p q x := by
    change x + ((Multiplicative.toAdd (p : Multiplicative ℚ) +
        Multiplicative.toAdd (q : Multiplicative ℚ) : ℚ) : ℝ) =
      (x + ((Multiplicative.toAdd (q : Multiplicative ℚ) : ℚ) : ℝ)) +
        ((Multiplicative.toAdd (p : Multiplicative ℚ) : ℚ) : ℝ)
    push_cast
    ring

theorem ratShift_smul_def (q : RatShift) (x : ℝ) :
    q • x = x + ((Multiplicative.toAdd (q : Multiplicative ℚ) : ℚ) : ℝ) :=
  rfl

instance : IsIsometricSMul RatShift ℝ :=
  ⟨fun q ↦ Isometry.of_dist_eq fun x y ↦ by
    simp only [ratShift_smul_def, Real.dist_eq]
    congr 1
    ring⟩

/-- The quotient distance vanishes between the distinct orbits of `0` and `√2`, so for an
isometric action with orbits that are not closed it is only a pseudometric. -/
theorem quotientDist_eq_zero :
    quotientDist RatShift (orbitMap RatShift ℝ 0) (orbitMap RatShift ℝ (Real.sqrt 2)) = 0 ∧
      orbitMap RatShift ℝ 0 ≠ orbitMap RatShift ℝ (Real.sqrt 2) := by
  refine ⟨?_, ?_⟩
  · rw [quotientDist_orbitMap]
    apply le_antisymm
    · refine le_of_forall_pos_lt_add fun ε hε ↦ ?_
      obtain ⟨r, hr1, hr2⟩ :=
        exists_rat_btwn (show -Real.sqrt 2 - ε < -Real.sqrt 2 + ε by linarith)
      refine (ciInf_le (bddBelow_range_dist (G := RatShift) 0 (Real.sqrt 2))
        (Multiplicative.ofAdd r : Multiplicative ℚ)).trans_lt ?_
      rw [ratShift_smul_def, Real.dist_eq, abs_lt]
      simp only [toAdd_ofAdd]
      constructor <;> linarith
    · exact le_ciInf fun _ ↦ dist_nonneg
  · intro h
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp h
    rw [ratShift_smul_def] at hg
    have : Real.sqrt 2 = ((-(Multiplicative.toAdd (g : Multiplicative ℚ)) : ℚ) : ℝ) := by
      push_cast
      linarith
    exact irrational_sqrt_two.ne_rat _ this

end LatentEquivariant.Sharpness
