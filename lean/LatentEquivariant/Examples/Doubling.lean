import LatentEquivariant.Orbit.Quotient

/-!
# A repeller on a mirror, as a map

A worked example: the doubling map `f(x) = 2x` on `ℝ`, equivariant for the group of order two
acting by the mirror `x ↦ -x`, with the isolated invariant set `S = {0}`.

The group of order two is `ℤˣ = {1, -1}` (`card_units_int`), acting on `ℝ` through Mathlib's
instances, `u • x = (u : ℤ) • x`, so that `-1` acts by `x ↦ -x` (`neg_one_smul_eq_neg`). Mathlib
provides both `MulAction ℤˣ ℝ` and `ContinuousConstSMul ℤˣ ℝ`, so no local instance is needed and
`IsIndexPair.image_orbitMap` applies directly.

The main results are:

* `isEquivariant_doubling`, `isGInvariant_N`, `isGInvariant_L`: `f`, `N = [-1, 1]` and
  `L = [-1, -1/2] ∪ [1/2, 1]` are compatible with the mirror.
* `Inv_doubling_Icc_half`: `Inv([-1/2, 1/2], f) = {0}`, a special case of `Inv_doubling_eq`.
* `isIndexPair_doubling`: `(N, L)` is an index pair for the map `f` in the sense of Szymczak
  (1995), Definition 3.4. Its exit set is thick. It also satisfies the nontrivial conditions of
  Mrozek (1990), Definition 2.1 (`mrozek_conditions`), with equality
  `N \ L = int N ∩ f⁻¹(int N)` in (2.3) (`N_diff_L_eq_interior_inter_preimage`).
* `not_isIndexPair_thin`: the thin pair `([-1, 1], {-1, 1})`, an index pair for the flow `ẋ = x`
  (not formalized: the library has no flows), is not an index pair for the map `f`. It fails the
  exit condition only (`thin_not_exit`, `thin_other_conditions`): `0.6 ∈ N` and
  `f(0.6) = 1.2 ∉ N`, but `0.6 ∉ {-1, 1}`.
* `isIndexPair_quotient`: `(N/G, L/G)` is an index pair for the induced map `f/G`, and
  `Inv_quotient` shows that it isolates the single orbit `[0]`.
* `image_orbitMap_N`, `image_orbitMap_L`: `N/G = [0, 1]/G` and `L/G = [1/2, 1]/G`. Under the
  homeomorphism `orbitHomeomorph : ℝ/G ≃ₜ [0, ∞)`, `[x] ↦ |x|`, the pair downstairs is
  `([0, 1], [1/2, 1])` (`orbitHomeomorph_image_N`, `orbitHomeomorph_image_L`), and `f/G` is
  `y ↦ 2y` (`orbitAbs_quotientMap`).

The homology computations (upstairs `H₁(N, L) = 𝔽` with `G` acting by `-1`;
downstairs the index is `0`; the sign component) are not formalized here. Neither is the claim
that the thin pair is an index pair for the flow `ẋ = x`.
-/

namespace LatentEquivariant.Doubling

open Set MulAction

/-! ### The mirror -/

/-- The group of order two is `ℤˣ = {1, -1}`. -/
theorem card_units_int : Fintype.card ℤˣ = 2 := by
  decide

/-- The action of `ℤˣ` on `ℝ` is multiplication: `u • x = u x`. -/
theorem units_smul_eq_mul (u : ℤˣ) (x : ℝ) : u • x = ((u : ℤ) : ℝ) * x := by
  rw [Units.smul_def, zsmul_eq_mul]

/-- The nontrivial element `-1` of `ℤˣ` acts on `ℝ` by the mirror `x ↦ -x`. -/
theorem neg_one_smul_eq_neg (x : ℝ) : (-1 : ℤˣ) • x = -x := by
  simp

/-- A subset of `ℝ` is invariant under the mirror if and only if it is symmetric, `-A = A`. -/
theorem isGInvariant_iff_neg {A : Set ℝ} : IsGInvariant ℤˣ A ↔ ∀ x ∈ A, -x ∈ A := by
  constructor
  · intro hA x hx
    have h := hA (-1) x hx
    rwa [neg_one_smul_eq_neg] at h
  · intro hA u x hx
    rcases Int.units_eq_one_or u with rfl | rfl
    · rwa [one_smul]
    · rw [neg_one_smul_eq_neg]
      exact hA x hx

/-- Two points of `ℝ` lie in the same orbit of the mirror if and only if `|x| = |y|`. -/
theorem orbitMap_eq_iff_abs {x y : ℝ} : orbitMap ℤˣ ℝ x = orbitMap ℤˣ ℝ y ↔ |x| = |y| := by
  rw [orbitMap_eq_iff, abs_eq_abs]
  constructor
  · rintro ⟨u, rfl⟩
    rcases Int.units_eq_one_or u with rfl | rfl
    · exact Or.inl (one_smul _ _)
    · exact Or.inr (neg_one_smul_eq_neg _)
  · rintro (rfl | rfl)
    · exact ⟨1, one_smul _ _⟩
    · exact ⟨-1, neg_one_smul_eq_neg _⟩

/-! ### The orbit space `ℝ/G ≅ [0, ∞)` -/

/-- The map `ℝ/G → ℝ`, `[x] ↦ |x|`, which identifies the orbit space of the mirror with
`[0, ∞)`. -/
def orbitAbs : orbitRel.Quotient ℤˣ ℝ → ℝ :=
  Quotient.lift (fun x : ℝ ↦ |x|) fun _ _ hab ↦ orbitMap_eq_iff_abs.mp (Quotient.sound hab)

@[simp] theorem orbitAbs_orbitMap (x : ℝ) : orbitAbs (orbitMap ℤˣ ℝ x) = |x| :=
  rfl

/-- `[x] ↦ |x|` takes values in `[0, ∞)`. -/
theorem orbitAbs_nonneg (q : orbitRel.Quotient ℤˣ ℝ) : 0 ≤ orbitAbs q := by
  obtain ⟨x, rfl⟩ := orbitMap_surjective q
  exact abs_nonneg x

/-- `[x] ↦ |x|` is injective on the orbit space. -/
theorem orbitAbs_injective : Function.Injective orbitAbs := by
  intro p q h
  obtain ⟨x, rfl⟩ := orbitMap_surjective p
  obtain ⟨y, rfl⟩ := orbitMap_surjective q
  exact orbitMap_eq_iff_abs.mpr h

/-- `[x] ↦ |x|` maps the orbit space onto `[0, ∞)`. -/
theorem range_orbitAbs : range orbitAbs = Ici 0 := by
  ext y
  constructor
  · rintro ⟨q, rfl⟩
    exact orbitAbs_nonneg q
  · intro hy
    exact ⟨orbitMap ℤˣ ℝ y, abs_of_nonneg hy⟩

/-- `[x] ↦ |x|` is continuous on the orbit space. -/
theorem continuous_orbitAbs : Continuous orbitAbs :=
  continuous_abs.quotient_lift _

/-- The orbit space of the mirror is homeomorphic to `[0, ∞)` by `[x] ↦ |x|`. -/
def orbitHomeomorph : orbitRel.Quotient ℤˣ ℝ ≃ₜ Ici (0 : ℝ) where
  toFun q := ⟨orbitAbs q, orbitAbs_nonneg q⟩
  invFun y := orbitMap ℤˣ ℝ y
  left_inv q := by
    obtain ⟨x, rfl⟩ := orbitMap_surjective q
    exact orbitMap_eq_iff_abs.mpr (abs_abs x)
  right_inv y := Subtype.ext (abs_of_nonneg y.2)
  continuous_toFun := continuous_orbitAbs.subtype_mk _
  continuous_invFun := continuous_orbitMap.comp continuous_subtype_val

@[simp] theorem orbitHomeomorph_apply (q : orbitRel.Quotient ℤˣ ℝ) :
    (orbitHomeomorph q : ℝ) = orbitAbs q :=
  rfl

/-- The image of a set of orbits under `orbitHomeomorph`, in the coordinate `y ∈ [0, ∞)`. -/
theorem orbitHomeomorph_image (A : Set (orbitRel.Quotient ℤˣ ℝ)) :
    orbitHomeomorph '' A = {y : Ici (0 : ℝ) | (y : ℝ) ∈ orbitAbs '' A} := by
  ext y
  constructor
  · rintro ⟨q, hq, rfl⟩
    exact ⟨q, hq, rfl⟩
  · rintro ⟨q, hq, hqy⟩
    exact ⟨q, hq, Subtype.ext hqy⟩

/-- The image of `A/G` under `[x] ↦ |x|` is `|A|`. -/
theorem orbitAbs_image_orbitMap (A : Set ℝ) : orbitAbs '' (orbitMap ℤˣ ℝ '' A) = abs '' A := by
  rw [image_image]
  rfl

/-- Two subsets of `ℝ` have the same image in the orbit space of the mirror if and only if they
have the same image under `x ↦ |x|`. -/
theorem image_orbitMap_eq_iff_abs {A B : Set ℝ} :
    orbitMap ℤˣ ℝ '' A = orbitMap ℤˣ ℝ '' B ↔ abs '' A = abs '' B := by
  constructor
  · intro hAB
    rw [← orbitAbs_image_orbitMap, ← orbitAbs_image_orbitMap, hAB]
  · intro hAB
    apply orbitAbs_injective.image_injective
    rw [orbitAbs_image_orbitMap, orbitAbs_image_orbitMap, hAB]

/-- `|A| = A` for `A ⊆ [0, ∞)`. -/
theorem abs_image_of_nonneg {A : Set ℝ} (hA : ∀ x ∈ A, 0 ≤ x) : abs '' A = A := by
  ext y
  constructor
  · rintro ⟨x, hx, rfl⟩
    rwa [abs_of_nonneg (hA x hx)]
  · intro hy
    exact ⟨y, hy, abs_of_nonneg (hA y hy)⟩

/-! ### The doubling map and the index pairs -/

/-- The doubling map `f(x) = 2x` on `ℝ`. Its only bounded solution is `0`, a repeller. -/
def doubling : ℝ → ℝ := fun x ↦ 2 * x

@[simp] theorem doubling_apply (x : ℝ) : doubling x = 2 * x :=
  rfl

/-- The doubling map is continuous. -/
theorem continuous_doubling : Continuous doubling :=
  continuous_const.mul continuous_id

/-- The iterates of the doubling map: `fⁿ(x) = 2ⁿ x`. -/
theorem doubling_iterate (n : ℕ) (x : ℝ) : doubling^[n] x = 2 ^ n * x := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih, doubling_apply, pow_succ]
    ring

/-- The isolating neighborhood `N = [-1, 1]` of the example. -/
def N : Set ℝ := Icc (-1) 1

/-- The thick exit set `L = [-1, -1/2] ∪ [1/2, 1]` of the example. -/
def L : Set ℝ := Icc (-1) (-1/2) ∪ Icc (1/2) 1

/-- The doubling map commutes with the mirror, `f(-x) = -f(x)`. -/
theorem isEquivariant_doubling : IsEquivariant ℤˣ doubling := by
  intro u x
  simp only [doubling_apply, units_smul_eq_mul]
  ring

/-- `N = [-1, 1]` is invariant under the mirror. -/
theorem isGInvariant_N : IsGInvariant ℤˣ N := by
  refine isGInvariant_iff_neg.mpr fun x hx ↦ ?_
  simp only [N, mem_Icc] at hx ⊢
  constructor <;> linarith [hx.1, hx.2]

/-- `L = [-1, -1/2] ∪ [1/2, 1]` is invariant under the mirror. -/
theorem isGInvariant_L : IsGInvariant ℤˣ L := by
  refine isGInvariant_iff_neg.mpr fun x hx ↦ ?_
  simp only [L, mem_union, mem_Icc] at hx ⊢
  rcases hx with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · exact Or.inr ⟨by linarith, by linarith⟩
  · exact Or.inl ⟨by linarith, by linarith⟩

/-- A point whose forward orbit under the doubling map stays in a bounded set is `0`. -/
theorem eq_zero_of_forall_iterate_mem {A : Set ℝ} (hA : Bornology.IsBounded A) {x : ℝ}
    (hx : ∀ n : ℕ, doubling^[n] x ∈ A) : x = 0 := by
  obtain ⟨C, hC⟩ := hA.exists_norm_le
  by_contra hx0
  have hx0' : 0 < |x| := abs_pos.mpr hx0
  obtain ⟨n, hn⟩ := pow_unbounded_of_one_lt (C / |x|) (by norm_num : (1 : ℝ) < 2)
  have h1 := hC _ (hx n)
  rw [doubling_iterate, Real.norm_eq_abs, abs_mul, abs_pow, abs_two] at h1
  rw [div_lt_iff₀ hx0'] at hn
  linarith

/-- Every solution of the doubling map in a bounded set is the zero solution:
`Inv(A, f) ⊆ {0}` for every bounded `A ⊆ ℝ`. -/
theorem Inv_doubling_subset {A : Set ℝ} (hA : Bornology.IsBounded A) :
    Inv doubling A ⊆ {0} := by
  rintro x ⟨γ, hγ, rfl, hγA⟩
  refine eq_zero_of_forall_iterate_mem hA fun n ↦ ?_
  rw [hγ.iterate n]
  exact hγA n

/-- `Inv(A, f) = {0}` for every bounded `A ⊆ ℝ` that contains `0`: the fixed point `0` is the only
bounded solution of the doubling map. -/
theorem Inv_doubling_eq {A : Set ℝ} (hA : Bornology.IsBounded A) (h0 : (0 : ℝ) ∈ A) :
    Inv doubling A = {0} := by
  refine Subset.antisymm (Inv_doubling_subset hA) ?_
  rintro _ rfl
  exact ⟨fun _ ↦ 0, fun _ ↦ by simp, rfl, fun _ ↦ h0⟩

/-- `Inv([-1/2, 1/2], f) = {0}`. -/
theorem Inv_doubling_Icc_half : Inv doubling (Icc (-1/2) (1/2)) = {0} :=
  Inv_doubling_eq (Metric.isBounded_Icc _ _) ⟨by norm_num, by norm_num⟩

/-- `N = [-1, 1]` is an isolating neighborhood of `S = {0}`: `Inv(N, f) = {0} ⊆ int N`
(Szymczak 1995, Definition 3.2). -/
theorem isIsolatingNeighborhood_N : IsIsolatingNeighborhood doubling N := by
  refine ⟨isCompact_Icc, ?_⟩
  rw [N, Inv_doubling_eq (Metric.isBounded_Icc _ _) ⟨by norm_num, by norm_num⟩, interior_Icc]
  rintro _ rfl
  constructor <;> norm_num

/-- `N \ L = (-1/2, 1/2)`. -/
theorem N_diff_L : N \ L = Ioo (-1/2) (1/2) := by
  ext x
  simp only [N, L, mem_sdiff, mem_Icc, mem_union, mem_Ioo, not_or, not_and, not_le]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3, h4⟩
    refine ⟨h3 h1, ?_⟩
    by_contra h
    linarith [h4 (le_of_not_gt h)]
  · rintro ⟨h1, h2⟩
    refine ⟨⟨by linarith, by linarith⟩, fun _ ↦ h1, fun h ↦ absurd h (not_le.mpr h2)⟩

/-- The equality in Mrozek (1990), Definition 2.1, (2.3):
`N \ L = (-1/2, 1/2) = int N ∩ f⁻¹(int N)`. -/
theorem N_diff_L_eq_interior_inter_preimage :
    N \ L = interior N ∩ doubling ⁻¹' interior N := by
  rw [N_diff_L, N, interior_Icc]
  ext x
  simp only [mem_Ioo, mem_inter_iff, mem_preimage, doubling_apply]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨⟨by linarith, by linarith⟩, by linarith, by linarith⟩
  · rintro ⟨-, h1, h2⟩
    exact ⟨by linarith, by linarith⟩

/-- Mrozek (1990), Definition 2.1, for `(P₁, P₂) = (N, L)` in the isolating neighborhood `N`:
`L ⊆ N`; (2.1) `L` is positively invariant with respect to `N`, `L ∩ f⁻¹(N) ⊆ f⁻¹(L)`;
(2.2) the positively invariant part `Inv⁺ N = {x | fⁿ(x) ∈ N for all n ≥ 0}` lies in `N \ L`;
(2.3) `N \ L ⊆ int N ∩ f⁻¹(int N)`, with equality by `N_diff_L_eq_interior_inter_preimage`.
The remaining parts of (2.1) and (2.2) are not stated: for `P₁ = N` they read
`N ∩ f⁻¹(N) ⊆ f⁻¹(N)`, which is trivial, and `Inv⁻ N ⊆ int_N N = N`, which holds because
Mrozek's `ℤ⁻` contains `0`. Mrozek's `Inv⁻` uses the negative iterates of `f`, and his standing
assumption is that `f` is a homeomorphism, which the doubling map is (not stated here).
Compactness of `N` and `L` is in `isIndexPair_doubling`. -/
theorem mrozek_conditions :
    L ⊆ N ∧ L ∩ doubling ⁻¹' N ⊆ doubling ⁻¹' L ∧
      {x | ∀ n : ℕ, doubling^[n] x ∈ N} ⊆ N \ L ∧
      N \ L ⊆ interior N ∩ doubling ⁻¹' interior N := by
  refine ⟨?_, ?_, ?_, N_diff_L_eq_interior_inter_preimage.subset⟩
  · intro x hx
    simp only [L, N, mem_union, mem_Icc] at hx ⊢
    rcases hx with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · exact ⟨h1, by linarith⟩
    · exact ⟨by linarith, h2⟩
  · rintro x ⟨hx, hfx⟩
    simp only [L, N, mem_union, mem_Icc, mem_preimage, doubling_apply] at hx hfx ⊢
    rcases hx with ⟨-, h2⟩ | ⟨h1, -⟩
    · exact Or.inl ⟨hfx.1, by linarith⟩
    · exact Or.inr ⟨by linarith, hfx.2⟩
  · intro x hx
    rw [eq_zero_of_forall_iterate_mem (Metric.isBounded_Icc (-1 : ℝ) 1) hx, N_diff_L]
    constructor <;> norm_num

/-- `Inv(cl(N \ L), f) = {0}`: the index pair `(N, L)` is an index pair for `S = {0}`. -/
theorem Inv_closure_N_diff_L : Inv doubling (closure (N \ L)) = {0} := by
  rw [N_diff_L, closure_Ioo (by norm_num : (-1/2 : ℝ) ≠ 1/2), Inv_doubling_Icc_half]

/-- `(N, L) = ([-1, 1], [-1, -1/2] ∪ [1/2, 1])` is an index pair for the doubling map in the
sense of Szymczak (1995), Definition 3.4. The exit set `L` is thick: it contains every point of
`N` that leaves `N` in one step. -/
theorem isIndexPair_doubling : IsIndexPair doubling N L where
  isCompact_left := isCompact_Icc
  isCompact_right := isCompact_Icc.union isCompact_Icc
  isolating := by
    rw [Inv_closure_N_diff_L, N_diff_L, interior_Ioo]
    rintro _ rfl
    constructor <;> norm_num
  posInv := by
    rw [N_diff_L]
    intro x hx hfx
    simp only [L, mem_union, mem_Icc] at hx
    simp only [mem_Ioo, doubling_apply] at hfx
    rcases hx with ⟨-, h⟩ | ⟨h, -⟩ <;> linarith [hfx.1, hfx.2]
  exit := by
    intro x hx hfx
    simp only [N, mem_Icc, doubling_apply, not_and_or, not_le] at hx hfx
    simp only [L, mem_union, mem_Icc]
    rcases hfx with h | h
    · exact Or.inl ⟨hx.1, by linarith⟩
    · exact Or.inr ⟨by linarith, hx.2⟩

/-! ### The thin exit set -/

/-- The thin exit set `{-1, 1}` is not an exit set for the doubling map. The point `0.6 ∈ N`
leaves `N`, since `f(0.6) = 1.2 ∉ N`, but `0.6 ∉ {-1, 1}`. -/
theorem thin_not_exit : ¬ ∀ x ∈ N, doubling x ∉ N → x ∈ ({-1, 1} : Set ℝ) := by
  intro h
  have h06 := h 0.6 (by norm_num [N]) (by norm_num [N])
  norm_num at h06

/-- The thin pair `([-1, 1], {-1, 1})` satisfies every condition of Szymczak (1995),
Definition 3.4, except the exit condition. -/
theorem thin_other_conditions :
    IsCompact N ∧ IsCompact ({-1, 1} : Set ℝ) ∧
      Inv doubling (closure (N \ {-1, 1})) ⊆ interior (N \ {-1, 1}) ∧
      ∀ x ∈ ({-1, 1} : Set ℝ), doubling x ∉ N \ {-1, 1} := by
  refine ⟨isCompact_Icc, (toFinite _).isCompact, ?_, ?_⟩
  · rw [N, Icc_sdiff_both, closure_Ioo (by norm_num : (-1 : ℝ) ≠ 1), interior_Ioo,
      Inv_doubling_eq (Metric.isBounded_Icc _ _) ⟨by norm_num, by norm_num⟩]
    rintro _ rfl
    constructor <;> norm_num
  · rintro x (rfl | rfl) ⟨hN, -⟩ <;> norm_num [N] at hN

/-- The thin pair `([-1, 1], {-1, 1})`, an index pair for the flow `ẋ = x` (not formalized; the
library has no flows), is not an index pair for its time-`log 2` map, the doubling map. By
`thin_other_conditions`, only the exit condition fails. -/
theorem not_isIndexPair_thin : ¬ IsIndexPair doubling N {-1, 1} :=
  fun h ↦ thin_not_exit h.exit

/-! ### The orbit space -/

/-- `(N/G, L/G)` is an index pair for the map `f/G` induced on the orbit space of the mirror,
by `IsIndexPair.image_orbitMap`. -/
theorem isIndexPair_quotient :
    IsIndexPair (quotientMap doubling isEquivariant_doubling) (orbitMap ℤˣ ℝ '' N)
      (orbitMap ℤˣ ℝ '' L) :=
  isIndexPair_doubling.image_orbitMap isEquivariant_doubling isGInvariant_N isGInvariant_L

/-- The index pair `(N/G, L/G)` isolates the single orbit `[0]`:
`Inv(cl(N/G \ L/G), f/G) = {[0]}`. -/
theorem Inv_quotient :
    Inv (quotientMap doubling isEquivariant_doubling)
      (closure (orbitMap ℤˣ ℝ '' N \ orbitMap ℤˣ ℝ '' L)) = {orbitMap ℤˣ ℝ 0} := by
  have hNL : IsGInvariant ℤˣ (N \ L) := isGInvariant_N.diff isGInvariant_L
  rw [← isGInvariant_L.image_diff, ← hNL.image_closure,
    Inv_quotientMap isEquivariant_doubling hNL.closure, Inv_closure_N_diff_L, image_singleton]

/-- `|N| = [0, 1]`. -/
theorem abs_image_N : abs '' N = Icc 0 1 := by
  ext y
  constructor
  · rintro ⟨x, ⟨h1, h2⟩, rfl⟩
    exact ⟨abs_nonneg x, abs_le.mpr ⟨h1, h2⟩⟩
  · rintro ⟨h1, h2⟩
    exact ⟨y, ⟨by linarith, h2⟩, abs_of_nonneg h1⟩

/-- `|L| = [1/2, 1]`. -/
theorem abs_image_L : abs '' L = Icc (1/2) 1 := by
  ext y
  constructor
  · rintro ⟨x, (⟨h1, h2⟩ | ⟨h1, h2⟩), rfl⟩
    · rw [abs_of_neg (by linarith)]
      constructor <;> linarith
    · rw [abs_of_pos (by linarith)]
      exact ⟨h1, h2⟩
  · rintro ⟨h1, h2⟩
    exact ⟨y, Or.inr ⟨h1, h2⟩, abs_of_pos (by linarith)⟩

/-- `N/G = [0, 1]/G`: every orbit meeting `N = [-1, 1]` meets `[0, 1]`. -/
theorem image_orbitMap_N : orbitMap ℤˣ ℝ '' N = orbitMap ℤˣ ℝ '' Icc 0 1 := by
  rw [image_orbitMap_eq_iff_abs, abs_image_N, abs_image_of_nonneg fun x hx ↦ hx.1]

/-- `L/G = [1/2, 1]/G`: every orbit meeting `L = [-1, -1/2] ∪ [1/2, 1]` meets `[1/2, 1]`. -/
theorem image_orbitMap_L : orbitMap ℤˣ ℝ '' L = orbitMap ℤˣ ℝ '' Icc (1/2) 1 := by
  rw [image_orbitMap_eq_iff_abs, abs_image_L, abs_image_of_nonneg fun x hx ↦ by linarith [hx.1]]

/-- Under the identification `ℝ/G ≃ₜ [0, ∞)`, `[x] ↦ |x|`, the set `N/G` is `[0, 1]`. -/
theorem orbitHomeomorph_image_N :
    orbitHomeomorph '' (orbitMap ℤˣ ℝ '' N) = {y : Ici (0 : ℝ) | (y : ℝ) ∈ Icc 0 1} := by
  rw [orbitHomeomorph_image, orbitAbs_image_orbitMap, abs_image_N]

/-- Under the identification `ℝ/G ≃ₜ [0, ∞)`, `[x] ↦ |x|`, the set `L/G` is `[1/2, 1]`. -/
theorem orbitHomeomorph_image_L :
    orbitHomeomorph '' (orbitMap ℤˣ ℝ '' L) = {y : Ici (0 : ℝ) | (y : ℝ) ∈ Icc (1/2) 1} := by
  rw [orbitHomeomorph_image, orbitAbs_image_orbitMap, abs_image_L]

/-- Downstairs the map is again the doubling map: `|f(x)| = 2|x|`, so under `[x] ↦ |x|` the map
`f/G` is `y ↦ 2y` on `[0, ∞)`, with the repelling fixed point `0` on the boundary. -/
theorem orbitAbs_quotientMap (q : orbitRel.Quotient ℤˣ ℝ) :
    orbitAbs (quotientMap doubling isEquivariant_doubling q) = 2 * orbitAbs q := by
  obtain ⟨x, rfl⟩ := orbitMap_surjective q
  rw [quotientMap_orbitMap, orbitAbs_orbitMap, orbitAbs_orbitMap, doubling_apply, abs_mul,
    abs_two]

end LatentEquivariant.Doubling
