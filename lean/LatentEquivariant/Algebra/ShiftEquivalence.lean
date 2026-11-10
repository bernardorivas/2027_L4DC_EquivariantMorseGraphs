import Mathlib
import LatentEquivariant.Algebra.Isotypic

/-!
# Shift equivalence of index maps

For a map `f : X → X`, the Conley index of an isolated invariant set `S` is the shift equivalence
class of the index map `f_(N,L)` on `H_*(N, L)` (Szymczak 1995, Franks-Richeson 2000). Mrozek's
index is the Leray reduction of the index map in cohomology (Mrozek 1990, Definition 2.5 and
Theorem 2.6, for homeomorphisms). This file formalizes the linear algebra of these notions for one
module, that is, for one degree of `H_*(N, L)`. Most statements hold over any semiring `k` and for
modules of any dimension. The modules `V` and `W` may lie in different universes.

## Main statements

* (a) `IsShiftEquivOfLag A B m`, `ShiftEquivalent A B`, `IsConjugate A B`. Shift equivalence is
  reflexive, symmetric and transitive, and the lags add (`IsShiftEquivOfLag.refl`,
  `IsShiftEquivOfLag.symm`, `IsShiftEquivOfLag.trans`). A lag can be raised
  (`IsShiftEquivOfLag.mono`). Conjugacy is shift equivalence with lag `0`
  (`isShiftEquivOfLag_zero_iff`).
* (b) The trivial index. If `W = 0`, then `A` and `B` are shift equivalent if and only if `A` is
  nilpotent (`shiftEquivalent_iff_isNilpotent`), and nilpotency is an invariant of the shift
  class (`ShiftEquivalent.isNilpotent_iff`). The paper assumes `H_*(N, L)` finite dimensional over
  a field. Neither hypothesis is needed.
* (c) Automorphisms. Shift equivalent automorphisms are conjugate, over any semiring
  (`shiftEquivalent_iff_isConjugate_of_isUnit`). It suffices that `A` is injective and `B` is
  surjective (`IsShiftEquivOfLag.isConjugate`), although this covers no new pairs: the conclusion
  makes both maps bijective. Neither half can be dropped: the identity of `k`
  and the idempotent `P` below are shift equivalent in both orders, `P` is neither injective nor
  surjective, and they are not conjugate (`isShiftEquivOfLag_one_inl_comp_fst`,
  `not_injective_inl_comp_fst`, `not_surjective_inl_comp_fst`,
  `not_isConjugate_one_inl_comp_fst`).
* (d) The sentence "over a field the shift equivalence class of an automorphism is its conjugacy
  class" is false as worded. Over every nontrivial semiring the identity of `k` is shift
  equivalent with lag `1` to the idempotent `P = inl ∘ fst` of `k × k`, which is not invertible
  (`isShiftEquivOfLag_one_inl_comp_fst`, `not_isUnit_inl_comp_fst`,
  `not_isConjugate_one_inl_comp_fst`). What the paper uses is (c): two identities are shift
  equivalent if and only if the modules are isomorphic (`shiftEquivalent_one_iff`), so, for free
  modules over a ring with the strong rank condition, for instance over a field, if and only if
  the ranks agree (`shiftEquivalent_one_iff_lift_rank_eq`, `shiftEquivalent_one_fin_iff`).
* (e) The Leray reduction. `A` is shift equivalent with lag `n` to its restriction to
  `range (A ^ n)` (`isShiftEquivOfLag_restrict_range_pow`). `lerayReduction A` is the restriction
  of `A` to its eventual range `eventualRange A = ⋂ n, range (A ^ n)`. On an Artinian module `A` is
  shift equivalent to it (`shiftEquivalent_lerayReduction`) and it is surjective; on a Noetherian
  module over a ring it is injective. For `V` Noetherian and Artinian the restriction of `A` to
  `range (A ^ n)` is bijective for `n` large (`eventually_bijective_restrict_range_pow`, Fitting).
  For `V` Noetherian and Artinian and `W` Artinian, `A` and `B` are shift equivalent if and only
  if their Leray reductions are conjugate (`shiftEquivalent_iff_isConjugate_lerayReduction`,
  finite-dimensional case `shiftEquivalent_iff_isConjugate_lerayReduction_of_finiteDimensional`).
  So the eventual range, not the module, is an invariant of the shift class up to isomorphism
  (`ShiftEquivalent.nonempty_linearEquiv_eventualRange`,
  `ShiftEquivalent.finrank_eventualRange_eq`).
  The correct form of the sentence of (d): if `A` is an automorphism and `W` is Artinian, then
  `B` is shift equivalent to `A` if and only if the Leray reduction of `B` is conjugate to `A`
  (`shiftEquivalent_iff_isConjugate_lerayReduction_of_injective`). The theorem assumes only `A`
  injective, but an injective `A` shift equivalent to an endomorphism of an Artinian module is
  surjective (`ShiftEquivalent.surjective_of_injective`), so no new cases arise. For an
  automorphism `A` the forward direction needs no hypothesis on `W`
  (`isConjugate_lerayReduction_of_bijective`). For an identity index map:
  `shiftEquivalent_one_iff_lerayReduction`, `shiftEquivalent_one_iff_finrank`.
  Mrozek's Leray functor is the composite `L_I ∘ L_M` (Mrozek 1990, Section 4). `L_M` passes to
  the map induced on `V / ⋃ n, ker (A ^ n)`, which is `lerayQuotient`, and `L_I` restricts that map
  to its generalized image. On modules of finite length `lerayQuotient` is already bijective, so it
  is Mrozek's Leray reduction, and it is conjugate to `lerayReduction`
  (`isConjugate_lerayReduction_lerayQuotient`, `shiftEquivalent_iff_isConjugate_lerayQuotient`).
* (f) Equivariance, for the reconstruction of the index from the orbit space. A shift
  equivalence `(r, s)` restricts to submodules preserved by `A`, `B`, `r` and `s`
  (`isShiftEquivOfLag_restrict`), in particular to the kernels of endomorphisms that commute with
  `A`, `B` and `r`, `s` (`isShiftEquivOfLag_restrict_ker`), to the `+1` and `-1` eigenspaces of
  `σ` and `τ` (`isShiftEquivOfLag_restrict_eigenspaces`), and, for a group action, to the
  invariants and to the kernel of the averaging map (`isShiftEquivOfLag_restrict_invariants`,
  `isShiftEquivOfLag_restrict_ker_averageMap`). So the shift class of each of these components of
  an equivariant index map is an invariant of its equivariant shift class. It is not determined by
  the plain shift class of the total map, even when one involution acts on both modules and the two
  maps are conjugate (`exists_not_shiftEquivalent_restrict_eigenspace`,
  `exists_not_shiftEquivalent_restrict_eigenspace_of_isConjugate`).

## Not formalized

* Index pairs and index maps, and the independence of the shift class of `f_(N,L)` from the index
  pair. For (f), that the shift equivalences between two `G`-invariant index pairs can be chosen
  equivariant, as Section `sec:conley-symmetry` states. The results of (f) are the algebraic half
  of that statement.
* Graded index maps. The shift equivalences of `f_(N,L)` on `H_*(N, L)` preserve degrees, so they
  are families of shift equivalences in each degree with a common lag.
  `exists_isShiftEquivOfLag_of_forall` provides a common lag for finitely many degrees.
* The converse of (f), that shift equivalences of the components assemble into an equivariant
  shift equivalence.
* The Noetherian hypothesis on `V` in `shiftEquivalent_iff_isConjugate_lerayReduction` cannot be
  dropped, by a hand computation not checked in Lean: on
  `V = W = (ℤ[1/2]/ℤ)²`, which is Artinian and not Noetherian over `ℤ`, the matrices
  `A = diag(2, 4)` and `B = [[2, 1], [0, 4]]` act surjectively, so their eventual ranges are all
  of `V`; `r = [[1, 1], [0, 2]]` and `s = [[2, -1], [0, 2]]` satisfy `r A = B r`, `s B = A s`,
  `s r = A`, `r s = B`; and a conjugacy would be a matrix in `GL₂(ℤ₂)` mapping the eigenvectors
  `e₁`, `e₂` of `A` to the eigenvectors `(1, 0)`, `(1, 2)` of `B`, whose determinant is not a
  unit (`End(ℤ[1/2]/ℤ) = ℤ₂`; every solution of `e A = B e` is `[[a, b], [0, 2b]]`).
* For `G = ℤ/2` over a field with `char ≠ 2` and finite-dimensional homology, the shift class of
  each component is an invariant of `S` without equivariant shift equivalences, by a hand argument:
  the `+1` component is, through the transfer, the index map of `f/G` on `(N/G, L/G)`, an
  invariant of `S/G`; Leray reductions are additive; and modules of finite length over `K[t]`
  cancel (Krull-Schmidt). This needs the transfer for the index pair, which is not formalized.
-/

namespace LatentEquivariant

open Function Filter Set

universe u v w x

section Semiring

variable {k : Type u} [Semiring k]
  {V : Type v} [AddCommMonoid V] [Module k V]
  {W : Type w} [AddCommMonoid W] [Module k W]
  {U : Type x} [AddCommMonoid U] [Module k U]

/-- Shift equivalence with lag `m` (Williams 1973; Szymczak 1995, Franks-Richeson 2000): linear
maps `r : V → W` and `s : W → V` with `r ∘ A = B ∘ r`, `s ∘ B = A ∘ s`, `s ∘ r = A ^ m` and
`r ∘ s = B ^ m`. The Conley index of `S` is the shift equivalence class of `f_(N,L)`. -/
def IsShiftEquivOfLag (A : Module.End k V) (B : Module.End k W) (m : ℕ) : Prop :=
  ∃ (r : V →ₗ[k] W) (s : W →ₗ[k] V), r ∘ₗ A = B ∘ₗ r ∧ s ∘ₗ B = A ∘ₗ s ∧
    s ∘ₗ r = A ^ m ∧ r ∘ₗ s = B ^ m

/-- Shift equivalence: shift equivalence with some lag. The Conley index of an isolated invariant
set of a map is the class of its index map for this relation. -/
def ShiftEquivalent (A : Module.End k V) (B : Module.End k W) : Prop :=
  ∃ m, IsShiftEquivOfLag A B m

/-- Conjugacy of endomorphisms of possibly different modules: a linear isomorphism `e : V ≃ W`
with `e ∘ A = B ∘ e`. -/
def IsConjugate (A : Module.End k V) (B : Module.End k W) : Prop :=
  ∃ e : V ≃ₗ[k] W, e.toLinearMap ∘ₗ A = B ∘ₗ e.toLinearMap

variable {A : Module.End k V} {B : Module.End k W} {C : Module.End k U} {m n : ℕ}

/-- A linear map that intertwines `A` and `B` intertwines their powers. -/
theorem comp_pow_eq_pow_comp {r : V →ₗ[k] W} (h : r ∘ₗ A = B ∘ₗ r) (n : ℕ) :
    r ∘ₗ (A ^ n) = (B ^ n) ∘ₗ r :=
  (Module.End.commute_pow_left_of_commute h.symm n).symm

/-- `A ^ n` commutes with `A`, pointwise. -/
theorem pow_apply_apply_comm (A : Module.End k V) (n : ℕ) (v : V) :
    (A ^ n) (A v) = A ((A ^ n) v) := by
  rw [← Module.End.mul_apply, ← pow_succ, pow_succ', Module.End.mul_apply]

namespace IsShiftEquivOfLag

/-- Every endomorphism is shift equivalent to itself with any lag `m`, through `r = A ^ m` and
`s = id`. -/
theorem refl_pow (A : Module.End k V) (m : ℕ) : IsShiftEquivOfLag A A m := by
  refine ⟨A ^ m, LinearMap.id, ?_, ?_, ?_, ?_⟩
  · rw [← Module.End.mul_eq_comp, ← Module.End.mul_eq_comp, pow_mul_comm']
  · rw [LinearMap.id_comp, LinearMap.comp_id]
  · rw [LinearMap.id_comp]
  · rw [LinearMap.comp_id]

/-- Shift equivalence is reflexive, with lag `0`. -/
theorem refl (A : Module.End k V) : IsShiftEquivOfLag A A 0 :=
  refl_pow A 0

/-- Shift equivalence is symmetric, with the same lag. -/
theorem symm (h : IsShiftEquivOfLag A B m) : IsShiftEquivOfLag B A m := by
  obtain ⟨r, s, hr, hs, hsr, hrs⟩ := h
  exact ⟨s, r, hs, hr, hrs, hsr⟩

/-- Shift equivalence is transitive, and the lags add: `(r' ∘ r, s ∘ s')` is a shift equivalence
of lag `m + n`. -/
theorem trans (h : IsShiftEquivOfLag A B m) (h' : IsShiftEquivOfLag B C n) :
    IsShiftEquivOfLag A C (m + n) := by
  obtain ⟨r, s, hr, hs, hsr, hrs⟩ := h
  obtain ⟨r', s', hr', hs', hsr', hrs'⟩ := h'
  refine ⟨r' ∘ₗ r, s ∘ₗ s', ?_, ?_, ?_, ?_⟩
  · rw [LinearMap.comp_assoc, hr, ← LinearMap.comp_assoc, hr', LinearMap.comp_assoc]
  · rw [LinearMap.comp_assoc, hs', ← LinearMap.comp_assoc, hs, LinearMap.comp_assoc]
  · calc (s ∘ₗ s') ∘ₗ (r' ∘ₗ r) = s ∘ₗ ((s' ∘ₗ r') ∘ₗ r) := by
          simp only [LinearMap.comp_assoc]
      _ = s ∘ₗ (r ∘ₗ (A ^ n)) := by rw [hsr', comp_pow_eq_pow_comp hr]
      _ = A ^ (m + n) := by
          rw [← LinearMap.comp_assoc, hsr, pow_add, Module.End.mul_eq_comp]
  · calc (r' ∘ₗ r) ∘ₗ (s ∘ₗ s') = r' ∘ₗ ((r ∘ₗ s) ∘ₗ s') := by
          simp only [LinearMap.comp_assoc]
      _ = r' ∘ₗ (s' ∘ₗ (C ^ m)) := by rw [hrs, comp_pow_eq_pow_comp hs']
      _ = C ^ (m + n) := by
          rw [← LinearMap.comp_assoc, hrs', add_comm, pow_add, Module.End.mul_eq_comp]

/-- A lag can be raised: replace `r` by `r ∘ A ^ j`. -/
theorem mono (h : IsShiftEquivOfLag A B m) (hmn : m ≤ n) : IsShiftEquivOfLag A B n := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hmn
  exact h.trans (refl_pow B j)

end IsShiftEquivOfLag

namespace ShiftEquivalent

/-- Shift equivalence is reflexive. -/
theorem refl (A : Module.End k V) : ShiftEquivalent A A :=
  ⟨0, .refl A⟩

/-- Shift equivalence is symmetric. -/
theorem symm (h : ShiftEquivalent A B) : ShiftEquivalent B A :=
  h.elim fun m h ↦ ⟨m, h.symm⟩

/-- Shift equivalence is transitive. -/
theorem trans (h : ShiftEquivalent A B) (h' : ShiftEquivalent B C) : ShiftEquivalent A C :=
  h.elim fun m h ↦ h'.elim fun n h' ↦ ⟨m + n, h.trans h'⟩

end ShiftEquivalent

/-- Shift equivalence is an equivalence relation on the endomorphisms of a module. -/
theorem shiftEquivalent_equivalence :
    Equivalence (ShiftEquivalent (k := k) (V := V) (W := V)) :=
  ⟨ShiftEquivalent.refl, ShiftEquivalent.symm, ShiftEquivalent.trans⟩

/-- A family of shift equivalences indexed by a finite type, for instance the degrees of a
graded index map `f_(N,L)` on `H_*(N, L)` with finitely many nonzero degrees, has a common lag. -/
theorem exists_isShiftEquivOfLag_of_forall {ι : Type*} [Finite ι] {M N : ι → Type*}
    [∀ i, AddCommMonoid (M i)] [∀ i, Module k (M i)] [∀ i, AddCommMonoid (N i)]
    [∀ i, Module k (N i)] {A : ∀ i, Module.End k (M i)} {B : ∀ i, Module.End k (N i)}
    (h : ∀ i, ShiftEquivalent (A i) (B i)) : ∃ m, ∀ i, IsShiftEquivOfLag (A i) (B i) m := by
  choose l hl using h
  obtain ⟨M, hM⟩ := Finite.exists_le l
  exact ⟨M, fun i ↦ (hl i).mono (hM i)⟩

namespace IsConjugate

/-- Conjugate endomorphisms are shift equivalent with lag `0`, through `r = e` and
`s = e⁻¹`. -/
theorem isShiftEquivOfLag_zero (h : IsConjugate A B) : IsShiftEquivOfLag A B 0 := by
  obtain ⟨e, he⟩ := h
  refine ⟨e.toLinearMap, e.symm.toLinearMap, he, ?_, ?_, ?_⟩
  · ext w
    have := LinearMap.congr_fun he (e.symm w)
    simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply] at this
    simp only [LinearMap.comp_apply, LinearEquiv.coe_coe]
    rw [LinearEquiv.symm_apply_eq, this]
  · ext v
    simp
  · ext w
    simp

/-- Conjugate endomorphisms are shift equivalent. -/
theorem shiftEquivalent (h : IsConjugate A B) : ShiftEquivalent A B :=
  ⟨0, h.isShiftEquivOfLag_zero⟩

/-- An endomorphism conjugate to a bijective one is bijective. -/
theorem bijective (h : IsConjugate A B) (hA : Bijective A) : Bijective B := by
  obtain ⟨e, he⟩ := h
  have hB : ⇑B = ⇑e ∘ ⇑A ∘ ⇑e.symm := by
    funext w
    have := LinearMap.congr_fun he (e.symm w)
    simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply] at this
    simp [this]
  rw [hB]
  exact e.bijective.comp (hA.comp e.symm.bijective)

/-- Conjugacy is reflexive. -/
theorem refl (A : Module.End k V) : IsConjugate A A :=
  ⟨LinearEquiv.refl k V, LinearMap.ext fun _ ↦ rfl⟩

end IsConjugate

/-- Shift equivalence with lag `0` is conjugacy. -/
theorem isShiftEquivOfLag_zero_iff : IsShiftEquivOfLag A B 0 ↔ IsConjugate A B := by
  refine ⟨fun ⟨r, s, hr, _, hsr, hrs⟩ ↦ ?_, IsConjugate.isShiftEquivOfLag_zero⟩
  rw [pow_zero, Module.End.one_eq_id] at hsr hrs
  exact ⟨LinearEquiv.ofLinear r s hrs hsr, hr⟩

/-- Conjugacy is symmetric. -/
theorem IsConjugate.symm (h : IsConjugate A B) : IsConjugate B A :=
  isShiftEquivOfLag_zero_iff.mp h.isShiftEquivOfLag_zero.symm

/-- Conjugacy is transitive. -/
theorem IsConjugate.trans (h : IsConjugate A B) (h' : IsConjugate B C) : IsConjugate A C :=
  isShiftEquivOfLag_zero_iff.mp (h.isShiftEquivOfLag_zero.trans h'.isShiftEquivOfLag_zero)

/-! ### Nilpotency and the trivial index -/

/-- If `A` and `B` are shift equivalent with lag `m`, then `B ^ (n + m) = r ∘ A ^ n ∘ s`, so `B`
is nilpotent if `A` is. -/
theorem IsShiftEquivOfLag.isNilpotent (h : IsShiftEquivOfLag A B m) (hA : IsNilpotent A) :
    IsNilpotent B := by
  obtain ⟨r, s, hr, -, -, hrs⟩ := h
  obtain ⟨n, hn⟩ := hA
  refine ⟨n + m, ?_⟩
  rw [pow_add, Module.End.mul_eq_comp, ← hrs, ← LinearMap.comp_assoc,
    ← comp_pow_eq_pow_comp hr, hn, LinearMap.comp_zero, LinearMap.zero_comp]

/-- Shift equivalent endomorphisms are nilpotent together. -/
theorem IsShiftEquivOfLag.isNilpotent_iff (h : IsShiftEquivOfLag A B m) :
    IsNilpotent A ↔ IsNilpotent B :=
  ⟨h.isNilpotent, h.symm.isNilpotent⟩

/-- Nilpotency is an invariant of the shift equivalence class, over any semiring and in any
dimension. -/
theorem ShiftEquivalent.isNilpotent_iff (h : ShiftEquivalent A B) :
    IsNilpotent A ↔ IsNilpotent B :=
  h.elim fun _ h ↦ h.isNilpotent_iff

/-- The trivial index: an endomorphism is shift equivalent to the endomorphism of a zero module
if and only if it is nilpotent. This holds over any semiring and in any dimension, not only for
finite-dimensional `H_*(N, L)` over a field. -/
theorem shiftEquivalent_iff_isNilpotent [Subsingleton W] :
    ShiftEquivalent A B ↔ IsNilpotent A := by
  constructor
  · rintro ⟨m, r, s, -, -, hsr, -⟩
    refine ⟨m, ?_⟩
    rw [← hsr]
    ext v
    simp [Subsingleton.elim (r v) 0]
  · rintro ⟨n, hn⟩
    refine ⟨n, 0, 0, ?_, ?_, ?_, ?_⟩
    · rw [LinearMap.zero_comp, LinearMap.comp_zero]
    · rw [LinearMap.zero_comp, LinearMap.comp_zero]
    · rw [LinearMap.zero_comp, hn]
    · ext w
      exact Subsingleton.elim _ _

/-! ### Automorphisms -/

/-- Shift equivalent automorphisms are conjugate, over any semiring: `s ∘ r = A ^ m` is injective
and `r ∘ s = B ^ m` is surjective, so `r` is bijective and conjugates `A` to `B`. Only the
injectivity of `A` and the surjectivity of `B` are used. See Franks-Richeson 2000. -/
theorem IsShiftEquivOfLag.isConjugate (h : IsShiftEquivOfLag A B m) (hA : Injective A)
    (hB : Surjective B) : IsConjugate A B := by
  obtain ⟨r, s, hr, -, hsr, hrs⟩ := h
  have hinj : Injective (s ∘ r) := by
    rw [← LinearMap.coe_comp, hsr]
    exact Module.End.iterate_injective hA m
  have hsurj : Surjective (r ∘ s) := by
    rw [← LinearMap.coe_comp, hrs]
    exact Module.End.iterate_surjective hB m
  refine ⟨LinearEquiv.ofBijective r ⟨hinj.of_comp, hsurj.of_comp⟩, ?_⟩
  ext v
  simpa using LinearMap.congr_fun hr v

/-- Shift equivalent endomorphisms, the first injective and the second surjective, are
conjugate. -/
theorem ShiftEquivalent.isConjugate (h : ShiftEquivalent A B) (hA : Injective A)
    (hB : Surjective B) : IsConjugate A B :=
  h.elim fun _ h ↦ h.isConjugate hA hB

/-- For `A` injective and `B` surjective, shift equivalence is conjugacy. -/
theorem shiftEquivalent_iff_isConjugate (hA : Injective A) (hB : Surjective B) :
    ShiftEquivalent A B ↔ IsConjugate A B :=
  ⟨fun h ↦ h.isConjugate hA hB, IsConjugate.shiftEquivalent⟩

/-- Two automorphisms are shift equivalent if and only if they are conjugate, over any semiring
(Franks-Richeson 2000). -/
theorem shiftEquivalent_iff_isConjugate_of_isUnit (hA : IsUnit A) (hB : IsUnit B) :
    ShiftEquivalent A B ↔ IsConjugate A B :=
  shiftEquivalent_iff_isConjugate ((Module.End.isUnit_iff A).mp hA).1
    ((Module.End.isUnit_iff B).mp hB).2

/-! ### Identities -/

/-- An endomorphism is conjugate to an identity if and only if it is the identity of an
isomorphic module. -/
theorem isConjugate_one_iff :
    IsConjugate (1 : Module.End k V) B ↔ B = 1 ∧ Nonempty (V ≃ₗ[k] W) := by
  constructor
  · rintro ⟨e, he⟩
    refine ⟨LinearMap.ext fun w ↦ ?_, ⟨e⟩⟩
    have := LinearMap.congr_fun he (e.symm w)
    simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, Module.End.one_apply,
      LinearEquiv.apply_symm_apply] at this
    rw [Module.End.one_apply]
    exact this.symm
  · rintro ⟨rfl, ⟨e⟩⟩
    exact ⟨e, LinearMap.ext fun _ ↦ rfl⟩

/-- Two identities are shift equivalent if and only if the modules are isomorphic. This describes
the index of an index map that is the identity in every degree. -/
theorem shiftEquivalent_one_iff :
    ShiftEquivalent (1 : Module.End k V) (1 : Module.End k W) ↔ Nonempty (V ≃ₗ[k] W) := by
  rw [shiftEquivalent_iff_isConjugate_of_isUnit isUnit_one isUnit_one, isConjugate_one_iff]
  exact and_iff_right rfl

/-- For free modules over a semiring with the strong rank condition, for instance vector spaces,
two identities are shift equivalent if and only if the ranks agree: the index of an identity
index map is determined by the ranks. -/
theorem shiftEquivalent_one_iff_lift_rank_eq [StrongRankCondition k] [Module.Free k V]
    [Module.Free k W] :
    ShiftEquivalent (1 : Module.End k V) (1 : Module.End k W) ↔
      Cardinal.lift.{w} (Module.rank k V) = Cardinal.lift.{v} (Module.rank k W) :=
  shiftEquivalent_one_iff.trans LinearEquiv.nonempty_equiv_iff_lift_rank_eq

/-- The identities of `k ^ a` and `k ^ b` are shift equivalent if and only if `a = b`, over a
semiring with the strong rank condition. -/
theorem shiftEquivalent_one_fin_iff [StrongRankCondition k] {a b : ℕ} :
    ShiftEquivalent (1 : Module.End k (Fin a → k)) (1 : Module.End k (Fin b → k)) ↔ a = b := by
  rw [shiftEquivalent_one_iff]
  constructor
  · rintro ⟨e⟩
    simpa using e.finrank_eq
  · rintro rfl
    exact ⟨LinearEquiv.refl k _⟩

/-- The identity of `k` is shift equivalent with lag `1` to the idempotent `P = inl ∘ fst` of
`k × k`, `P (a, b) = (a, 0)`, through `r = inl` and `s = fst`. So the shift equivalence class of
an automorphism contains endomorphisms that are not invertible. -/
theorem isShiftEquivOfLag_one_inl_comp_fst :
    IsShiftEquivOfLag (1 : Module.End k k) (LinearMap.inl k k k ∘ₗ LinearMap.fst k k k) 1 := by
  refine ⟨LinearMap.inl k k k, LinearMap.fst k k k, ?_, ?_, ?_, (pow_one _).symm⟩
  · ext <;> simp
  · ext <;> simp
  · ext
    simp

/-- `P = inl ∘ fst` is not injective over a nontrivial semiring. -/
theorem not_injective_inl_comp_fst [Nontrivial k] :
    ¬ Injective (LinearMap.inl k k k ∘ₗ LinearMap.fst k k k) := by
  intro h
  have : ((0 : k), (1 : k)) = 0 := h (by simp)
  simp at this

/-- `P = inl ∘ fst` is not surjective over a nontrivial semiring. -/
theorem not_surjective_inl_comp_fst [Nontrivial k] :
    ¬ Surjective (LinearMap.inl k k k ∘ₗ LinearMap.fst k k k) := by
  intro h
  obtain ⟨x, hx⟩ := h ((0 : k), (1 : k))
  have := congr_arg Prod.snd hx
  simp at this

/-- `P = inl ∘ fst` is not invertible over a nontrivial semiring. -/
theorem not_isUnit_inl_comp_fst [Nontrivial k] :
    ¬ IsUnit (LinearMap.inl k k k ∘ₗ LinearMap.fst k k k) :=
  fun h ↦ not_injective_inl_comp_fst ((Module.End.isUnit_iff _).mp h).1

/-- The identity of `k` is shift equivalent to `P = inl ∘ fst` but not conjugate to it: the
shift equivalence class of an automorphism is larger than its conjugacy class, so the sentence
"over a field the shift equivalence class of an automorphism is its conjugacy class" is false as
worded. -/
theorem not_isConjugate_one_inl_comp_fst [Nontrivial k] :
    ¬ IsConjugate (1 : Module.End k k) (LinearMap.inl k k k ∘ₗ LinearMap.fst k k k) :=
  fun h ↦ not_injective_inl_comp_fst (h.bijective ((Module.End.isUnit_iff 1).mp isUnit_one)).1

/-! ### The Leray reduction -/

/-- `A` maps `range (A ^ n)` into itself. -/
theorem mapsTo_range_pow (A : Module.End k V) (n : ℕ) :
    MapsTo A (LinearMap.range (A ^ n)) (LinearMap.range (A ^ n)) := by
  rintro _ ⟨v, rfl⟩
  exact ⟨A v, pow_apply_apply_comm A n v⟩

/-- If `p = range (A ^ n)`, then `A` is shift equivalent with lag `n` to its restriction to `p`,
through `r = A ^ n` with values in `p` and the inclusion `s` of `p`. -/
theorem isShiftEquivOfLag_restrict_of_range_pow_eq {p : Submodule k V} (hp : MapsTo A p p)
    (hn : LinearMap.range (A ^ n) = p) : IsShiftEquivOfLag A (A.restrict hp) n := by
  have hmem : ∀ v, (A ^ n) v ∈ p := fun v ↦ hn.le (LinearMap.mem_range_self (A ^ n) v)
  refine ⟨LinearMap.codRestrict p (A ^ n) hmem, p.subtype, ?_, LinearMap.ext fun _ ↦ rfl,
    LinearMap.ext fun _ ↦ rfl, ?_⟩
  · ext v
    exact pow_apply_apply_comm A n v
  · ext x
    rw [Module.End.pow_restrict]
    rfl

/-- `A` is shift equivalent with lag `n` to its restriction to `range (A ^ n)`, over any semiring.
For `n` large on an Artinian module this restriction is the Leray reduction
(`eventually_isShiftEquivOfLag_lerayReduction`). -/
theorem isShiftEquivOfLag_restrict_range_pow (A : Module.End k V) (n : ℕ) :
    IsShiftEquivOfLag A (A.restrict (mapsTo_range_pow A n)) n :=
  isShiftEquivOfLag_restrict_of_range_pow_eq _ rfl

/-- The eventual range `⋂ n, range (A ^ n)` of `A`, on which the Leray reduction lives
(Mrozek 1990). -/
def eventualRange (A : Module.End k V) : Submodule k V :=
  ⨅ n, LinearMap.range (A ^ n)

/-- The eventual range is contained in every `range (A ^ n)`. -/
theorem eventualRange_le_range_pow (A : Module.End k V) (n : ℕ) :
    eventualRange A ≤ LinearMap.range (A ^ n) :=
  iInf_le _ n

/-- `A` maps its eventual range into itself. -/
theorem mapsTo_eventualRange (A : Module.End k V) :
    MapsTo A (eventualRange A) (eventualRange A) := by
  intro x hx
  simp only [SetLike.mem_coe, eventualRange, Submodule.mem_iInf] at hx ⊢
  exact fun n ↦ mapsTo_range_pow A n (hx n)

/-- The Leray reduction of `A`, in the form used here: the restriction of `A` to its eventual
range `⋂ n, range (A ^ n)`. Mrozek's Leray functor is `L_I ∘ L_M` (Mrozek 1990, Section 4), of
which `lerayQuotient` is the first step `L_M`. On modules of finite length `L_M` is already
bijective, and the three forms are conjugate (`isConjugate_lerayReduction_lerayQuotient`). In
infinite dimension they differ. -/
def lerayReduction (A : Module.End k V) : Module.End k (eventualRange A) :=
  A.restrict (mapsTo_eventualRange A)

/-- On an Artinian module the ranges of the powers of `A` stabilize at the eventual range. -/
theorem eventually_range_pow_eq_eventualRange [IsArtinian k V] (A : Module.End k V) :
    ∀ᶠ n in atTop, LinearMap.range (A ^ n) = eventualRange A :=
  (LinearMap.eventually_iInf_range_pow_eq A).mono fun _ h ↦ h.symm

/-- On an Artinian module, `A` is shift equivalent with lag `n` to its Leray reduction for every
large `n`. -/
theorem eventually_isShiftEquivOfLag_lerayReduction [IsArtinian k V] (A : Module.End k V) :
    ∀ᶠ n in atTop, IsShiftEquivOfLag A (lerayReduction A) n :=
  (eventually_range_pow_eq_eventualRange A).mono fun _ h ↦
    isShiftEquivOfLag_restrict_of_range_pow_eq _ h

/-- On an Artinian module, `A` is shift equivalent to its Leray reduction. -/
theorem shiftEquivalent_lerayReduction [IsArtinian k V] (A : Module.End k V) :
    ShiftEquivalent A (lerayReduction A) :=
  (eventually_isShiftEquivOfLag_lerayReduction A).exists

/-- If `range (A ^ n) = range (A ^ (n + 1)) = p`, then `A` maps `p` onto itself. -/
theorem surjective_restrict_of_range_pow_eq {p : Submodule k V} (hp : MapsTo A p p)
    (hn : LinearMap.range (A ^ n) = p) (hn' : LinearMap.range (A ^ (n + 1)) = p) :
    Surjective (A.restrict hp) := by
  rintro ⟨y, hy⟩
  obtain ⟨x, rfl⟩ := LinearMap.mem_range.mp (hn'.ge hy)
  refine ⟨⟨(A ^ n) x, hn.le (LinearMap.mem_range_self _ x)⟩, ?_⟩
  ext
  change A ((A ^ n) x) = (A ^ (n + 1)) x
  rw [pow_succ', Module.End.mul_apply]

/-- On an Artinian module the Leray reduction is surjective. -/
theorem surjective_lerayReduction [IsArtinian k V] (A : Module.End k V) :
    Surjective (lerayReduction A) := by
  obtain ⟨n, hn⟩ := ((eventually_range_pow_eq_eventualRange A).and
    ((tendsto_add_atTop_nat 1).eventually (eventually_range_pow_eq_eventualRange A))).exists
  exact surjective_restrict_of_range_pow_eq _ hn.1 hn.2

/-- The correct form of the sentence "over a field the shift equivalence class of an automorphism
is its conjugacy class": if `A` is injective, for instance an automorphism, and `W` is Artinian,
then `B` is shift equivalent to `A` if and only if the Leray reduction of `B` is conjugate to `A`.
Over any semiring, with no hypothesis on `V`. -/
theorem shiftEquivalent_iff_isConjugate_lerayReduction_of_injective [IsArtinian k W]
    (hA : Injective A) : ShiftEquivalent A B ↔ IsConjugate A (lerayReduction B) := by
  constructor
  · intro h
    exact (h.trans (shiftEquivalent_lerayReduction B)).isConjugate hA
      (surjective_lerayReduction B)
  · intro h
    exact h.shiftEquivalent.trans (shiftEquivalent_lerayReduction B).symm

/-- The injectivity hypothesis of `shiftEquivalent_iff_isConjugate_lerayReduction_of_injective`
covers no new cases: an injective endomorphism shift equivalent to an endomorphism of an Artinian
module is surjective. -/
theorem ShiftEquivalent.surjective_of_injective [IsArtinian k W] (hA : Injective A)
    (h : ShiftEquivalent A B) : Surjective A := by
  obtain ⟨e, he⟩ := (shiftEquivalent_iff_isConjugate_lerayReduction_of_injective hA).mp h
  intro v
  obtain ⟨x, hx⟩ := surjective_lerayReduction B (e v)
  refine ⟨e.symm x, e.injective ?_⟩
  have := LinearMap.congr_fun he (e.symm x)
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply] at this
  rw [this, hx]

/-- For an automorphism `A`, the forward direction of
`shiftEquivalent_iff_isConjugate_lerayReduction_of_injective` needs no hypothesis on `W`, over any
semiring: `r` maps `V` onto the eventual range of `B`, because `r (A ^ n u) = B ^ n (r u)` and
`B ^ m = r ∘ s`. -/
theorem isConjugate_lerayReduction_of_bijective (h : ShiftEquivalent A B) (hA : Bijective A) :
    IsConjugate A (lerayReduction B) := by
  obtain ⟨m, r, s, hr, -, hsr, hrs⟩ := h
  have hmem : ∀ v, r v ∈ eventualRange B := by
    intro v
    simp only [eventualRange, Submodule.mem_iInf]
    intro n
    obtain ⟨u, rfl⟩ := (Module.End.iterate_surjective hA.2 n) v
    exact ⟨r u, (LinearMap.congr_fun (comp_pow_eq_pow_comp hr n) u).symm⟩
  let r' : V →ₗ[k] eventualRange B := LinearMap.codRestrict (eventualRange B) r hmem
  have hinj : Injective r := by
    have : Injective (s ∘ r) := by
      rw [← LinearMap.coe_comp, hsr]
      exact Module.End.iterate_injective hA.1 m
    exact this.of_comp
  have hsurj : Surjective r' := by
    rintro ⟨y, hy⟩
    have hy' : y ∈ LinearMap.range (B ^ m) := eventualRange_le_range_pow B m hy
    rw [← hrs] at hy'
    obtain ⟨w, rfl⟩ := hy'
    exact ⟨s w, rfl⟩
  refine ⟨LinearEquiv.ofBijective r' ⟨fun x y hxy ↦ hinj (congr_arg Subtype.val hxy), hsurj⟩, ?_⟩
  ext v
  simpa [r', lerayReduction, LinearMap.coe_restrict_apply] using LinearMap.congr_fun hr v

/-- On an Artinian module `W`, an endomorphism `B` is shift equivalent to the identity of `V` if
and only if `B` is the identity on its eventual range and the eventual range is isomorphic to `V`
(index maps that are the identity). -/
theorem shiftEquivalent_one_iff_lerayReduction [IsArtinian k W] :
    ShiftEquivalent (1 : Module.End k V) B ↔
      lerayReduction B = 1 ∧ Nonempty (V ≃ₗ[k] eventualRange B) := by
  rw [shiftEquivalent_iff_isConjugate_lerayReduction_of_injective
    ((Module.End.isUnit_iff 1).mp isUnit_one).1, isConjugate_one_iff]

/-! ### Restriction to invariant submodules -/

/-- A shift equivalence `(r, s)` of lag `m` between `A` and `B` restricts to a shift equivalence
of lag `m` between the restrictions of `A` and `B` to submodules `p` and `q` that `A`, `B`, `r`
and `s` preserve. This is the algebraic half of the statement that the shift class of a component
of an equivariant index map is an invariant: it is one once the shift equivalences between index
maps preserve the components. -/
theorem isShiftEquivOfLag_restrict {r : V →ₗ[k] W} {s : W →ₗ[k] V} (hr : r ∘ₗ A = B ∘ₗ r)
    (hs : s ∘ₗ B = A ∘ₗ s) (hsr : s ∘ₗ r = A ^ m) (hrs : r ∘ₗ s = B ^ m)
    {p : Submodule k V} {q : Submodule k W} (hAp : MapsTo A p p) (hBq : MapsTo B q q)
    (hrp : MapsTo r p q) (hsq : MapsTo s q p) :
    IsShiftEquivOfLag (A.restrict hAp) (B.restrict hBq) m := by
  refine ⟨r.restrict hrp, s.restrict hsq, ?_, ?_, ?_, ?_⟩
  · ext x
    exact LinearMap.congr_fun hr x
  · ext x
    exact LinearMap.congr_fun hs x
  · ext x
    rw [Module.End.pow_restrict]
    exact LinearMap.congr_fun hsr x
  · ext x
    rw [Module.End.pow_restrict]
    exact LinearMap.congr_fun hrs x

end Semiring

section Ring

variable {k : Type u} [Ring k]
  {V : Type v} [AddCommGroup V] [Module k V]
  {W : Type w} [AddCommGroup W] [Module k W]
  {A : Module.End k V} {B : Module.End k W} {m : ℕ}

/-- If `p` meets `ker A` only in `0`, the restriction of `A` to `p` is injective. -/
theorem injective_restrict_of_disjoint {p : Submodule k V} (hp : MapsTo A p p)
    (hd : Disjoint p (LinearMap.ker A)) : Injective (A.restrict hp) := by
  rw [← LinearMap.ker_eq_bot, Submodule.eq_bot_iff]
  rintro ⟨x, hx⟩ hAx
  rw [LinearMap.mem_ker] at hAx
  have h0 : A x = 0 := congr_arg Subtype.val hAx
  exact Subtype.ext (Submodule.disjoint_def.mp hd x hx (LinearMap.mem_ker.mpr h0))

/-- On a Noetherian module the eventual range meets `ker A` only in `0` (Fitting). -/
theorem disjoint_eventualRange_ker [IsNoetherian k V] (A : Module.End k V) :
    Disjoint (eventualRange A) (LinearMap.ker A) := by
  obtain ⟨n, hn⟩ := ((Module.End.eventually_disjoint_ker_pow_range_pow A).and
    (eventually_ge_atTop 1)).exists
  refine (hn.1.symm.mono_left (eventualRange_le_range_pow A n)).mono_right ?_
  intro x hx
  rw [LinearMap.mem_ker] at hx ⊢
  exact Module.End.pow_map_zero_of_le hn.2 (by rwa [pow_one])

/-- On a Noetherian module the Leray reduction is injective. -/
theorem injective_lerayReduction [IsNoetherian k V] (A : Module.End k V) :
    Injective (lerayReduction A) :=
  injective_restrict_of_disjoint _ (disjoint_eventualRange_ker A)

/-- On a module of finite length, for instance a finite-dimensional vector space, the Leray
reduction is an automorphism (Mrozek 1990, Section 4). -/
theorem bijective_lerayReduction [IsNoetherian k V] [IsArtinian k V] (A : Module.End k V) :
    Bijective (lerayReduction A) :=
  ⟨injective_lerayReduction A, surjective_lerayReduction A⟩

/-- On a module of finite length, for instance a finite-dimensional vector space, the restriction
of `A` to `range (A ^ n)` is bijective for every large `n` (Fitting). -/
theorem eventually_bijective_restrict_range_pow [IsNoetherian k V] [IsArtinian k V]
    (A : Module.End k V) :
    ∀ᶠ n in atTop, Bijective (A.restrict (mapsTo_range_pow A n)) := by
  filter_upwards [eventually_range_pow_eq_eventualRange A,
    (tendsto_add_atTop_nat 1).eventually (eventually_range_pow_eq_eventualRange A)] with n h h'
  refine ⟨injective_restrict_of_disjoint _ ?_,
    surjective_restrict_of_range_pow_eq _ rfl (h'.trans h.symm)⟩
  rw [h]
  exact disjoint_eventualRange_ker A

/-- The correct form of the sentence "over a field the shift equivalence class of an automorphism
is its conjugacy class": for `V` of finite length and `W` Artinian, for instance
finite-dimensional vector spaces, `A` and `B` are shift equivalent if and only if their Leray
reductions are conjugate. `W` need not be Noetherian. -/
theorem shiftEquivalent_iff_isConjugate_lerayReduction [IsNoetherian k V] [IsArtinian k V]
    [IsArtinian k W] :
    ShiftEquivalent A B ↔ IsConjugate (lerayReduction A) (lerayReduction B) := by
  constructor
  · intro h
    exact ((shiftEquivalent_lerayReduction A).symm.trans
      (h.trans (shiftEquivalent_lerayReduction B))).isConjugate (injective_lerayReduction A)
      (surjective_lerayReduction B)
  · intro h
    exact (shiftEquivalent_lerayReduction A).trans
      (h.shiftEquivalent.trans (shiftEquivalent_lerayReduction B).symm)

/-- For `V` of finite length and `W` Artinian, the eventual range is an invariant of the shift
class up to isomorphism. The module itself is not (`isShiftEquivOfLag_one_inl_comp_fst`), so the
ranks of `H_*(N, L)` describe the index only when the index map is an automorphism. -/
theorem ShiftEquivalent.nonempty_linearEquiv_eventualRange [IsNoetherian k V] [IsArtinian k V]
    [IsArtinian k W] (h : ShiftEquivalent A B) :
    Nonempty (eventualRange A ≃ₗ[k] eventualRange B) :=
  (shiftEquivalent_iff_isConjugate_lerayReduction.mp h).elim fun e _ ↦ ⟨e⟩

/-- The generalized kernel `⋃ n, ker (A ^ n)` of `A`, which Mrozek's Leray reduction divides out
(Mrozek 1990). -/
def generalizedKernel (A : Module.End k V) : Submodule k V :=
  ⨆ n, LinearMap.ker (A ^ n)

/-- `A` maps its generalized kernel into itself. -/
theorem generalizedKernel_le_comap (A : Module.End k V) :
    generalizedKernel A ≤ (generalizedKernel A).comap A :=
  iSup_le fun n x hx ↦ by
    rw [Submodule.mem_comap]
    refine Submodule.mem_iSup_of_mem n ?_
    rw [LinearMap.mem_ker] at hx ⊢
    rw [pow_apply_apply_comm, hx, map_zero]

/-- The first step `L_M` of Mrozek's Leray functor `L_I ∘ L_M` (Mrozek 1990, Section 4, (4.1)):
the map induced by `A` on `V / ⋃ n, ker (A ^ n)`, which is injective on a Noetherian module. On
modules of finite length it is bijective, so `L_I` changes nothing and it is Mrozek's Leray
reduction. In infinite dimension it is not: for the unilateral shift on `k^(ℕ)` the generalized
kernel is `0`, so `lerayQuotient` is the shift itself, while `L` is `0`. -/
def lerayQuotient (A : Module.End k V) : Module.End k (V ⧸ generalizedKernel A) :=
  (generalizedKernel A).mapQ (generalizedKernel A) A (generalizedKernel_le_comap A)

/-- On a module of finite length the two forms of the Leray reduction are conjugate: by Fitting,
`V` is the direct sum of the generalized kernel and the eventual range, and the projection of the
eventual range to the quotient conjugates them. -/
theorem isConjugate_lerayReduction_lerayQuotient [IsNoetherian k V] [IsArtinian k V]
    (A : Module.End k V) : IsConjugate (lerayReduction A) (lerayQuotient A) := by
  have hc : IsCompl (generalizedKernel A) (eventualRange A) :=
    LinearMap.isCompl_iSup_ker_pow_iInf_range_pow A
  refine ⟨(Submodule.quotientEquivOfIsCompl _ _ hc).symm, LinearMap.ext fun y ↦ ?_⟩
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe,
    Submodule.quotientEquivOfIsCompl_symm_apply, lerayQuotient, Submodule.mapQ_apply]
  rfl

/-- For modules of finite length, for instance finite-dimensional vector spaces, `A` and `B` are
shift equivalent if and only if their Leray reductions in Mrozek's form are conjugate. So, for an
endomorphism of a module of finite length, the shift class and Mrozek's Leray reduction determine
each other. Mrozek's index is the Leray reduction of
the index map in cohomology, for homeomorphisms; over a field with finite-dimensional homology it
corresponds to the homological index by duality, which is not formalized. -/
theorem shiftEquivalent_iff_isConjugate_lerayQuotient [IsNoetherian k V] [IsArtinian k V]
    [IsNoetherian k W] [IsArtinian k W] :
    ShiftEquivalent A B ↔ IsConjugate (lerayQuotient A) (lerayQuotient B) := by
  rw [shiftEquivalent_iff_isConjugate_lerayReduction]
  constructor
  · intro h
    exact (isConjugate_lerayReduction_lerayQuotient A).symm.trans
      (h.trans (isConjugate_lerayReduction_lerayQuotient B))
  · intro h
    exact (isConjugate_lerayReduction_lerayQuotient A).trans
      (h.trans (isConjugate_lerayReduction_lerayQuotient B).symm)

/-! ### Equivariant shift equivalences -/

/-- A shift equivalence `(r, s)` restricts to the kernels of endomorphisms `φ` of `V` and `ψ` of
`W` that commute with `A` and `B` and that `r` and `s` intertwine. -/
theorem isShiftEquivOfLag_restrict_ker {r : V →ₗ[k] W} {s : W →ₗ[k] V}
    (hr : r ∘ₗ A = B ∘ₗ r) (hs : s ∘ₗ B = A ∘ₗ s) (hsr : s ∘ₗ r = A ^ m)
    (hrs : r ∘ₗ s = B ^ m) {φ : Module.End k V} {ψ : Module.End k W} (hA : Commute A φ)
    (hB : Commute B ψ) (hrφ : r ∘ₗ φ = ψ ∘ₗ r) (hsψ : s ∘ₗ ψ = φ ∘ₗ s) :
    IsShiftEquivOfLag (A.restrict (mapsTo_ker_of_commute hA))
      (B.restrict (mapsTo_ker_of_commute hB)) m := by
  refine isShiftEquivOfLag_restrict hr hs hsr hrs _ _ ?_ ?_
  · intro x hx
    rw [SetLike.mem_coe, LinearMap.mem_ker] at hx ⊢
    rw [← LinearMap.comp_apply, ← hrφ, LinearMap.comp_apply, hx, map_zero]
  · intro x hx
    rw [SetLike.mem_coe, LinearMap.mem_ker] at hx ⊢
    rw [← LinearMap.comp_apply, ← hsψ, LinearMap.comp_apply, hx, map_zero]

/-- `ℤ/2`: if `σ` and `τ` commute with `A` and `B`, and the shift equivalence `(r, s)` is
equivariant, `r ∘ σ = τ ∘ r` and `s ∘ τ = σ ∘ s`, then it restricts to the `+1` eigenspaces
`ker (σ - 1)`, `ker (τ - 1)` and to the `-1` eigenspaces `ker (σ + 1)`, `ker (τ + 1)`, with the
same lag. So the index of each summand of Proposition `prop:z2_reconstruction` is an invariant
of the equivariant shift class of the index map.
Neither `σ * σ = 1` nor the invertibility of `2` is needed. -/
theorem isShiftEquivOfLag_restrict_eigenspaces {r : V →ₗ[k] W} {s : W →ₗ[k] V}
    (hr : r ∘ₗ A = B ∘ₗ r) (hs : s ∘ₗ B = A ∘ₗ s) (hsr : s ∘ₗ r = A ^ m)
    (hrs : r ∘ₗ s = B ^ m) {σ : Module.End k V} {τ : Module.End k W} (hA : Commute A σ)
    (hB : Commute B τ) (hrσ : r ∘ₗ σ = τ ∘ₗ r) (hsτ : s ∘ₗ τ = σ ∘ₗ s) :
    IsShiftEquivOfLag (A.restrict (mapsTo_ker_of_commute (hA.sub_right (Commute.one_right A))))
        (B.restrict (mapsTo_ker_of_commute (hB.sub_right (Commute.one_right B)))) m ∧
      IsShiftEquivOfLag
        (A.restrict (mapsTo_ker_of_commute (hA.add_right (Commute.one_right A))))
        (B.restrict (mapsTo_ker_of_commute (hB.add_right (Commute.one_right B)))) m := by
  have hr' := LinearMap.congr_fun hrσ
  have hs' := LinearMap.congr_fun hsτ
  simp only [LinearMap.comp_apply] at hr' hs'
  refine ⟨isShiftEquivOfLag_restrict_ker hr hs hsr hrs (hA.sub_right (Commute.one_right A))
      (hB.sub_right (Commute.one_right B)) ?_ ?_,
    isShiftEquivOfLag_restrict_ker hr hs hsr hrs (hA.add_right (Commute.one_right A))
      (hB.add_right (Commute.one_right B)) ?_ ?_⟩ <;>
  · ext x
    simp [hr', hs']

/-- The equivariance of `r` and `s` in `isShiftEquivOfLag_restrict_eigenspaces` is needed. Over a
nontrivial ring in which `2` is not a zero divisor, take `σ = 1` and `τ = -1` on `k`. The identity
of `k` is shift equivalent to itself, but the restrictions of the identity to the `+1`
eigenspaces, the identity of `k` and the endomorphism of `0`, are not shift equivalent. So the
eigenspace components are invariants of the equivariant shift class (Proposition
`prop:z2_reconstruction`), not of the plain one. -/
theorem exists_not_shiftEquivalent_restrict_eigenspace [Nontrivial k]
    (h2 : IsLeftRegular (2 : k)) :
    ∃ (σ τ : Module.End k k) (hσ : Commute (1 : Module.End k k) σ)
      (hτ : Commute (1 : Module.End k k) τ), σ * σ = 1 ∧ τ * τ = 1 ∧
      ShiftEquivalent (1 : Module.End k k) (1 : Module.End k k) ∧
      ¬ ShiftEquivalent
        ((1 : Module.End k k).restrict (mapsTo_ker_of_commute (hσ.sub_right (Commute.one_right 1))))
        ((1 : Module.End k k).restrict
          (mapsTo_ker_of_commute (hτ.sub_right (Commute.one_right 1)))) := by
  refine ⟨1, -1, Commute.one_left _, Commute.one_left _, mul_one 1,
    LinearMap.ext fun x ↦ by simp, .refl 1, ?_⟩
  have hsub : Subsingleton (LinearMap.ker ((-1 : Module.End k k) - 1)) := by
    have key : ∀ z : LinearMap.ker ((-1 : Module.End k k) - 1), z = 0 := by
      rintro ⟨z, hz⟩
      rw [LinearMap.mem_ker, LinearMap.sub_apply, LinearMap.neg_apply, Module.End.one_apply,
        ← neg_add', neg_eq_zero] at hz
      refine Subtype.ext (h2 ?_)
      change 2 * z = 2 * 0
      rw [two_mul, hz, mul_zero]
    exact ⟨fun x y ↦ by rw [key x, key y]⟩
  rw [shiftEquivalent_iff_isNilpotent]
  rintro ⟨n, hn⟩
  have h1 : (1 : k) ∈ LinearMap.ker ((1 : Module.End k k) - 1) := by simp
  have := congr_arg Subtype.val (LinearMap.congr_fun hn ⟨1, h1⟩)
  rw [Module.End.pow_restrict] at this
  simp at this

/-- A sharper form of `exists_not_shiftEquivalent_restrict_eigenspace`: one involution
`σ = diag(1, -1)` of `ℚ²` acting on both sides, and `A = diag(1, 0)`, `B = diag(0, 1)` commuting
with it and conjugate by the swap, which is not equivariant. The restrictions of `A` and `B` to the
`+1` eigenspace, the identity and `0`, are not shift equivalent. -/
theorem exists_not_shiftEquivalent_restrict_eigenspace_of_isConjugate :
    ∃ (σ A B : Module.End ℚ (ℚ × ℚ)) (hA : Commute A σ) (hB : Commute B σ),
      σ * σ = 1 ∧ IsConjugate A B ∧
      ¬ ShiftEquivalent (A.restrict (mapsTo_ker_of_commute (hA.sub_right (Commute.one_right A))))
        (B.restrict (mapsTo_ker_of_commute (hB.sub_right (Commute.one_right B)))) := by
  have hA : Commute (LinearMap.inl ℚ ℚ ℚ ∘ₗ LinearMap.fst ℚ ℚ ℚ : Module.End ℚ (ℚ × ℚ))
      (LinearMap.prodMap LinearMap.id (-LinearMap.id)) := by
    show _ * _ = _ * _
    ext <;> simp
  have hB : Commute (LinearMap.inr ℚ ℚ ℚ ∘ₗ LinearMap.snd ℚ ℚ ℚ : Module.End ℚ (ℚ × ℚ))
      (LinearMap.prodMap LinearMap.id (-LinearMap.id)) := by
    show _ * _ = _ * _
    ext <;> simp
  refine ⟨_, _, _, hA, hB, ?_, ⟨LinearEquiv.prodComm ℚ ℚ ℚ, ?_⟩, ?_⟩
  · ext <;> simp
  · ext <;> simp
  · intro h
    have hB0 : IsNilpotent ((LinearMap.inr ℚ ℚ ℚ ∘ₗ LinearMap.snd ℚ ℚ ℚ :
        Module.End ℚ (ℚ × ℚ)).restrict
          (mapsTo_ker_of_commute (hB.sub_right (Commute.one_right _)))) := by
      refine ⟨1, LinearMap.ext fun z ↦ Subtype.ext ?_⟩
      obtain ⟨⟨x, y⟩, hz⟩ := z
      have hy : y = 0 := by
        have := congr_arg Prod.snd (LinearMap.mem_ker.mp hz)
        simp at this
        linarith
      subst hy
      simp
    obtain ⟨n, hn⟩ := h.isNilpotent_iff.mpr hB0
    have hpow : ∀ j : ℕ, ((LinearMap.inl ℚ ℚ ℚ ∘ₗ LinearMap.fst ℚ ℚ ℚ :
        Module.End ℚ (ℚ × ℚ)) ^ j) ((1 : ℚ), (0 : ℚ)) = ((1 : ℚ), (0 : ℚ)) := by
      intro j
      induction j with
      | zero => simp
      | succ j ih => rw [pow_succ, Module.End.mul_apply]; simpa using ih
    have hmem : ((1 : ℚ), (0 : ℚ)) ∈ LinearMap.ker
        ((LinearMap.prodMap LinearMap.id (-LinearMap.id) : Module.End ℚ (ℚ × ℚ)) - 1) := by
      simp
    have h1 := LinearMap.congr_fun hn ⟨((1 : ℚ), (0 : ℚ)), hmem⟩
    rw [Module.End.pow_restrict, LinearMap.restrict_apply, LinearMap.zero_apply] at h1
    have h2 := congr_arg Subtype.val h1
    simp only [hpow] at h2
    simp at h2

end Ring

section Group

variable {k : Type u} [CommRing k] {G : Type*} [Group G]
  {V : Type v} [AddCommGroup V] [Module k V]
  {W : Type w} [AddCommGroup W] [Module k W]
  {ρ : Representation k G V} {ρ' : Representation k G W}
  {A : Module.End k V} {B : Module.End k W} {m : ℕ}

/-- An equivariant shift equivalence restricts to the invariants: if `A` and `B` commute with the
actions and `r` and `s` are equivariant, the restrictions of `A` and `B` to `V^G` and `W^G` are
shift equivalent with the same lag. The invariants are the trivial isotypic component. -/
theorem isShiftEquivOfLag_restrict_invariants {r : V →ₗ[k] W} {s : W →ₗ[k] V}
    (hr : r ∘ₗ A = B ∘ₗ r) (hs : s ∘ₗ B = A ∘ₗ s) (hsr : s ∘ₗ r = A ^ m)
    (hrs : r ∘ₗ s = B ^ m) (hA : ∀ g, Commute A (ρ g)) (hB : ∀ g, Commute B (ρ' g))
    (hrρ : ∀ g, r ∘ₗ ρ g = ρ' g ∘ₗ r) (hsρ : ∀ g, s ∘ₗ ρ' g = ρ g ∘ₗ s) :
    IsShiftEquivOfLag (A.restrict (mapsTo_invariants hA)) (B.restrict (mapsTo_invariants hB))
      m := by
  refine isShiftEquivOfLag_restrict hr hs hsr hrs _ _ ?_ ?_
  · intro v hv
    rw [SetLike.mem_coe, Representation.mem_invariants] at hv ⊢
    intro g
    rw [← LinearMap.comp_apply, ← hrρ g, LinearMap.comp_apply, hv g]
  · intro w hw
    rw [SetLike.mem_coe, Representation.mem_invariants] at hw ⊢
    intro g
    rw [← LinearMap.comp_apply, ← hsρ g, LinearMap.comp_apply, hw g]

/-- An equivariant shift equivalence restricts to the kernels of the averaging maps, which over a
field with `char ∤ |G|` are the sums of the nontrivial isotypic components. -/
theorem isShiftEquivOfLag_restrict_ker_averageMap [Fintype G] [Invertible (Fintype.card G : k)]
    {r : V →ₗ[k] W} {s : W →ₗ[k] V}
    (hr : r ∘ₗ A = B ∘ₗ r) (hs : s ∘ₗ B = A ∘ₗ s) (hsr : s ∘ₗ r = A ^ m)
    (hrs : r ∘ₗ s = B ^ m) (hA : ∀ g, Commute A (ρ g)) (hB : ∀ g, Commute B (ρ' g))
    (hrρ : ∀ g, r ∘ₗ ρ g = ρ' g ∘ₗ r) (hsρ : ∀ g, s ∘ₗ ρ' g = ρ g ∘ₗ s) :
    IsShiftEquivOfLag (A.restrict (mapsTo_ker_averageMap hA))
      (B.restrict (mapsTo_ker_averageMap hB)) m := by
  have hr' : ∀ g v, r (ρ g v) = ρ' g (r v) := fun g ↦ LinearMap.congr_fun (hrρ g)
  have hs' : ∀ g w, s (ρ' g w) = ρ g (s w) := fun g ↦ LinearMap.congr_fun (hsρ g)
  refine isShiftEquivOfLag_restrict_ker hr hs hsr hrs (commute_averageMap hA)
    (commute_averageMap hB) ?_ ?_
  · rw [averageMap_eq_sum, averageMap_eq_sum]
    ext v
    simp [map_sum, hr']
  · rw [averageMap_eq_sum, averageMap_eq_sum]
    ext w
    simp [map_sum, hs']

end Group

section DivisionRing

variable {K : Type u} [DivisionRing K]
  {V : Type v} [AddCommGroup V] [Module K V] [FiniteDimensional K V]
  {W : Type w} [AddCommGroup W] [Module K W] [FiniteDimensional K W]

/-- Over a field, or a division ring, for finite-dimensional `V` and `W`, two endomorphisms are
shift equivalent if and only if their Leray reductions, the restrictions to the eventual ranges,
are conjugate. This is the correct form of the sentence "over a field the shift equivalence class
of an automorphism is its conjugacy class". -/
theorem shiftEquivalent_iff_isConjugate_lerayReduction_of_finiteDimensional
    {A : Module.End K V} {B : Module.End K W} :
    ShiftEquivalent A B ↔ IsConjugate (lerayReduction A) (lerayReduction B) :=
  shiftEquivalent_iff_isConjugate_lerayReduction

/-- Over a field, or a division ring, for finite-dimensional `V` and `W`, an endomorphism `B` is
shift equivalent to the identity of `V` if and only if `B` is the identity on its eventual range
and the eventual range has dimension `dim V` (index maps that are the identity). -/
theorem shiftEquivalent_one_iff_finrank {B : Module.End K W} :
    ShiftEquivalent (1 : Module.End K V) B ↔
      lerayReduction B = 1 ∧ Module.finrank K V = Module.finrank K (eventualRange B) := by
  rw [shiftEquivalent_one_iff_lerayReduction, FiniteDimensional.nonempty_linearEquiv_iff_finrank_eq]

/-- Over a field, or a division ring, the dimension of the eventual range is an invariant of the
shift class of an endomorphism of a finite-dimensional space, while the dimension of the space is
not (`isShiftEquivOfLag_one_inl_comp_fst`). These are the ranks that describe the index. -/
theorem ShiftEquivalent.finrank_eventualRange_eq {A : Module.End K V} {B : Module.End K W}
    (h : ShiftEquivalent A B) :
    Module.finrank K (eventualRange A) = Module.finrank K (eventualRange B) :=
  (h.nonempty_linearEquiv_eventualRange).elim LinearEquiv.finrank_eq

end DivisionRing

end LatentEquivariant
