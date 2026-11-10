import Mathlib

/-!
# The isotypic splitting and the index map

Linear algebra behind the reconstruction of the index from the orbit space. Over a
field, with `H_*(N, L)` finite dimensional, the index is trivial exactly when the index map is
nilpotent. An equivariant index map commutes with the group,
so it preserves the invariants and the kernel of the averaging map, and it is nilpotent exactly
when it is nilpotent on both. Over a field with `char ∤ |G|` that kernel is the sum of the
nontrivial isotypic components. The decomposition into all isotypic components, for a general
finite group, is not formalized here. Everything holds over any commutative ring in which `|G|`
(respectively `2`) is invertible, and for modules of any dimension.

* `isNilpotent_iff_restrict`: an endomorphism preserving two complementary submodules is nilpotent
  if and only if both restrictions are.
* Finite group, `|G|` invertible: `isCompl_invariants_ker_averageMap`, `mapsTo_invariants`,
  `mapsTo_ker_averageMap`, `isNilpotent_iff_invariants_and_ker`.
* An involution, `2` invertible (`G = ℤ/2`): the eigenspace part of Proposition
  `prop:z2_reconstruction`, `isCompl_ker_sub_one_ker_add_one`, `mapsTo_ker_of_commute`,
  `isNilpotent_iff_eigenspaces`. The identification of the eigenspaces with orbit-space
  homology is not in this file.
-/

namespace LatentEquivariant

open Module LinearMap Set

section Nilpotent

variable {R M : Type*} [Ring R] [AddCommGroup M] [Module R M]

/-- An endomorphism that preserves two complementary submodules is nilpotent if and only if its
restrictions to both are nilpotent. -/
theorem isNilpotent_iff_restrict {p q : Submodule R M} (hpq : IsCompl p q) {T : Module.End R M}
    (hp : MapsTo T p p) (hq : MapsTo T q q) :
    IsNilpotent T ↔ IsNilpotent (T.restrict hp) ∧ IsNilpotent (T.restrict hq) := by
  refine ⟨fun h ↦ ⟨Module.End.isNilpotent.restrict hp h, Module.End.isNilpotent.restrict hq h⟩,
    ?_⟩
  rintro ⟨⟨a, ha⟩, ⟨b, hb⟩⟩
  have hp0 : ∀ y ∈ p, (T ^ a) y = 0 := fun y hy ↦ by
    have h := LinearMap.congr_fun ha ⟨y, hy⟩
    rw [Module.End.pow_restrict, LinearMap.restrict_apply, LinearMap.zero_apply] at h
    simpa using congr_arg Subtype.val h
  have hq0 : ∀ z ∈ q, (T ^ b) z = 0 := fun z hz ↦ by
    have h := LinearMap.congr_fun hb ⟨z, hz⟩
    rw [Module.End.pow_restrict, LinearMap.restrict_apply, LinearMap.zero_apply] at h
    simpa using congr_arg Subtype.val h
  refine ⟨a + b, LinearMap.ext fun x ↦ ?_⟩
  have hx : x ∈ p ⊔ q := by
    rw [hpq.sup_eq_top]
    exact Submodule.mem_top
  obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.mp hx
  have e1 : (T ^ (a + b)) y = 0 := by
    rw [add_comm, pow_add, Module.End.mul_apply, hp0 y hy, map_zero]
  have e2 : (T ^ (a + b)) z = 0 := by
    rw [pow_add, Module.End.mul_apply, hq0 z hz, map_zero]
  rw [map_add, e1, e2, add_zero, LinearMap.zero_apply]

/-- An endomorphism commuting with `S` preserves `ker S`. -/
theorem mapsTo_ker_of_commute {T S : Module.End R M} (h : Commute T S) :
    MapsTo T (ker S) (ker S) := by
  intro v hv
  rw [SetLike.mem_coe, mem_ker] at hv ⊢
  rw [← Module.End.mul_apply, ← h.eq, Module.End.mul_apply, hv, map_zero]

end Nilpotent

section Average

variable {k G V : Type*} [CommRing k] [Group G] [AddCommGroup V] [Module k V]
  (ρ : Representation k G V) [Fintype G] [Invertible (Fintype.card G : k)]

theorem averageMap_eq_sum : ρ.averageMap = ⅟(Fintype.card G : k) • ∑ g : G, ρ g := by
  simp [Representation.averageMap, GroupAlgebra.average, map_sum]

/-- The averaging projection splits `V` into the invariants and its kernel. Over a field with
`char ∤ |G|` these are the trivial isotypic component and the sum of the other components. -/
theorem isCompl_invariants_ker_averageMap :
    IsCompl ρ.invariants (LinearMap.ker ρ.averageMap) :=
  ρ.isProj_averageMap.isCompl

variable {ρ}

theorem commute_averageMap {T : Module.End k V} (hT : ∀ g, Commute T (ρ g)) :
    Commute T ρ.averageMap := by
  rw [averageMap_eq_sum]
  exact (Commute.sum_right _ _ _ fun g _ ↦ hT g).smul_right _

omit [Fintype G] [Invertible (Fintype.card G : k)] in
theorem mapsTo_invariants {T : Module.End k V} (hT : ∀ g, Commute T (ρ g)) :
    MapsTo T ρ.invariants ρ.invariants := by
  intro v hv g
  have h := LinearMap.congr_fun (hT g).eq v
  rw [Module.End.mul_apply, Module.End.mul_apply, (Representation.mem_invariants ρ v).mp hv g]
    at h
  exact h.symm

theorem mapsTo_ker_averageMap {T : Module.End k V} (hT : ∀ g, Commute T (ρ g)) :
    MapsTo T (LinearMap.ker ρ.averageMap) (LinearMap.ker ρ.averageMap) :=
  mapsTo_ker_of_commute (commute_averageMap hT)

/-- An equivariant index map is nilpotent if and only if it is nilpotent on the trivial component
and on the sum of the other components. -/
theorem isNilpotent_iff_invariants_and_ker {T : Module.End k V} (hT : ∀ g, Commute T (ρ g)) :
    IsNilpotent T ↔ IsNilpotent (T.restrict (mapsTo_invariants hT)) ∧
      IsNilpotent (T.restrict (mapsTo_ker_averageMap hT)) :=
  isNilpotent_iff_restrict (isCompl_invariants_ker_averageMap ρ) _ _

end Average

section Involution

variable {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M] [Invertible (2 : R)]
  {σ : Module.End R M}

/-- The `+1` and `-1` eigenspaces of an involution are complementary when `2` is invertible. -/
theorem isCompl_ker_sub_one_ker_add_one (hσ : σ * σ = 1) :
    IsCompl (ker (σ - 1)) (ker (σ + 1)) := by
  have hσx : ∀ x, σ (σ x) = x := fun x ↦ by
    rw [← Module.End.mul_apply, hσ, Module.End.one_apply]
  let P : Module.End R M := ⅟(2 : R) • (1 + σ)
  have hPx : ∀ x, P x = ⅟(2 : R) • (x + σ x) := fun x ↦ rfl
  have hP : LinearMap.IsProj (ker (σ - 1)) P := by
    constructor
    · intro x
      rw [mem_ker, LinearMap.sub_apply, Module.End.one_apply, hPx, map_smul, map_add, hσx,
        add_comm (σ x), sub_self]
    · intro x hx
      rw [mem_ker, LinearMap.sub_apply, Module.End.one_apply, sub_eq_zero] at hx
      rw [hPx, hx, ← two_smul R x, smul_smul, invOf_mul_self, one_smul]
  have hker : ker P = ker (σ + 1) := by
    ext x
    rw [mem_ker, mem_ker, hPx, LinearMap.add_apply, Module.End.one_apply, add_comm (σ x)]
    constructor
    · intro h
      have := congr_arg (fun y ↦ (2 : R) • y) h
      simpa [smul_smul] using this
    · intro h
      rw [h, smul_zero]
  rw [← hker]
  exact hP.isCompl

/-- `ℤ/2`: an index map commuting with the involution is nilpotent if and only if it is nilpotent
on both eigenspaces. -/
theorem isNilpotent_iff_eigenspaces (hσ : σ * σ = 1) {T : Module.End R M} (hT : Commute T σ) :
    IsNilpotent T ↔
      IsNilpotent (T.restrict (mapsTo_ker_of_commute (hT.sub_right (Commute.one_right T)))) ∧
        IsNilpotent (T.restrict (mapsTo_ker_of_commute (hT.add_right (Commute.one_right T)))) :=
  isNilpotent_iff_restrict (isCompl_ker_sub_one_ker_add_one hσ) _ _

end Involution

end LatentEquivariant
