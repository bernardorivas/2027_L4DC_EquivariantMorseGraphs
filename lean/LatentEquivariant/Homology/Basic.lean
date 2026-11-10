import Mathlib
import LatentEquivariant.Algebra.Chains
import LatentEquivariant.Algebra.Isotypic

/-!
# Homology of finite cell complexes

The homology layer under the reconstruction of the index from the orbit space.
CMGDB computes `H_*(N, L)` of an index pair from the cells of a grid, with chains as functions on
cells and the boundary `bd inc` of `LatentEquivariant.Algebra.Chains`. This file sets up that
homology on top of Mathlib's homological algebra (`HomologicalComplex`, `ChainComplex.of`,
`ShortComplex`, `ModuleCat`), so that later work can use Mathlib's API.

## Main definitions and statements

* `CellComplex k`: finitely many cells in each dimension, incidence numbers `inc n e e'`, and
  `∂ ∘ ∂ = 0`. `CellComplex.toChainComplex` is its chain complex in `ModuleCat k`, built with
  `ChainComplex.of`, whose homology in degree `n` is Mathlib's `K.toChainComplex.homology n`.
* `CellComplex.H K n = ker ∂_n ⧸ range ∂_(n + 1)` (with `∂_0 = 0`) and
  `CellComplex.homologyEquiv`, a linear equivalence with Mathlib's homology. It comes from the
  general `moduleCatHomologyEquiv` for short complexes of modules, which is natural
  (`moduleCatHomologyEquiv_naturality`).
* Functoriality: `CellComplex.ChainMap` (linear maps commuting with the boundaries) induces
  `ChainMap.homologyMap` on Mathlib's homology and `ChainMap.mapH` on `ker ∂ ⧸ range ∂`, both
  compatible with identities and composition and intertwined by `homologyEquiv`
  (`ChainMap.homologyEquiv_naturality`, `ChainMap.isNilpotent_mapH_iff`).
  `CellComplex.ChainEquiv` gives isomorphic homology, in both forms.
* `CellComplex.restrict` and `CellComplex.relative`: the complex on the cells of `N` not in `L`,
  for subcomplexes `N` and `L`, with the restricted incidence numbers. This is the convention of
  `OrbitChains` and `SignChains` for relative chains `C_*(N, L)`.
* `subcomplex`, `kerSubcomplex`, `plusSubcomplex`, `minusSubcomplex`: subcomplexes of a complex
  of modules, and the `+1` and `-1` eigen-subcomplexes `C⁺ = ker (σ - 1)` and `C⁻ = ker (σ + 1)`
  of a chain endomorphism `σ`. `isoOfLinearEquivs` builds isomorphisms of complexes from
  degreewise linear equivalences.
* Homology preserves direct sums (`homologyIsoBiprod`, `homologyEquivProd`), because the
  homology maps are additive (`HomologicalComplex.homologyMap_add`).
* The splitting step of Proposition `prop:z2_reconstruction`, at the homology level: for an
  involution `σ` of a complex `C` of `k`-modules with `2` invertible,
  `isoBiprodEigen : C ≅ C⁺ ⊞ C⁻`, `homologyEquivEigen : H_i(C) ≃ H_i(C⁺) × H_i(C⁻)` and
  `homologyIsoBiprodEigen`. The summands are the `+1` and `-1` eigenspaces of `σ` on homology
  (`range_homologyMap_plusι`, `range_homologyMap_minusι`, `injective_homologyMap_plusι`,
  `injective_homologyMap_minusι`, `isCompl_range_homologyMap`), and a chain map commuting with
  the involutions preserves the splitting (`homologyEquivEigen_naturality`). For a cell complex,
  `ChainMap.ofInvolutive` is the action of an involution of the cells that preserves the
  incidence numbers, whose eigen-subcomplexes are the chains of `symChains` and `antiChains` of
  `SignChains` (`ChainMap.mem_plus_ofInvolutive_iff`, `ChainMap.mem_minus_ofInvolutive_iff`).
* Nilpotency (over a field the index is trivial if and only if the index
  map is nilpotent): `isNilpotent_homologyMap_iff` for a chain endomorphism `T` commuting with
  `σ`, and `isNilpotent_iff_homology_eigenspaces` for any endomorphism of `H_i(C)` commuting with
  the action of `σ`, from `isNilpotent_iff_eigenspaces` of `LatentEquivariant.Algebra.Isotypic`.

None of this needs a field or finite dimension: `k` is any commutative ring, with `2` invertible
for the splitting, and the complexes of modules may have any `ComplexShape`.

The orbit-space terms of the proposition are not identified in this file. The identification of
`C⁺` with the chains of the orbit space and of `C⁻` with the twisted complex is degreewise linear
algebra (`OrbitChains`, `SignChains`); `isoOfLinearEquivs` and
`HomologicalComplex.homologyMapIso` are the tools to turn it into isomorphisms of complexes and of
homology. Cellular homology is not compared with singular homology here.
-/

universe u v w

namespace LatentEquivariant

open CategoryTheory Limits

/-! ### Linear algebra -/

section LinearAlgebra

variable {k : Type u} [CommRing k] {M N : Type*} [AddCommGroup M] [Module k M] [AddCommGroup N]
  [Module k N]

/-- Nilpotency is invariant under conjugation by a linear equivalence. -/
theorem isNilpotent_iff_of_linearEquiv (e : M ≃ₗ[k] N) {f : Module.End k M}
    {g : Module.End k N} (h : ∀ x, e (f x) = g (e x)) : IsNilpotent f ↔ IsNilpotent g := by
  have hg : e.conjAlgEquiv k f = g := LinearMap.ext fun y ↦ by
    simp [LinearEquiv.conjAlgEquiv_apply, h]
  rw [← hg]
  exact (IsNilpotent.map_iff (e.conjAlgEquiv k).injective).symm

/-- A pair is nilpotent if and only if both entries are. -/
theorem isNilpotent_prod_mk_iff {R S : Type*} [MonoidWithZero R] [MonoidWithZero S] {a : R}
    {b : S} : IsNilpotent (a, b) ↔ IsNilpotent a ∧ IsNilpotent b := by
  constructor
  · rintro ⟨n, hn⟩
    rw [Prod.pow_mk, Prod.mk_eq_zero] at hn
    exact ⟨⟨n, hn.1⟩, ⟨n, hn.2⟩⟩
  · rintro ⟨⟨m, hm⟩, ⟨n, hn⟩⟩
    exact ⟨max m n, by rw [Prod.pow_mk, pow_eq_zero_of_le (le_max_left m n) hm,
      pow_eq_zero_of_le (le_max_right m n) hn, Prod.mk_zero_zero]⟩

/-- `f × g` is nilpotent if and only if `f` and `g` are. -/
theorem isNilpotent_prodMap_iff {f : Module.End k M} {g : Module.End k N} :
    IsNilpotent (f.prodMap g) ↔ IsNilpotent f ∧ IsNilpotent g := by
  have hinj : Function.Injective (LinearMap.prodMapAlgHom k M N) := by
    rintro ⟨f₁, g₁⟩ ⟨f₂, g₂⟩ h
    exact Prod.ext (LinearMap.ext fun x ↦ congrArg Prod.fst (LinearMap.congr_fun h (x, 0)))
      (LinearMap.ext fun y ↦ congrArg Prod.snd (LinearMap.congr_fun h (0, y)))
  exact (IsNilpotent.map_iff (f := LinearMap.prodMapAlgHom k M N) hinj).trans
    isNilpotent_prod_mk_iff

variable [Invertible (2 : k)]

/-- `(x + x) / 2 = x` when `2` is invertible. -/
theorem invOf_two_smul_add_self (x : M) : ⅟(2 : k) • (x + x) = x := by
  rw [← two_smul k x, smul_smul, invOf_mul_self, one_smul]

variable (k) in
include k in
/-- If `2` is invertible in `k`, then `x = -x` only for `x = 0` in a `k`-module. -/
theorem eq_zero_of_eq_neg_of_invertible {x : M} (h : x = -x) : x = 0 := by
  rw [← invOf_two_smul_add_self (k := k) x]
  nth_rewrite 2 [h]
  rw [add_neg_cancel, smul_zero]

end LinearAlgebra

/-! ### Homology of short complexes of modules -/

section ModuleCatHomology

variable {k : Type u} [CommRing k]

/-- The homology of a short complex `S` of `k`-modules is `Z ⧸ B`, for any submodules `Z` and `B`
of `S.X₂` equal to `ker S.g` and `range S.f`. This is Mathlib's `moduleCatHomologyIso` followed by
the change of names. -/
noncomputable def moduleCatHomologyEquiv (S : ShortComplex (ModuleCat.{w} k))
    {Z B : Submodule k S.X₂} (hZ : LinearMap.ker S.g.hom = Z) (hB : LinearMap.range S.f.hom = B) :
    S.homology ≃ₗ[k] Z ⧸ B.comap Z.subtype :=
  S.moduleCatLeftHomologyData.homologyIso.toLinearEquiv.trans
    (Submodule.Quotient.equiv (LinearMap.range S.moduleCatToCycles) (B.comap Z.subtype)
      (LinearEquiv.ofEq _ _ hZ) (by
        subst hZ hB
        ext x
        simp [LinearMap.range_codRestrict]))

variable {S₁ S₂ : ShortComplex (ModuleCat.{w} k)} (φ : S₁ ⟶ S₂)

/-- A morphism of short complexes of modules maps `ker g` into `ker g`. -/
theorem moduleCat_mem_ker_map {x : S₁.X₂} (hx : x ∈ LinearMap.ker S₁.g.hom) :
    φ.τ₂.hom x ∈ LinearMap.ker S₂.g.hom := by
  rw [LinearMap.mem_ker] at hx ⊢
  have h := LinearMap.congr_fun (congrArg ModuleCat.Hom.hom φ.comm₂₃) x
  simp only [ModuleCat.hom_comp, LinearMap.comp_apply] at h
  rw [h, hx, map_zero]

/-- The maps induced by a morphism of short complexes of modules on `ker g` and on
`ker g ⧸ range f`, as data computing its homology map. -/
noncomputable def moduleCatLeftHomologyMapData :
    ShortComplex.LeftHomologyMapData φ S₁.moduleCatLeftHomologyData
      S₂.moduleCatLeftHomologyData where
  φK := ModuleCat.ofHom (φ.τ₂.hom.restrict fun _ ↦ moduleCat_mem_ker_map φ)
  φH := ModuleCat.ofHom (Submodule.mapQ _ _ (φ.τ₂.hom.restrict fun _ ↦ moduleCat_mem_ker_map φ)
    (by
      rintro _ ⟨y, rfl⟩
      exact ⟨φ.τ₁.hom y, Subtype.ext
        (LinearMap.congr_fun (congrArg ModuleCat.Hom.hom φ.comm₁₂) y)⟩))
  commi := by ext; rfl
  commf' := by
    ext y
    exact Subtype.ext (LinearMap.congr_fun (congrArg ModuleCat.Hom.hom φ.comm₁₂) y).symm
  commπ := by ext; rfl

/-- Naturality of `moduleCatHomologyEquiv`: under it, the homology map of `φ` is the map induced
by `φ.τ₂` on `Z ⧸ B`. -/
theorem moduleCatHomologyEquiv_naturality {Z₁ B₁ : Submodule k S₁.X₂}
    {Z₂ B₂ : Submodule k S₂.X₂} (hZ₁ : LinearMap.ker S₁.g.hom = Z₁)
    (hB₁ : LinearMap.range S₁.f.hom = B₁) (hZ₂ : LinearMap.ker S₂.g.hom = Z₂)
    (hB₂ : LinearMap.range S₂.f.hom = B₂) (hφZ : ∀ x ∈ Z₁, φ.τ₂.hom x ∈ Z₂)
    (hφB : B₁.comap Z₁.subtype ≤ (B₂.comap Z₂.subtype).comap (φ.τ₂.hom.restrict hφZ))
    (x : S₁.homology) :
    moduleCatHomologyEquiv S₂ hZ₂ hB₂ ((ShortComplex.homologyMap φ).hom x) =
      Submodule.mapQ _ _ (φ.τ₂.hom.restrict hφZ) hφB (moduleCatHomologyEquiv S₁ hZ₁ hB₁ x) := by
  have h := LinearMap.congr_fun
    (congrArg ModuleCat.Hom.hom (moduleCatLeftHomologyMapData φ).homologyMap_comm) x
  simp only [ModuleCat.hom_comp, LinearMap.comp_apply] at h
  simp only [moduleCatHomologyEquiv, LinearEquiv.trans_apply, Iso.toLinearEquiv_apply, h]
  generalize (S₁.moduleCatLeftHomologyData.homologyIso.hom).hom x = y
  induction y using Submodule.Quotient.induction_on with
  | H z => rfl

end ModuleCatHomology

/-! ### Finite cell complexes and their chain complexes -/

/-- A finite cell complex over a commutative ring `k`, given combinatorially: finitely many cells
in each dimension, incidence numbers `[e : e']` of cells in consecutive dimensions, and `∂ ∘ ∂ = 0`
for the boundary `bd`. This models the cellular chains of a grid as CMGDB builds them, whose
relative homology is `H_*(N, L)`. -/
structure CellComplex (k : Type u) [CommRing k] where
  /-- The cells of dimension `n`. -/
  cells : ℕ → Type v
  /-- Each dimension has finitely many cells. -/
  [fintype : ∀ n, Fintype (cells n)]
  /-- The incidence number `[e : e']` of an `(n + 1)`-cell `e` and an `n`-cell `e'`. -/
  inc : ∀ n, cells (n + 1) → cells n → k
  /-- The boundary squares to zero. -/
  bd_bd : ∀ n c, bd (inc n) (bd (inc (n + 1)) c) = 0

attribute [instance] CellComplex.fintype

namespace CellComplex

variable {k : Type u} [CommRing k] (K : CellComplex.{u, v} k)

/-- `∂ ∘ ∂ = 0` as an equation of linear maps. -/
theorem bd_comp_bd (n : ℕ) : bd (K.inc n) ∘ₗ bd (K.inc (n + 1)) = 0 :=
  LinearMap.ext (K.bd_bd n)

/-- The cellular chain complex `C_*(K; k)`, with `C_n = (cells n → k)` and differential
`bd (inc n) : C_(n + 1) → C_n`, as a chain complex of `k`-modules indexed by `ℕ`. Its homology
`K.toChainComplex.homology n` is `H_n` through Mathlib's API. -/
abbrev toChainComplex : ChainComplex (ModuleCat.{max v u} k) ℕ :=
  ChainComplex.of (fun n ↦ ModuleCat.of k (K.cells n → k))
    (fun n ↦ ModuleCat.ofHom (bd (K.inc n)))
    (fun n ↦ ModuleCat.hom_ext (K.bd_comp_bd n))

@[simp] theorem toChainComplex_X (n : ℕ) :
    K.toChainComplex.X n = ModuleCat.of k (K.cells n → k) :=
  rfl

@[simp] theorem toChainComplex_d (n : ℕ) :
    K.toChainComplex.d (n + 1) n = ModuleCat.ofHom (bd (K.inc n)) := by
  simp [toChainComplex]

@[simp] theorem toChainComplex_d_hom (n : ℕ) :
    (K.toChainComplex.d (n + 1) n).hom = bd (K.inc n) := by
  simp [toChainComplex]

@[simp] theorem toChainComplex_d_zero (j : ℕ) : K.toChainComplex.d 0 j = 0 :=
  K.toChainComplex.shape _ _ (by simp)

/-- The boundary `∂_n : C_n → C_(n - 1)` out of degree `n`, with `∂_0 = 0`. -/
def dOut : ∀ n : ℕ, (K.cells n → k) →ₗ[k] (K.cells (n - 1) → k)
  | 0 => 0
  | n + 1 => bd (K.inc n)

/-- The cycles `Z_n = ker ∂_n`. -/
def cycles (n : ℕ) : Submodule k (K.cells n → k) :=
  LinearMap.ker (K.dOut n)

/-- The boundaries `B_n = range ∂_(n + 1)`. -/
def boundaries (n : ℕ) : Submodule k (K.cells n → k) :=
  LinearMap.range (bd (K.inc n))

theorem mem_cycles_zero (c : K.cells 0 → k) : c ∈ K.cycles 0 := by
  simp [cycles, dOut]

theorem mem_cycles_succ {n : ℕ} {c : K.cells (n + 1) → k} :
    c ∈ K.cycles (n + 1) ↔ bd (K.inc n) c = 0 :=
  Iff.rfl

/-- Every boundary is a cycle. -/
theorem boundaries_le_cycles (n : ℕ) : K.boundaries n ≤ K.cycles n := by
  rintro _ ⟨c, rfl⟩
  cases n with
  | zero => exact K.mem_cycles_zero _
  | succ n => exact K.bd_bd n c

/-- Cellular homology `H_n(K; k) = ker ∂_n ⧸ range ∂_(n + 1)`, as a `k`-module: the concrete form
of `H_*(N, L)`. -/
abbrev H (n : ℕ) : Type (max v u) :=
  K.cycles n ⧸ (K.boundaries n).comap (K.cycles n).subtype

/-- The cycles of Mathlib's short complex in degree `n` are `ker ∂_n`. -/
theorem ker_sc_g (n : ℕ) : LinearMap.ker (K.toChainComplex.sc n).g.hom = K.cycles n := by
  cases n with
  | zero =>
    show LinearMap.ker (K.toChainComplex.d 0 _).hom = _
    rw [toChainComplex_d_zero]
    ext x
    simp [cycles, dOut]
  | succ n =>
    show LinearMap.ker (K.toChainComplex.d (n + 1) ((ComplexShape.down ℕ).next (n + 1))).hom = _
    rw [ChainComplex.next_nat_succ, toChainComplex_d_hom]
    rfl

/-- The boundaries of Mathlib's short complex in degree `n` are `range ∂_(n + 1)`. -/
theorem range_sc_f (n : ℕ) :
    LinearMap.range (K.toChainComplex.sc n).f.hom = K.boundaries n := by
  show LinearMap.range (K.toChainComplex.d ((ComplexShape.down ℕ).prev n) n).hom = _
  rw [ChainComplex.prev, toChainComplex_d_hom]
  rfl

/-- Mathlib's homology of `C_*(K; k)` in degree `n` is `ker ∂_n ⧸ range ∂_(n + 1)`. -/
noncomputable def homologyEquiv (n : ℕ) : K.toChainComplex.homology n ≃ₗ[k] K.H n :=
  moduleCatHomologyEquiv (K.toChainComplex.sc n) (K.ker_sc_g n) (K.range_sc_f n)

/-! ### Chain maps -/

/-- A chain map of cell complexes: linear maps of chains in each degree that commute with the
boundaries. Chain-level index maps and the action of a group on the chains are of this form. -/
@[ext]
structure ChainMap (K L : CellComplex.{u, v} k) where
  /-- The linear map in degree `n`. -/
  app : ∀ n, (K.cells n → k) →ₗ[k] (L.cells n → k)
  /-- The map commutes with the boundaries. -/
  comm : ∀ n, bd (L.inc n) ∘ₗ app (n + 1) = app n ∘ₗ bd (K.inc n)

namespace ChainMap

variable {K} {L M : CellComplex.{u, v} k}

/-- The identity chain map. -/
@[simps]
def id (K : CellComplex.{u, v} k) : K.ChainMap K where
  app _ := LinearMap.id
  comm _ := rfl

/-- The composition `ψ ∘ φ` of chain maps. -/
@[simps]
def comp (ψ : L.ChainMap M) (φ : K.ChainMap L) : K.ChainMap M where
  app n := ψ.app n ∘ₗ φ.app n
  comm n := by
    rw [← LinearMap.comp_assoc, ψ.comm, LinearMap.comp_assoc, φ.comm, LinearMap.comp_assoc]

/-- The morphism of chain complexes `C_*(K) ⟶ C_*(L)` of a chain map. -/
def toHom (φ : K.ChainMap L) : K.toChainComplex ⟶ L.toChainComplex :=
  ChainComplex.ofHom (fun n ↦ ModuleCat.ofHom (φ.app n))
    (fun n ↦ ModuleCat.hom_ext (by simpa using φ.comm n))

@[simp] theorem toHom_f (φ : K.ChainMap L) (n : ℕ) :
    φ.toHom.f n = ModuleCat.ofHom (φ.app n) :=
  rfl

@[simp] theorem toHom_id : (ChainMap.id K).toHom = 𝟙 K.toChainComplex := by
  ext n : 2
  rfl

@[simp] theorem toHom_comp (ψ : L.ChainMap M) (φ : K.ChainMap L) :
    (ψ.comp φ).toHom = φ.toHom ≫ ψ.toHom := by
  ext n : 2
  rfl

/-- The linear map induced on Mathlib's homology `H_n(K) → H_n(L)`. -/
noncomputable def homologyMap (φ : K.ChainMap L) (n : ℕ) :
    K.toChainComplex.homology n →ₗ[k] L.toChainComplex.homology n :=
  (HomologicalComplex.homologyMap φ.toHom n).hom

/-- The identity induces the identity on homology. -/
@[simp] theorem homologyMap_id (n : ℕ) : (ChainMap.id K).homologyMap n = LinearMap.id := by
  simp [homologyMap]

/-- Homology is functorial: `(ψ ∘ φ)_* = ψ_* ∘ φ_*`. -/
theorem homologyMap_comp (ψ : L.ChainMap M) (φ : K.ChainMap L) (n : ℕ) :
    (ψ.comp φ).homologyMap n = ψ.homologyMap n ∘ₗ φ.homologyMap n := by
  simp [homologyMap, HomologicalComplex.homologyMap_comp]

/-- A chain map maps cycles to cycles. -/
theorem mapsTo_cycles (φ : K.ChainMap L) (n : ℕ) :
    ∀ c ∈ K.cycles n, φ.app n c ∈ L.cycles n := by
  intro c hc
  cases n with
  | zero => exact L.mem_cycles_zero _
  | succ n =>
    rw [mem_cycles_succ] at hc ⊢
    rw [← LinearMap.comp_apply, φ.comm, LinearMap.comp_apply, hc, map_zero]

/-- A chain map maps boundaries to boundaries. -/
theorem le_comap_boundaries (φ : K.ChainMap L) (n : ℕ) :
    (K.boundaries n).comap (K.cycles n).subtype ≤
      ((L.boundaries n).comap (L.cycles n).subtype).comap
        ((φ.app n).restrict (φ.mapsTo_cycles n)) := by
  rintro z ⟨c, hc⟩
  refine ⟨φ.app (n + 1) c, ?_⟩
  simp only [Submodule.coe_subtype, LinearMap.coe_restrict_apply] at hc ⊢
  rw [← LinearMap.comp_apply, φ.comm, LinearMap.comp_apply, hc]

/-- The linear map induced on `ker ∂ ⧸ range ∂`, `[z] ↦ [φ z]`. -/
def mapH (φ : K.ChainMap L) (n : ℕ) : K.H n →ₗ[k] L.H n :=
  Submodule.mapQ _ _ ((φ.app n).restrict (φ.mapsTo_cycles n)) (φ.le_comap_boundaries n)

@[simp] theorem mapH_mk (φ : K.ChainMap L) (n : ℕ) (z : K.cycles n) :
    φ.mapH n (Submodule.Quotient.mk z) =
      Submodule.Quotient.mk ⟨φ.app n z, φ.mapsTo_cycles n z z.2⟩ :=
  rfl

/-- The identity induces the identity on `ker ∂ ⧸ range ∂`. -/
@[simp] theorem mapH_id (n : ℕ) : (ChainMap.id K).mapH n = LinearMap.id := by
  ext z
  rfl

/-- Functoriality on `ker ∂ ⧸ range ∂`: `(ψ ∘ φ)_* = ψ_* ∘ φ_*`. -/
theorem mapH_comp (ψ : L.ChainMap M) (φ : K.ChainMap L) (n : ℕ) :
    (ψ.comp φ).mapH n = ψ.mapH n ∘ₗ φ.mapH n := by
  ext z
  rfl

/-- `homologyEquiv` is natural: it intertwines `homologyMap` and `mapH`. -/
theorem homologyEquiv_naturality (φ : K.ChainMap L) (n : ℕ)
    (x : K.toChainComplex.homology n) :
    L.homologyEquiv n (φ.homologyMap n x) = φ.mapH n (K.homologyEquiv n x) :=
  moduleCatHomologyEquiv_naturality ((HomologicalComplex.shortComplexFunctor _ _ n).map φ.toHom)
    (K.ker_sc_g n) (K.range_sc_f n) (L.ker_sc_g n) (L.range_sc_f n) (φ.mapsTo_cycles n)
    (φ.le_comap_boundaries n) x

/-- An endomorphism is nilpotent on `ker ∂ ⧸ range ∂` if and only if it is nilpotent on Mathlib's
homology. -/
theorem isNilpotent_mapH_iff (T : K.ChainMap K) (n : ℕ) :
    IsNilpotent (T.mapH n) ↔ IsNilpotent (T.homologyMap n) :=
  (isNilpotent_iff_of_linearEquiv (K.homologyEquiv n) (T.homologyEquiv_naturality n)).symm

end ChainMap

/-- An isomorphism of cell complexes: linear equivalences of chains in each degree that commute
with the boundaries. -/
structure ChainEquiv (K L : CellComplex.{u, v} k) where
  /-- The linear equivalence in degree `n`. -/
  app : ∀ n, (K.cells n → k) ≃ₗ[k] (L.cells n → k)
  /-- The equivalence commutes with the boundaries. -/
  comm : ∀ n, bd (L.inc n) ∘ₗ (app (n + 1)).toLinearMap = (app n).toLinearMap ∘ₗ bd (K.inc n)

namespace ChainEquiv

variable {K} {L : CellComplex.{u, v} k}

/-- The chain map of a chain equivalence. -/
def toChainMap (e : K.ChainEquiv L) : K.ChainMap L where
  app n := e.app n
  comm := e.comm

/-- The inverse chain equivalence. -/
def symm (e : K.ChainEquiv L) : L.ChainEquiv K where
  app n := (e.app n).symm
  comm n := by
    ext c : 1
    apply (e.app n).injective
    have h := LinearMap.congr_fun (e.comm n) ((e.app (n + 1)).symm c)
    simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply] at h ⊢
    exact h.symm

@[simp] theorem symm_comp (e : K.ChainEquiv L) :
    e.symm.toChainMap.comp e.toChainMap = ChainMap.id K :=
  ChainMap.ext (funext fun n ↦ LinearMap.ext fun c ↦ by simp [toChainMap, symm])

@[simp] theorem comp_symm (e : K.ChainEquiv L) :
    e.toChainMap.comp e.symm.toChainMap = ChainMap.id L :=
  ChainMap.ext (funext fun n ↦ LinearMap.ext fun c ↦ by simp [toChainMap, symm])

/-- A chain equivalence is an isomorphism of chain complexes. -/
def toIso (e : K.ChainEquiv L) : K.toChainComplex ≅ L.toChainComplex where
  hom := e.toChainMap.toHom
  inv := e.symm.toChainMap.toHom
  hom_inv_id := by rw [← ChainMap.toHom_comp, symm_comp, ChainMap.toHom_id]
  inv_hom_id := by rw [← ChainMap.toHom_comp, comp_symm, ChainMap.toHom_id]

/-- Isomorphic cell complexes have isomorphic homology. -/
noncomputable def homologyLinearEquiv (e : K.ChainEquiv L) (n : ℕ) :
    K.toChainComplex.homology n ≃ₗ[k] L.toChainComplex.homology n :=
  (HomologicalComplex.homologyMapIso e.toIso n).toLinearEquiv

/-- Isomorphic cell complexes have isomorphic modules `ker ∂ ⧸ range ∂`. -/
def equivH (e : K.ChainEquiv L) (n : ℕ) : K.H n ≃ₗ[k] L.H n :=
  LinearEquiv.ofLinear (e.toChainMap.mapH n) (e.symm.toChainMap.mapH n)
    (by rw [← ChainMap.mapH_comp, comp_symm, ChainMap.mapH_id])
    (by rw [← ChainMap.mapH_comp, symm_comp, ChainMap.mapH_id])

end ChainEquiv

/-! ### Restriction to a set of cells and relative complexes -/

/-- A family of sets of cells is convex for the face relation if no product of incidences
`[l : i] [i : j]` with `l` and `j` in it passes through a cell `i` outside it. Then the restricted
incidence numbers still satisfy `∂ ∘ ∂ = 0` (`restrict`). -/
def IsFaceConvex (S : ∀ n, Set (K.cells n)) : Prop :=
  ∀ n (l : K.cells (n + 1 + 1)) (i : K.cells (n + 1)) (j : K.cells n), l ∈ S (n + 1 + 1) →
    i ∉ S (n + 1) → j ∈ S n → K.inc (n + 1) l i * K.inc n i j = 0

/-- A family of sets of cells is a subcomplex if it contains every face of each of its cells,
where a face is a cell of one dimension less with nonzero incidence number. -/
def IsSubcomplex (N : ∀ n, Set (K.cells n)) : Prop :=
  ∀ n (e : K.cells (n + 1)) (e' : K.cells n), e ∈ N (n + 1) → K.inc n e e' ≠ 0 → e' ∈ N n

variable {K}

/-- A subcomplex is convex for the face relation. -/
theorem IsSubcomplex.isFaceConvex {N : ∀ n, Set (K.cells n)} (hN : K.IsSubcomplex N) :
    K.IsFaceConvex N := by
  intro n l i j hl hi _
  have : K.inc (n + 1) l i = 0 := by
    by_contra h
    exact hi (hN (n + 1) l i hl h)
  rw [this, zero_mul]

/-- For subcomplexes `N` and `L`, the cells of `N` not in `L` are convex for the face
relation. -/
theorem IsSubcomplex.isFaceConvex_diff {N L : ∀ n, Set (K.cells n)} (hN : K.IsSubcomplex N)
    (hL : K.IsSubcomplex L) : K.IsFaceConvex fun n ↦ N n \ L n := by
  intro n l i j hl hi hj
  by_cases hiN : i ∈ N (n + 1)
  · have hiL : i ∈ L (n + 1) := by
      by_contra h
      exact hi ⟨hiN, h⟩
    have : K.inc n i j = 0 := by
      by_contra h
      exact hj.2 (hL n i j hiL h)
    rw [this, mul_zero]
  · have : K.inc (n + 1) l i = 0 := by
      by_contra h
      exact hiN (hN (n + 1) l i hl.1 h)
    rw [this, zero_mul]

variable (K)

/-- The cell complex on a family of sets of cells that is convex for the face relation, with the
restricted incidence numbers. -/
def restrict (S : ∀ n, Set (K.cells n)) [∀ n, DecidablePred (· ∈ S n)] (hS : K.IsFaceConvex S) :
    CellComplex.{u, v} k where
  cells n := S n
  inc n e e' := K.inc n e e'
  bd_bd n c := by
    ext j
    let c' : K.cells (n + 1 + 1) → k := fun l ↦ if h : l ∈ S (n + 1 + 1) then c ⟨l, h⟩ else 0
    have hc' : ∀ i, ∑ l, c' l * K.inc (n + 1) l i =
        ∑ l : S (n + 1 + 1), c l * K.inc (n + 1) l i := by
      intro i
      rw [← Fintype.sum_subtype_add_sum_subtype (· ∈ S (n + 1 + 1))]
      have h0 : ∑ l : {l // l ∉ S (n + 1 + 1)}, c' l * K.inc (n + 1) l i = 0 :=
        Finset.sum_eq_zero fun l _ ↦ by simp [c', l.2]
      rw [h0, add_zero]
      exact Finset.sum_congr rfl fun l _ ↦ by simp [c', l.2]
    have hout : ∑ i : {i // i ∉ S (n + 1)}, (∑ l, c' l * K.inc (n + 1) l i) * K.inc n i j = 0 := by
      refine Finset.sum_eq_zero fun i _ ↦ ?_
      rw [Finset.sum_mul]
      refine Finset.sum_eq_zero fun l _ ↦ ?_
      by_cases hl : l ∈ S (n + 1 + 1)
      · rw [mul_assoc, hS n l i j hl i.2 j.2, mul_zero]
      · simp [c', hl]
    have h := congr_fun (K.bd_bd n c') j
    simp only [bd_apply, Pi.zero_apply] at h ⊢
    rw [← Fintype.sum_subtype_add_sum_subtype (· ∈ S (n + 1)), hout, add_zero] at h
    simpa only [hc'] using h

/-- The relative cell complex `C_*(N, L)` of subcomplexes `N` and `L`: the cells of `N` not in
`L`, with the restricted incidence numbers. This is the model of relative chains in
`OrbitChains` and `SignChains`. -/
def relative (N L : ∀ n, Set (K.cells n)) [∀ n, DecidablePred (· ∈ N n)]
    [∀ n, DecidablePred (· ∈ L n)] (hN : K.IsSubcomplex N) (hL : K.IsSubcomplex L) :
    CellComplex.{u, v} k :=
  K.restrict (fun n ↦ N n \ L n) (hN.isFaceConvex_diff hL)

variable {K} {S : ∀ n, Set (K.cells n)} [∀ n, DecidablePred (· ∈ S n)] {hS : K.IsFaceConvex S}

/-- A map of cells that preserves `S` restricts to the cells of `K.restrict S hS`. -/
def restrictCells (s : ∀ n, K.cells n → K.cells n) (hsS : ∀ n, ∀ e ∈ S n, s n e ∈ S n) (n : ℕ)
    (e : (K.restrict S hS).cells n) : (K.restrict S hS).cells n :=
  ⟨s n e.1, hsS n e.1 e.2⟩

/-- The restriction of an involution of the cells is an involution. -/
theorem involutive_restrictCells {s : ∀ n, K.cells n → K.cells n}
    (hsS : ∀ n, ∀ e ∈ S n, s n e ∈ S n) (hs : ∀ n, Function.Involutive (s n)) (n : ℕ) :
    Function.Involutive (restrictCells (hS := hS) s hsS n) :=
  fun e ↦ Subtype.ext (hs n e.1)

/-- The restriction of a map of cells that preserves the incidence numbers preserves the
restricted incidence numbers. -/
theorem inc_restrictCells {s : ∀ n, K.cells n → K.cells n} (hsS : ∀ n, ∀ e ∈ S n, s n e ∈ S n)
    (hinc : ∀ n e e', K.inc n (s (n + 1) e) (s n e') = K.inc n e e') (n : ℕ)
    (e : (K.restrict S hS).cells (n + 1)) (e' : (K.restrict S hS).cells n) :
    (K.restrict S hS).inc n (restrictCells s hsS (n + 1) e) (restrictCells s hsS n e') =
      (K.restrict S hS).inc n e e' :=
  hinc n e.1 e'.1

/-- An involution of the cells that preserves the subcomplexes `N` and `L` preserves the cells
of `N` not in `L`, so it acts on `C_*(N, L)` (Proposition `prop:z2_reconstruction` for
`G`-invariant subcomplexes). -/
theorem mapsTo_diff_of_involutive {s : ∀ n, K.cells n → K.cells n}
    (hs : ∀ n, Function.Involutive (s n)) {N L : ∀ n, Set (K.cells n)}
    (hsN : ∀ n, ∀ e ∈ N n, s n e ∈ N n) (hsL : ∀ n, ∀ e ∈ L n, s n e ∈ L n) :
    ∀ n, ∀ e ∈ N n \ L n, s n e ∈ N n \ L n := fun n e he ↦
  ⟨hsN n e he.1, fun h ↦ he.2 (by simpa only [hs n e] using hsL n (s n e) h)⟩

end CellComplex

/-! ### Subcomplexes of complexes of modules -/

section Subcomplex

variable {k : Type u} [CommRing k] {ι : Type*} {c : ComplexShape ι}

/-- An isomorphism of complexes of modules from linear equivalences in each degree that commute
with the differentials. -/
def isoOfLinearEquivs {C D : HomologicalComplex (ModuleCat.{w} k) c}
    (e : ∀ i, C.X i ≃ₗ[k] D.X i)
    (he : ∀ i j, c.Rel i j →
      (D.d i j).hom ∘ₗ (e i).toLinearMap = (e j).toLinearMap ∘ₗ (C.d i j).hom) :
    C ≅ D :=
  HomologicalComplex.Hom.isoOfComponents (fun i ↦ (e i).toModuleIso) fun i j hij ↦
    ModuleCat.hom_ext (he i j hij)

/-- The subcomplex of `C` with terms `p i ⊆ C_i`, for submodules that the differentials
preserve. -/
abbrev subcomplex (C : HomologicalComplex (ModuleCat.{w} k) c) (p : ∀ i, Submodule k (C.X i))
    (hp : ∀ i j, ∀ x ∈ p i, (C.d i j).hom x ∈ p j) : HomologicalComplex (ModuleCat.{w} k) c where
  X i := ModuleCat.of k (p i)
  d i j := ModuleCat.ofHom ((C.d i j).hom.restrict (hp i j))
  shape i j hij := by
    ext x
    simp [C.shape i j hij]
  d_comp_d' i j l _ _ := by
    ext x
    have h := LinearMap.congr_fun (congrArg ModuleCat.Hom.hom (C.d_comp_d i j l)) x.1
    rw [ModuleCat.hom_comp, LinearMap.comp_apply] at h
    simpa using h

variable {C D : HomologicalComplex (ModuleCat.{w} k) c} {p : ∀ i, Submodule k (C.X i)}
  {hp : ∀ i j, ∀ x ∈ p i, (C.d i j).hom x ∈ p j}

/-- The inclusion of a subcomplex. -/
@[simps]
def subcomplexι : subcomplex C p hp ⟶ C where
  f i := ModuleCat.ofHom (p i).subtype
  comm' i j _ := by
    ext
    rfl

/-- Two chain maps to a subcomplex agree if they agree after the inclusion. -/
theorem subcomplex_hom_ext {φ ψ : D ⟶ subcomplex C p hp}
    (h : φ ≫ subcomplexι = ψ ≫ subcomplexι) : φ = ψ := by
  ext i x
  exact LinearMap.congr_fun (congrArg ModuleCat.Hom.hom (HomologicalComplex.congr_hom h i)) x

/-- A chain map with image in a subcomplex, as a chain map to the subcomplex. -/
@[simps]
def subcomplexLift (φ : D ⟶ C) (hφ : ∀ i x, (φ.f i).hom x ∈ p i) : D ⟶ subcomplex C p hp where
  f i := ModuleCat.ofHom ((φ.f i).hom.codRestrict (p i) (hφ i))
  comm' i j _ := by
    ext x
    have h := LinearMap.congr_fun (congrArg ModuleCat.Hom.hom (φ.comm i j)) x
    rw [ModuleCat.hom_comp, ModuleCat.hom_comp, LinearMap.comp_apply, LinearMap.comp_apply] at h
    simpa using h

@[reassoc (attr := simp)]
theorem subcomplexLift_ι (φ : D ⟶ C) (hφ : ∀ i x, (φ.f i).hom x ∈ p i) :
    subcomplexLift (hp := hp) φ hφ ≫ subcomplexι = φ := by
  ext
  rfl

/-- The restriction of a chain map to subcomplexes that it preserves. -/
def subcomplexMap {q : ∀ i, Submodule k (D.X i)} {hq : ∀ i j, ∀ x ∈ q i, (D.d i j).hom x ∈ q j}
    (φ : C ⟶ D) (hφ : ∀ i, ∀ x ∈ p i, (φ.f i).hom x ∈ q i) :
    subcomplex C p hp ⟶ subcomplex D q hq :=
  subcomplexLift (subcomplexι ≫ φ) fun i x ↦ hφ i x.1 x.2

@[reassoc (attr := simp)]
theorem subcomplexMap_ι {q : ∀ i, Submodule k (D.X i)}
    {hq : ∀ i j, ∀ x ∈ q i, (D.d i j).hom x ∈ q j} (φ : C ⟶ D)
    (hφ : ∀ i, ∀ x ∈ p i, (φ.f i).hom x ∈ q i) :
    subcomplexMap (hp := hp) (hq := hq) φ hφ ≫ subcomplexι = subcomplexι ≫ φ :=
  subcomplexLift_ι _ _

/-- The kernel of a chain map, as a subcomplex. -/
abbrev kerSubcomplex (φ : C ⟶ D) : HomologicalComplex (ModuleCat.{w} k) c :=
  subcomplex C (fun i ↦ LinearMap.ker (φ.f i).hom) fun i j x hx ↦ by
    rw [LinearMap.mem_ker] at hx ⊢
    have h := LinearMap.congr_fun (congrArg ModuleCat.Hom.hom (φ.comm i j)) x
    simp only [ModuleCat.hom_comp, LinearMap.comp_apply] at h
    rw [← h, hx, map_zero]

variable (σ : C ⟶ C)

/-- The `+1` eigen-subcomplex `C⁺ = ker (σ - 1)` of a chain endomorphism `σ`. For the action of
`g` on `C_*(N, L)` it is the `+1` eigenspace of Proposition `prop:z2_reconstruction`. -/
abbrev plusSubcomplex : HomologicalComplex (ModuleCat.{w} k) c :=
  kerSubcomplex (σ - 𝟙 C)

/-- The `-1` eigen-subcomplex `C⁻ = ker (σ + 1)` of a chain endomorphism `σ`. For the action of
`g` on `C_*(N, L)` it is the `-1` eigenspace of Proposition `prop:z2_reconstruction`. -/
abbrev minusSubcomplex : HomologicalComplex (ModuleCat.{w} k) c :=
  kerSubcomplex (σ + 𝟙 C)

theorem mem_ker_sub_id {i : ι} {x : C.X i} :
    x ∈ LinearMap.ker ((σ - 𝟙 C).f i).hom ↔ (σ.f i).hom x = x := by
  simp [sub_eq_zero]

theorem mem_ker_add_id {i : ι} {x : C.X i} :
    x ∈ LinearMap.ker ((σ + 𝟙 C).f i).hom ↔ (σ.f i).hom x = -x := by
  simp [add_eq_zero_iff_eq_neg]

end Subcomplex

/-! ### Homology of a direct sum of complexes -/

section Biproduct

variable {k : Type u} [CommRing k] {ι : Type*} {c : ComplexShape ι}

/-- Functoriality of homology, applied to an element. -/
theorem homologyMap_hom_comp_apply {X Y Z : HomologicalComplex (ModuleCat.{w} k) c}
    (φ : X ⟶ Y) (ψ : Y ⟶ Z) (i : ι) (x : X.homology i) :
    (HomologicalComplex.homologyMap ψ i).hom ((HomologicalComplex.homologyMap φ i).hom x) =
      (HomologicalComplex.homologyMap (φ ≫ ψ) i).hom x := by
  rw [HomologicalComplex.homologyMap_comp, ModuleCat.hom_comp, LinearMap.comp_apply]

variable {A B : HomologicalComplex (ModuleCat.{w} k) c} (b : BinaryBicone A B)
  (hb : b.fst ≫ b.inl + b.snd ≫ b.inr = 𝟙 b.pt) (i : ι)

/-- The image of a binary bicone of complexes under homology in degree `i`. -/
@[simps]
noncomputable def homologyBicone : BinaryBicone (A.homology i) (B.homology i) where
  pt := b.pt.homology i
  fst := HomologicalComplex.homologyMap b.fst i
  snd := HomologicalComplex.homologyMap b.snd i
  inl := HomologicalComplex.homologyMap b.inl i
  inr := HomologicalComplex.homologyMap b.inr i
  inl_fst := by
    rw [← HomologicalComplex.homologyMap_comp, b.inl_fst, HomologicalComplex.homologyMap_id]
  inl_snd := by
    rw [← HomologicalComplex.homologyMap_comp, b.inl_snd, HomologicalComplex.homologyMap_zero]
  inr_fst := by
    rw [← HomologicalComplex.homologyMap_comp, b.inr_fst, HomologicalComplex.homologyMap_zero]
  inr_snd := by
    rw [← HomologicalComplex.homologyMap_comp, b.inr_snd, HomologicalComplex.homologyMap_id]

include hb in
/-- The homology maps of the projections and inclusions of a direct sum add up to the
identity. -/
theorem homologyMap_total :
    HomologicalComplex.homologyMap b.fst i ≫ HomologicalComplex.homologyMap b.inl i +
      HomologicalComplex.homologyMap b.snd i ≫ HomologicalComplex.homologyMap b.inr i =
        𝟙 (b.pt.homology i) := by
  rw [← HomologicalComplex.homologyMap_comp, ← HomologicalComplex.homologyMap_comp,
    ← HomologicalComplex.homologyMap_add, hb, HomologicalComplex.homologyMap_id]

include hb in
theorem homologyBicone_total :
    (homologyBicone b i).fst ≫ (homologyBicone b i).inl +
      (homologyBicone b i).snd ≫ (homologyBicone b i).inr = 𝟙 (b.pt.homology i) :=
  homologyMap_total b hb i

include hb in
/-- Homology preserves direct sums: `H_i(A ⊕ B) ≅ H_i(A) ⊞ H_i(B)`, for a bicone `b` of
complexes whose maps satisfy `fst ≫ inl + snd ≫ inr = 𝟙`. -/
noncomputable def homologyIsoBiprod : b.pt.homology i ≅ A.homology i ⊞ B.homology i :=
  biprod.uniqueUpToIso _ _
    (isBinaryBilimitOfTotal (homologyBicone b i) (homologyBicone_total b hb i))

include hb in
/-- Homology preserves direct sums, as a linear equivalence `H_i(A ⊕ B) ≃ H_i(A) × H_i(B)`
given by the homology maps of the projections. -/
noncomputable def homologyEquivProd : b.pt.homology i ≃ₗ[k] A.homology i × B.homology i :=
  LinearEquiv.ofLinear
    ((HomologicalComplex.homologyMap b.fst i).hom.prod (HomologicalComplex.homologyMap b.snd i).hom)
    ((HomologicalComplex.homologyMap b.inl i).hom.coprod
      (HomologicalComplex.homologyMap b.inr i).hom)
    (by
      ext x
      · simp [homologyMap_hom_comp_apply]
      · simp [homologyMap_hom_comp_apply]
      · simp [homologyMap_hom_comp_apply]
      · simp [homologyMap_hom_comp_apply])
    (by
      ext x
      have h := LinearMap.congr_fun
        (congrArg (fun f : b.pt ⟶ b.pt ↦ (HomologicalComplex.homologyMap f i).hom) hb) x
      simp only [HomologicalComplex.homologyMap_add, ModuleCat.hom_add, LinearMap.add_apply,
        HomologicalComplex.homologyMap_id, ModuleCat.hom_id, LinearMap.id_apply] at h
      simpa [homologyMap_hom_comp_apply] using h)

@[simp] theorem homologyEquivProd_apply (x : b.pt.homology i) :
    homologyEquivProd b hb i x =
      ((HomologicalComplex.homologyMap b.fst i).hom x,
        (HomologicalComplex.homologyMap b.snd i).hom x) :=
  rfl

@[simp] theorem homologyEquivProd_symm_apply (y : A.homology i × B.homology i) :
    (homologyEquivProd b hb i).symm y =
      (HomologicalComplex.homologyMap b.inl i).hom y.1 +
        (HomologicalComplex.homologyMap b.inr i).hom y.2 :=
  rfl

/-- Naturality of `homologyEquivProd`: a chain map compatible with the projections acts
componentwise on homology. -/
theorem homologyEquivProd_naturality {A' B' : HomologicalComplex (ModuleCat.{w} k) c}
    (b' : BinaryBicone A' B') (hb' : b'.fst ≫ b'.inl + b'.snd ≫ b'.inr = 𝟙 b'.pt)
    (φ : b.pt ⟶ b'.pt) (φ₁ : A ⟶ A') (φ₂ : B ⟶ B') (h₁ : b.fst ≫ φ₁ = φ ≫ b'.fst)
    (h₂ : b.snd ≫ φ₂ = φ ≫ b'.snd) (x : b.pt.homology i) :
    homologyEquivProd b' hb' i ((HomologicalComplex.homologyMap φ i).hom x) =
      Prod.map (HomologicalComplex.homologyMap φ₁ i).hom (HomologicalComplex.homologyMap φ₂ i).hom
        (homologyEquivProd b hb i x) := by
  simp [homologyMap_hom_comp_apply, h₁, h₂]

end Biproduct

/-! ### The splitting by an involution -/

section Involution

variable {k : Type u} [CommRing k] [Invertible (2 : k)] {ι : Type*} {c : ComplexShape ι}
  {C D : HomologicalComplex (ModuleCat.{w} k) c} {σ : C ⟶ C} {τ : D ⟶ D}

omit [Invertible (2 : k)] in
/-- An involution of a complex is an involution in each degree. -/
theorem apply_apply_of_comp_self (hσ : σ ≫ σ = 𝟙 C) (i : ι) (x : C.X i) :
    (σ.f i).hom ((σ.f i).hom x) = x := by
  have h := LinearMap.congr_fun
    (congrArg ModuleCat.Hom.hom (HomologicalComplex.congr_hom hσ i)) x
  simpa using h

omit [Invertible (2 : k)] in
/-- An involution of `C` acts on homology by an involution. -/
theorem homologyMap_hom_mul_self (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    (HomologicalComplex.homologyMap σ i).hom * (HomologicalComplex.homologyMap σ i).hom = 1 := by
  rw [Module.End.mul_eq_comp, ← ModuleCat.hom_comp, ← HomologicalComplex.homologyMap_comp, hσ,
    HomologicalComplex.homologyMap_id, ModuleCat.hom_id, Module.End.one_eq_id]

omit [Invertible (2 : k)] in
/-- Chain maps that commute induce commuting maps on homology. -/
theorem commute_homologyMap_hom {T : C ⟶ C} (hT : T ≫ σ = σ ≫ T) (i : ι) :
    Commute (HomologicalComplex.homologyMap T i).hom (HomologicalComplex.homologyMap σ i).hom := by
  rw [Commute, SemiconjBy, Module.End.mul_eq_comp, Module.End.mul_eq_comp, ← ModuleCat.hom_comp,
    ← ModuleCat.hom_comp, ← HomologicalComplex.homologyMap_comp,
    ← HomologicalComplex.homologyMap_comp, hT]

omit [Invertible (2 : k)] in
@[reassoc]
theorem plusι_comp : (subcomplexι : plusSubcomplex σ ⟶ C) ≫ σ = subcomplexι := by
  ext i x
  exact (mem_ker_sub_id σ).mp x.2

omit [Invertible (2 : k)] in
@[reassoc]
theorem minusι_comp : (subcomplexι : minusSubcomplex σ ⟶ C) ≫ σ = -subcomplexι := by
  ext i x
  exact (mem_ker_add_id σ).mp x.2

variable (σ) in
/-- The projection `x ↦ (x + σ x) / 2` onto the `+1` eigen-subcomplex. -/
noncomputable def plusProj (hσ : σ ≫ σ = 𝟙 C) : C ⟶ plusSubcomplex σ :=
  subcomplexLift (⅟(2 : k) • (𝟙 C + σ)) fun i x ↦ (mem_ker_sub_id σ).mpr (by
    simp [apply_apply_of_comp_self hσ, add_comm])

variable (σ) in
/-- The projection `x ↦ (x - σ x) / 2` onto the `-1` eigen-subcomplex. -/
noncomputable def minusProj (hσ : σ ≫ σ = 𝟙 C) : C ⟶ minusSubcomplex σ :=
  subcomplexLift (⅟(2 : k) • (𝟙 C - σ)) fun i x ↦ (mem_ker_add_id σ).mpr (by
    simp [apply_apply_of_comp_self hσ, smul_sub])

@[reassoc (attr := simp)]
theorem plusProj_ι (hσ : σ ≫ σ = 𝟙 C) :
    plusProj σ hσ ≫ subcomplexι = ⅟(2 : k) • (𝟙 C + σ) :=
  subcomplexLift_ι _ _

@[reassoc (attr := simp)]
theorem minusProj_ι (hσ : σ ≫ σ = 𝟙 C) :
    minusProj σ hσ ≫ subcomplexι = ⅟(2 : k) • (𝟙 C - σ) :=
  subcomplexLift_ι _ _

/-- `(1 + σ) / 2 ∘ σ = (1 + σ) / 2`. -/
theorem comp_plusProj (hσ : σ ≫ σ = 𝟙 C) : σ ≫ plusProj σ hσ = plusProj σ hσ := by
  apply subcomplex_hom_ext
  rw [Category.assoc, plusProj_ι, Linear.comp_smul, Preadditive.comp_add, Category.comp_id, hσ,
    add_comm]

/-- `(1 - σ) / 2 ∘ σ = -(1 - σ) / 2`. -/
theorem comp_minusProj (hσ : σ ≫ σ = 𝟙 C) : σ ≫ minusProj σ hσ = -minusProj σ hσ := by
  apply subcomplex_hom_ext
  rw [Category.assoc, minusProj_ι, Preadditive.neg_comp, minusProj_ι, Linear.comp_smul,
    Preadditive.comp_sub, Category.comp_id, hσ, ← smul_neg, neg_sub]

/-- The splitting `C = C⁺ ⊕ C⁻` of a complex with an involution `σ`, `2` invertible, into the `+1`
and `-1` eigen-subcomplexes, as a binary bicone with projections `(1 ± σ) / 2`. This is the
chain-level splitting in the proof of Proposition `prop:z2_reconstruction`. -/
@[simps]
noncomputable abbrev eigenBicone (hσ : σ ≫ σ = 𝟙 C) :
    BinaryBicone (plusSubcomplex σ) (minusSubcomplex σ) where
  pt := C
  fst := plusProj σ hσ
  snd := minusProj σ hσ
  inl := subcomplexι
  inr := subcomplexι
  inl_fst := by
    apply subcomplex_hom_ext
    rw [Category.assoc, plusProj_ι, Linear.comp_smul, Preadditive.comp_add, Category.comp_id,
      plusι_comp, invOf_two_smul_add_self, Category.id_comp]
  inl_snd := by
    apply subcomplex_hom_ext
    rw [Category.assoc, minusProj_ι, Linear.comp_smul, Preadditive.comp_sub, Category.comp_id,
      plusι_comp, sub_self, smul_zero, zero_comp]
  inr_fst := by
    apply subcomplex_hom_ext
    rw [Category.assoc, plusProj_ι, Linear.comp_smul, Preadditive.comp_add, Category.comp_id,
      minusι_comp, add_neg_cancel, smul_zero, zero_comp]
  inr_snd := by
    apply subcomplex_hom_ext
    rw [Category.assoc, minusProj_ι, Linear.comp_smul, Preadditive.comp_sub, Category.comp_id,
      minusι_comp, sub_neg_eq_add, invOf_two_smul_add_self, Category.id_comp]

/-- `(1 + σ) / 2 + (1 - σ) / 2 = 1`. -/
theorem eigenBicone_total (hσ : σ ≫ σ = 𝟙 C) :
    (eigenBicone hσ).fst ≫ (eigenBicone hσ).inl + (eigenBicone hσ).snd ≫ (eigenBicone hσ).inr =
      𝟙 (eigenBicone hσ).pt := by
  show plusProj σ hσ ≫ subcomplexι + minusProj σ hσ ≫ subcomplexι = 𝟙 C
  rw [plusProj_ι, minusProj_ι, ← smul_add, add_add_sub_cancel, invOf_two_smul_add_self]

/-- The splitting `C ≅ C⁺ ⊞ C⁻` of a complex with an involution, `2` invertible, as an isomorphism
of complexes (proof of Proposition `prop:z2_reconstruction`). -/
noncomputable def isoBiprodEigen (hσ : σ ≫ σ = 𝟙 C) :
    C ≅ plusSubcomplex σ ⊞ minusSubcomplex σ :=
  biprod.uniqueUpToIso _ _ (isBinaryBilimitOfTotal (eigenBicone hσ) (eigenBicone_total hσ))

/-- The splitting step of Proposition `prop:z2_reconstruction`, at the homology level: for an
involution `σ` of a complex `C` of `k`-modules, `2` invertible, `H_i(C) ≃ H_i(C⁺) × H_i(C⁻)`, the
homologies of the `+1` and `-1` eigen-subcomplexes. -/
noncomputable def homologyEquivEigen (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    C.homology i ≃ₗ[k] (plusSubcomplex σ).homology i × (minusSubcomplex σ).homology i :=
  homologyEquivProd (eigenBicone hσ) (eigenBicone_total hσ) i

/-- The splitting of Proposition `prop:z2_reconstruction` as an isomorphism with the biproduct
`H_i(C⁺) ⊞ H_i(C⁻)` in `ModuleCat k`. -/
noncomputable def homologyIsoBiprodEigen (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    C.homology i ≅ (plusSubcomplex σ).homology i ⊞ (minusSubcomplex σ).homology i :=
  homologyIsoBiprod (eigenBicone hσ) (eigenBicone_total hσ) i

/-- Every class is the sum of its components in `H_i(C⁺)` and `H_i(C⁻)`. -/
theorem homologyMap_plusProj_add_minusProj (hσ : σ ≫ σ = 𝟙 C) (i : ι) (y : C.homology i) :
    (HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom
        ((HomologicalComplex.homologyMap (plusProj σ hσ) i).hom y) +
      (HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom
        ((HomologicalComplex.homologyMap (minusProj σ hσ) i).hom y) = y :=
  (homologyEquivEigen hσ i).symm_apply_apply y

omit [Invertible (2 : k)] in
/-- A chain map with `φ ∘ σ = τ ∘ φ` maps the `+1` eigen-subcomplex of `σ` to that of `τ`. -/
def plusMap (φ : C ⟶ D) (hφ : φ ≫ τ = σ ≫ φ) : plusSubcomplex σ ⟶ plusSubcomplex τ :=
  subcomplexMap φ fun i x hx ↦ (mem_ker_sub_id τ).mpr (by
    have h := LinearMap.congr_fun
      (congrArg ModuleCat.Hom.hom (HomologicalComplex.congr_hom hφ i)) x
    simp only [HomologicalComplex.comp_f, ModuleCat.hom_comp, LinearMap.comp_apply] at h
    rw [h, (mem_ker_sub_id σ).mp hx])

omit [Invertible (2 : k)] in
/-- A chain map with `φ ∘ σ = τ ∘ φ` maps the `-1` eigen-subcomplex of `σ` to that of `τ`. -/
def minusMap (φ : C ⟶ D) (hφ : φ ≫ τ = σ ≫ φ) : minusSubcomplex σ ⟶ minusSubcomplex τ :=
  subcomplexMap φ fun i x hx ↦ (mem_ker_add_id τ).mpr (by
    have h := LinearMap.congr_fun
      (congrArg ModuleCat.Hom.hom (HomologicalComplex.congr_hom hφ i)) x
    simp only [HomologicalComplex.comp_f, ModuleCat.hom_comp, LinearMap.comp_apply] at h
    rw [h, (mem_ker_add_id σ).mp hx, map_neg])

omit [Invertible (2 : k)] in
@[reassoc (attr := simp)]
theorem plusMap_ι (φ : C ⟶ D) (hφ : φ ≫ τ = σ ≫ φ) :
    plusMap φ hφ ≫ subcomplexι = subcomplexι ≫ φ :=
  subcomplexMap_ι _ _

omit [Invertible (2 : k)] in
@[reassoc (attr := simp)]
theorem minusMap_ι (φ : C ⟶ D) (hφ : φ ≫ τ = σ ≫ φ) :
    minusMap φ hφ ≫ subcomplexι = subcomplexι ≫ φ :=
  subcomplexMap_ι _ _

/-- The projections onto the `+1` eigen-subcomplexes are natural. -/
theorem plusProj_plusMap (hσ : σ ≫ σ = 𝟙 C) (hτ : τ ≫ τ = 𝟙 D) (φ : C ⟶ D)
    (hφ : φ ≫ τ = σ ≫ φ) : plusProj σ hσ ≫ plusMap φ hφ = φ ≫ plusProj τ hτ := by
  apply subcomplex_hom_ext
  rw [Category.assoc, plusMap_ι, plusProj_ι_assoc, Category.assoc, plusProj_ι, Linear.smul_comp,
    Linear.comp_smul, Preadditive.add_comp, Preadditive.comp_add, Category.id_comp,
    Category.comp_id, hφ]

/-- The projections onto the `-1` eigen-subcomplexes are natural. -/
theorem minusProj_minusMap (hσ : σ ≫ σ = 𝟙 C) (hτ : τ ≫ τ = 𝟙 D) (φ : C ⟶ D)
    (hφ : φ ≫ τ = σ ≫ φ) : minusProj σ hσ ≫ minusMap φ hφ = φ ≫ minusProj τ hτ := by
  apply subcomplex_hom_ext
  rw [Category.assoc, minusMap_ι, minusProj_ι_assoc, Category.assoc, minusProj_ι,
    Linear.smul_comp, Linear.comp_smul, Preadditive.sub_comp, Preadditive.comp_sub,
    Category.id_comp, Category.comp_id, hφ]

/-- A chain map commuting with the involutions preserves the splitting of Proposition
`prop:z2_reconstruction`: on `H_i(C⁺) × H_i(C⁻)` it acts by the homology maps of its
restrictions to the eigen-subcomplexes. -/
theorem homologyEquivEigen_naturality (hσ : σ ≫ σ = 𝟙 C) (hτ : τ ≫ τ = 𝟙 D) (φ : C ⟶ D)
    (hφ : φ ≫ τ = σ ≫ φ) (i : ι) (x : C.homology i) :
    homologyEquivEigen hτ i ((HomologicalComplex.homologyMap φ i).hom x) =
      Prod.map (HomologicalComplex.homologyMap (plusMap φ hφ) i).hom
        (HomologicalComplex.homologyMap (minusMap φ hφ) i).hom (homologyEquivEigen hσ i x) :=
  homologyEquivProd_naturality (eigenBicone hσ) (eigenBicone_total hσ) i (eigenBicone hτ)
    (eigenBicone_total hτ) φ (plusMap φ hφ) (minusMap φ hφ) (plusProj_plusMap hσ hτ φ hφ)
    (minusProj_minusMap hσ hτ φ hφ) x

/-- The summand `H_i(C⁺)` is the `+1` eigenspace of `σ` on `H_i(C)` (Proposition
`prop:z2_reconstruction`): the image of `H_i(C⁺) → H_i(C)` is `ker (σ_* - 1)`. -/
theorem range_homologyMap_plusι (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    LinearMap.range (HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom =
      LinearMap.ker ((HomologicalComplex.homologyMap σ i).hom - 1) := by
  apply le_antisymm
  · rintro _ ⟨y, rfl⟩
    rw [LinearMap.mem_ker, LinearMap.sub_apply, Module.End.one_apply, sub_eq_zero,
      homologyMap_hom_comp_apply, plusι_comp]
  · intro y hy
    rw [LinearMap.mem_ker, LinearMap.sub_apply, Module.End.one_apply, sub_eq_zero] at hy
    have h0 : (HomologicalComplex.homologyMap (minusProj σ hσ) i).hom y = 0 := by
      apply eq_zero_of_eq_neg_of_invertible k
      have h := homologyMap_hom_comp_apply σ (minusProj σ hσ) i y
      rw [hy, comp_minusProj, HomologicalComplex.homologyMap_neg, ModuleCat.hom_neg,
        LinearMap.neg_apply] at h
      exact h
    refine ⟨(HomologicalComplex.homologyMap (plusProj σ hσ) i).hom y, ?_⟩
    have h := homologyMap_plusProj_add_minusProj hσ i y
    rwa [h0, map_zero, add_zero] at h

/-- The summand `H_i(C⁻)` is the `-1` eigenspace of `σ` on `H_i(C)` (Proposition
`prop:z2_reconstruction`): the image of `H_i(C⁻) → H_i(C)` is `ker (σ_* + 1)`. -/
theorem range_homologyMap_minusι (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    LinearMap.range (HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom =
      LinearMap.ker ((HomologicalComplex.homologyMap σ i).hom + 1) := by
  apply le_antisymm
  · rintro _ ⟨y, rfl⟩
    rw [LinearMap.mem_ker, LinearMap.add_apply, Module.End.one_apply, homologyMap_hom_comp_apply,
      minusι_comp, HomologicalComplex.homologyMap_neg, ModuleCat.hom_neg, LinearMap.neg_apply,
      neg_add_cancel]
  · intro y hy
    rw [LinearMap.mem_ker, LinearMap.add_apply, Module.End.one_apply, add_eq_zero_iff_eq_neg]
      at hy
    have h0 : (HomologicalComplex.homologyMap (plusProj σ hσ) i).hom y = 0 := by
      apply eq_zero_of_eq_neg_of_invertible k
      have h := homologyMap_hom_comp_apply σ (plusProj σ hσ) i y
      rw [hy, comp_plusProj, map_neg] at h
      exact h.symm
    refine ⟨(HomologicalComplex.homologyMap (minusProj σ hσ) i).hom y, ?_⟩
    have h := homologyMap_plusProj_add_minusProj hσ i y
    rwa [h0, map_zero, zero_add] at h

/-- The summands `H_i(C⁺)` and `H_i(C⁻)` are complementary in `H_i(C)`: they are the `+1` and
`-1` eigenspaces of `σ` on homology (`isCompl_ker_sub_one_ker_add_one`). -/
theorem isCompl_range_homologyMap (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    IsCompl
      (LinearMap.range (HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom)
      (LinearMap.range
        (HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom) := by
  rw [range_homologyMap_plusι hσ, range_homologyMap_minusι hσ]
  exact isCompl_ker_sub_one_ker_add_one (homologyMap_hom_mul_self hσ i)

/-- The inclusion of the `+1` eigen-subcomplex is injective on homology. -/
theorem injective_homologyMap_plusι (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    Function.Injective
      (HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom := by
  intro y₁ y₂ h
  have h' := congrArg (HomologicalComplex.homologyMap (plusProj σ hσ) i).hom h
  rwa [homologyMap_hom_comp_apply, homologyMap_hom_comp_apply, ← eigenBicone_inl hσ,
    ← eigenBicone_fst hσ, (eigenBicone hσ).inl_fst, HomologicalComplex.homologyMap_id] at h'

/-- The inclusion of the `-1` eigen-subcomplex is injective on homology. -/
theorem injective_homologyMap_minusι (hσ : σ ≫ σ = 𝟙 C) (i : ι) :
    Function.Injective
      (HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom := by
  intro y₁ y₂ h
  have h' := congrArg (HomologicalComplex.homologyMap (minusProj σ hσ) i).hom h
  rwa [homologyMap_hom_comp_apply, homologyMap_hom_comp_apply, ← eigenBicone_inr hσ,
    ← eigenBicone_snd hσ, (eigenBicone hσ).inr_snd, HomologicalComplex.homologyMap_id] at h'

end Involution

/-! ### Nilpotency of the induced map -/

section Nilpotent

variable {k : Type u} [CommRing k] [Invertible (2 : k)] {ι : Type*} {c : ComplexShape ι}
  {C : HomologicalComplex (ModuleCat.{w} k) c} {σ : C ⟶ C}

/-- The index map is trivial on both summands of Proposition `prop:z2_reconstruction` exactly
when it is trivial (over a field the index is trivial if and only if the
index map is nilpotent). For a chain endomorphism `T` commuting with an involution `σ`, `2`
invertible, `T_*` is nilpotent on `H_i(C)` if and only if its restrictions are nilpotent on
`H_i(C⁺)` and on `H_i(C⁻)`. -/
theorem isNilpotent_homologyMap_iff (hσ : σ ≫ σ = 𝟙 C) (T : C ⟶ C) (hT : T ≫ σ = σ ≫ T)
    (i : ι) :
    IsNilpotent (HomologicalComplex.homologyMap T i).hom ↔
      IsNilpotent (HomologicalComplex.homologyMap (plusMap T hT) i).hom ∧
        IsNilpotent (HomologicalComplex.homologyMap (minusMap T hT) i).hom := by
  rw [isNilpotent_iff_of_linearEquiv (homologyEquivEigen hσ i)
    (g := (HomologicalComplex.homologyMap (plusMap T hT) i).hom.prodMap
      (HomologicalComplex.homologyMap (minusMap T hT) i).hom)
    (homologyEquivEigen_naturality hσ hσ T hT i), isNilpotent_prodMap_iff]

/-- An endomorphism `T` of `H_i(C)` commuting with the action of an involution `σ`, `2`
invertible, is nilpotent if and only if it is nilpotent on the `+1` and `-1` eigenspaces of `σ`
on `H_i(C)`, which are the summands `H_i(C⁺)` and `H_i(C⁻)` (`range_homologyMap_plusι`). This
applies to an index map defined on homology only, and is `isNilpotent_iff_eigenspaces` for
the action of `σ` on homology (Proposition `prop:z2_reconstruction`). -/
theorem isNilpotent_iff_homology_eigenspaces (hσ : σ ≫ σ = 𝟙 C) (i : ι)
    {T : Module.End k (C.homology i)} (hT : Commute T (HomologicalComplex.homologyMap σ i).hom) :
    IsNilpotent T ↔
      IsNilpotent (T.restrict (mapsTo_ker_of_commute (hT.sub_right (Commute.one_right T)))) ∧
        IsNilpotent (T.restrict (mapsTo_ker_of_commute (hT.add_right (Commute.one_right T)))) :=
  isNilpotent_iff_eigenspaces (homologyMap_hom_mul_self hσ i) hT

end Nilpotent

/-! ### Involutions of cell complexes -/

namespace CellComplex

variable {k : Type u} [CommRing k] {K : CellComplex.{u, v} k}

namespace ChainMap

/-- An involution of the chains that commutes with the boundaries is an involution of the chain
complex. -/
theorem toHom_comp_self {σ : K.ChainMap K} (hσ : σ.comp σ = ChainMap.id K) :
    σ.toHom ≫ σ.toHom = 𝟙 K.toChainComplex := by
  rw [← toHom_comp, hσ, toHom_id]

/-- Commuting chain maps give commuting morphisms of chain complexes. -/
theorem toHom_comm {σ T : K.ChainMap K} (h : T.comp σ = σ.comp T) :
    T.toHom ≫ σ.toHom = σ.toHom ≫ T.toHom := by
  rw [← toHom_comp, ← toHom_comp, h]

/-- The chain map `c ↦ c ∘ s` of an involution `s` of the cells that preserves the incidence
numbers, `[s e : s e'] = [e : e']`. This is the action of `g` on the cellular chains in
Proposition `prop:z2_reconstruction`, where `g` permutes the cells without signs (the grid
condition), as in `SignChains`. -/
def ofInvolutive (s : ∀ n, K.cells n → K.cells n) (hs : ∀ n, Function.Involutive (s n))
    (hinc : ∀ n e e', K.inc n (s (n + 1) e) (s n e') = K.inc n e e') : K.ChainMap K where
  app n := LinearMap.funLeft k k (s n)
  comm n := by
    refine LinearMap.ext fun c ↦ funext fun j ↦ ?_
    simp only [LinearMap.comp_apply, LinearMap.funLeft_apply, bd_apply]
    conv_rhs => rw [← Equiv.sum_comp ((hs (n + 1)).toPerm (s (n + 1)))]
    simp [hinc]

variable {s : ∀ n, K.cells n → K.cells n} {hs : ∀ n, Function.Involutive (s n)}
  {hinc : ∀ n e e', K.inc n (s (n + 1) e) (s n e') = K.inc n e e'}

@[simp] theorem ofInvolutive_app_apply (n : ℕ) (c : K.cells n → k) (e : K.cells n) :
    (ofInvolutive s hs hinc).app n c e = c (s n e) :=
  rfl

/-- The chain map of an involution of the cells is an involution. -/
theorem ofInvolutive_comp_self :
    (ofInvolutive s hs hinc).comp (ofInvolutive s hs hinc) = ChainMap.id K :=
  ChainMap.ext (funext fun n ↦ LinearMap.ext fun c ↦ funext fun e ↦ by
    simp [(hs n) e])

/-- In degree `n`, the `+1` eigen-subcomplex of the action of an involution of the cells consists
of the chains with `c (s e) = c e` (`symChains` of `SignChains`). -/
theorem mem_plus_ofInvolutive_iff {n : ℕ} {c : K.cells n → k} :
    c ∈ LinearMap.ker (((ofInvolutive s hs hinc).toHom - 𝟙 K.toChainComplex).f n).hom ↔
      ∀ e, c (s n e) = c e := by
  rw [mem_ker_sub_id, funext_iff]
  rfl

/-- In degree `n`, the `-1` eigen-subcomplex of the action of an involution of the cells consists
of the chains with `c (s e) = -c e` (`antiChains` of `SignChains`). -/
theorem mem_minus_ofInvolutive_iff {n : ℕ} {c : K.cells n → k} :
    c ∈ LinearMap.ker (((ofInvolutive s hs hinc).toHom + 𝟙 K.toChainComplex).f n).hom ↔
      ∀ e, c (s n e) = -c e := by
  rw [mem_ker_add_id, funext_iff]
  rfl

end ChainMap

variable [Invertible (2 : k)]

/-- Proposition `prop:z2_reconstruction` for a cell complex: an involution `σ` of the chains of
`K` that commutes with the boundaries, `2` invertible, splits `H_n(K)` into the homologies of the
`+1` and `-1` eigen-subcomplexes. -/
noncomputable def homologyEquivEigen {σ : K.ChainMap K} (hσ : σ.comp σ = ChainMap.id K)
    (n : ℕ) : K.toChainComplex.homology n ≃ₗ[k]
      (plusSubcomplex σ.toHom).homology n × (minusSubcomplex σ.toHom).homology n :=
  LatentEquivariant.homologyEquivEigen (ChainMap.toHom_comp_self hσ) n

/-- For a cell complex with an involution `σ` of the chains, `2` invertible, a chain endomorphism
`T` commuting with `σ` is nilpotent on `H_n(K)` if and only if it is nilpotent on the homologies of
both eigen-subcomplexes (Proposition `prop:z2_reconstruction`). -/
theorem isNilpotent_homologyMap_iff {σ : K.ChainMap K} (hσ : σ.comp σ = ChainMap.id K)
    {T : K.ChainMap K} (hT : T.comp σ = σ.comp T) (n : ℕ) :
    IsNilpotent (T.homologyMap n) ↔
      IsNilpotent
          (HomologicalComplex.homologyMap (plusMap T.toHom (ChainMap.toHom_comm hT)) n).hom ∧
        IsNilpotent
          (HomologicalComplex.homologyMap (minusMap T.toHom (ChainMap.toHom_comm hT)) n).hom :=
  LatentEquivariant.isNilpotent_homologyMap_iff (ChainMap.toHom_comp_self hσ) T.toHom
    (ChainMap.toHom_comm hT) n

end CellComplex

end LatentEquivariant
