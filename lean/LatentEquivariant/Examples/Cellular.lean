import LatentEquivariant.Homology.Basic
import LatentEquivariant.Algebra.OrbitChains
import LatentEquivariant.Algebra.SignChains

/-!
# Cellular examples: a mirror on an interval and `-I` on a square

Finite cell complexes, written down as `CellComplex` values with cells `Fin m`, and their
homology computed explicitly. `doubling` and `square` are relative cellular chain complexes
`C_*(N, L)` of `G`-invariant pairs on grids, for `G = {1, g}` of order two, with the cells
oriented so that `g` permutes them without signs (the model of `OrbitChains` and `SignChains`).
`doublingOrbit`, `twisted` and `untwisted` are orbit-space complexes. Example (c) is the one after
Proposition `prop:z2_reconstruction`. Ranks are stated as
linear equivalences: `H = 0` is `Subsingleton H`, and rank one is `H ≃ₗ[k] k`. The homology of a
cell complex is `CellComplex.H` (`ker ∂ ⧸ range ∂`, which `CellComplex.homologyEquiv` identifies
with Mathlib's homology); the eigen-subcomplexes `plusSubcomplex` and `minusSubcomplex` of
`Homology/Basic` are complexes of modules, with Mathlib's homology. The coefficient ring `k` is
any commutative ring unless a hypothesis on `2` is stated.

## (a) The doubling map on a mirror

A repeller on a mirror, as a map: `f(x) = 2x`, `g x = -x`, the index pair `N = [-1, 1]`,
`L = [-1, -1/2] ∪ [1/2, 1]` of `LatentEquivariant.Examples.Doubling`, on the grid with vertices
at the multiples of `1/2`. The cells of `N` not in `L` are the vertex `v = 0` and the edges
`e₋ = [-1/2, 0]`, oriented from `-1/2` to `0`, and `e₊ = [0, 1/2]`, oriented from `1/2` to `0`,
so `[e₋ : v] = [e₊ : v] = 1`. The mirror swaps `e₋` and `e₊` and fixes `v`.

* `doubling`, `doublingInvolution`: the complex and the action of `g` on chains.
* `doubling_ranks`, `doublingH1Equiv`, `doubling_H_one_eq_smul`: `H_0(N, L) = 0` and
  `H_1(N, L) ≅ k`, spanned by the class of `e₋ - e₊`, the interval `[-1/2, 1/2]`
  (`doublingH1Equiv_cycle`).
* `doublingInvolution_mapH`, `doublingInvolution_homologyMap`: `g` acts by `-1` on homology.
* `doubling_plus_acyclic`: if `2` is invertible, the `+1` eigen-subcomplex (the trivial
  component, which is the orbit-space term) is acyclic, and `doublingMinusEquiv`: the `-1`
  eigen-subcomplex has the homology of `(N, L)`.
* `doublingMinusH1Equiv`, `doublingMinusH0Equiv`, `subsingleton_doubling_minus_H_zero`: over any
  commutative ring, the `-1` eigen-subcomplex has `H_1 ≅ k`, and `H_0` is the `2`-torsion of `k`,
  which is `0` if `2` is not a zero divisor. With `2` invertible (or, for this example, not a zero
  divisor) the sign component carries the index; otherwise `C⁻` is not a summand of `C`, and over
  `ZMod 2` it equals `C⁺`.

## (b) The same complex over `ℤ`

* `doublingPlusH0Equiv`: `H_0` of the invariant chains is `k ⧸ 2k`, since the boundary of the
  invariant chain `e₋ + e₊` is `2v` (`doubling_bd_of_mem_plus`). So the `+1` eigen-subcomplex is
  acyclic only if `2` is a unit (`subsingleton_doubling_plus_H_zero_iff`), and over `ℤ` its `H_0`
  is `ZMod 2` (`doublingPlusH0EquivInt`, `nontrivial_doubling_plus_H_zero`).
* `doublingOrbit`, `doublingOrbit_acyclic`: the orbit complex, on the cells of
  `(N/G, L/G) = ([0, 1], [1/2, 1])`, one edge and one vertex with incidence `1`, is acyclic.
  Evaluation at the representatives, `e₋ + e₊ ↦ [e]` and `v ↦ [v]`, is not a chain map to it
  (`doubling_bd_ne_doublingOrbit_bd`); the identification in the proof of Proposition
  `prop:z2_reconstruction` is `π_#`, `e₋ + e₊ ↦ 2[e]`.
* `isEmpty_plus_homology_equiv_orbit`, `isEmpty_plus_iso_orbit`: over `ℤ` there is no linear
  equivalence between the `H_0` of the invariant chains and that of the orbit complex, so no
  isomorphism of complexes either: with `2` not invertible, the identification of the invariant
  chains with the orbit complex fails on homology, not only on chains (compare
  `not_exists_chainEquiv_reflection` of `OrbitChains`, a statement about chains over `ZMod 2`).
  `isEmpty_plus_homology_equiv_orbit_iff` and `isEmpty_plus_iso_orbit_of_not_isUnit` are the
  same over any ring in which `2` is not a unit.
* `doubling_splitting_fails`: if `2` is not a unit, `H_0(N, L) = 0` is not `H_0(C⁺) × M` for any
  `M`, so the splitting `H_*(C) ≅ H_*(C⁺) × H_*(C⁻)` of `homologyEquivEigen`, the chain-level step
  of the proof of Proposition `prop:z2_reconstruction`, fails. The proposition's formula with
  orbit-space terms still holds for this example over every ring (`doubling_formula`): the twisted
  term is `k` in degree `1`, one free pair of edges with the fixed vertex dropped. So this example
  does not show that `char ≠ 2` is needed for the formula; the square does
  (`twisted_formula_fails`).

## (c) `-I` on a square

The example after Proposition `prop:z2_reconstruction`:
`N = [-1, 1]²`, `L = ∂N`, the grid of unit squares, `G = {±I}`. The cells of `N` not in `L` are
the origin `O`, the edges `a = O → (1, 0)`, `b = O → (0, 1)`, `a' = O → (-1, 0)`,
`b' = O → (0, -1)`, and the squares `Q1 = [0, 1]²`, `Q2 = [-1, 0] × [0, 1]`, `Q3 = [-1, 0]²`,
`Q4 = [0, 1] × [-1, 0]` with the standard orientation. Then `∂Q1 = a - b`, `∂Q2 = b - a'`,
`∂Q3 = a' - b'`, `∂Q4 = b' - a` (`squareInc_eq`), and `∂a = ∂b = ∂a' = ∂b' = -O`. The half-turn
`-I` swaps `Q1 ↔ Q3`, `Q2 ↔ Q4`, `a ↔ a'`, `b ↔ b'` and fixes `O`.

* `square`, `squareInvolution`: the complex (`∂ ∘ ∂ = 0` is its field `bd_bd`) and the action of
  `-I` on chains, which preserves the incidences (`squareCellMap_inc`).
* `square_ranks`: `H_*(N, L)` has ranks `(0, 0, 1)`, and `H_2` is spanned by the class of
  `Q1 + Q2 + Q3 + Q4` (`squareH2Equiv_cycle`, `square_H_two_eq_smul`).
* `squareInvolution_mapH`: `-I` acts by `+1` on homology. So, if `2` is invertible, the `-1`
  eigen-subcomplex is acyclic (`square_minus_acyclic`) and the `+1` eigen-subcomplex has the
  homology of `(N, L)` (`squarePlusEquiv`, `squarePlusH2Equiv`).
* `twisted`, `twisted_acyclic`: the twisted complex on the representatives `{a, b}` and
  `{Q1, Q2}` (`isRepresentatives_squareRep`), with incidences `twInc` and the fixed vertex `O`
  dropped, is acyclic if `2` is invertible. Its `H_1` is `k ⧸ 2k` (`twistedH1Equiv`), so it is
  acyclic only if `2` is a unit (`subsingleton_twisted_H_one_iff`); over `ℤ` its `H_1` is
  `ZMod 2` (`twistedH1EquivInt`).
* `untwisted`, `untwisted_ranks`: the orbit-space complex relative to the fixed cell, with cells
  the free orbits and incidences `[r : q] + [r : g q]`, has homology of ranks `(0, 1, 1)`,
  spanned by the classes of `[a]` and `[Q1] + [Q2]` (`untwistedH1Equiv_cycle`,
  `untwistedH2Equiv_cycle`).
* `untwisted_formula_fails`: over a nontrivial ring, `H_1(N, L) = 0` is not `M × H_1` of the
  untwisted complex for any module `M`, so the formula of Proposition `prop:z2_reconstruction`
  with `F^w` replaced by `F` fails here, as the paper says after the proof.
* `twisted_formula_fails`: if `2` is not a unit, `H_1(N, L) = 0` is not `M × H_1` of the twisted
  complex for any `M`, since the latter is `k ⧸ 2k`. So the formula of Proposition
  `prop:z2_reconstruction` itself needs `2` invertible; over `ℤ` it fails for the square.

## Not formalized

* The grids. The relative complexes are written down directly. That they are the cells of `N`
  not in `L` of the grids, with the grid incidences and orientations chosen along orbits, is
  checked by hand above, not derived with `CellComplex.relative`.
* Orbit spaces. `doublingOrbit` and `untwisted` are given by the orbit-space incidences of
  `OrbitChains` and `SignChains`, not computed from cells of `N/G`. The identification of the
  `-1` eigen-subcomplex of the square with `twisted` (`antiChainsEquiv_comp_bd`) is not applied
  here; both are computed directly.
* Index maps, and the comparison of cellular with singular homology.
-/

namespace LatentEquivariant.Cellular

open CategoryTheory Limits

universe u v w

/-! ### Computing homology -/

section Helpers

variable {k : Type u} [CommRing k] {K : CellComplex.{u, v} k} {n : ℕ}

/-- `H_n(K) = 0` when every `n`-cycle is a boundary. -/
theorem subsingleton_H_of_le (h : K.cycles n ≤ K.boundaries n) : Subsingleton (K.H n) := by
  rw [Submodule.Quotient.subsingleton_iff, eq_top_iff]
  intro z _
  exact h z.2

/-- `H_n(K) = 0` when `K` has no `n`-cells. -/
theorem subsingleton_H_of_isEmpty (h : IsEmpty (K.cells n)) : Subsingleton (K.H n) := by
  infer_instance

/-- There are no `n`-boundaries when `K` has no `(n + 1)`-cells. -/
theorem boundaries_eq_bot_of_isEmpty (h : IsEmpty (K.cells (n + 1))) : K.boundaries n = ⊥ := by
  rw [CellComplex.boundaries, LinearMap.range_eq_bot]
  refine LinearMap.ext fun c ↦ ?_
  rw [Subsingleton.elim c 0, map_zero]
  rfl

/-- `H_n(K) ≃ M` from a surjective linear map `φ` on the `n`-cycles whose kernel is the
`n`-boundaries: the class of a cycle `z` goes to `φ z` (`hEquivOfSurjective_mk`). -/
noncomputable def hEquivOfSurjective {M : Type*} [AddCommGroup M] [Module k M]
    (φ : K.cycles n →ₗ[k] M) (hφ : Function.Surjective φ)
    (hker : ∀ z : K.cycles n, φ z = 0 ↔ (z : K.cells n → k) ∈ K.boundaries n) :
    K.H n ≃ₗ[k] M :=
  (Submodule.quotEquivOfEq _ _ (by ext z; exact (hker z).symm)).trans
    (φ.quotKerEquivOfSurjective hφ)

/-- `hEquivOfSurjective` maps the class of a cycle `z` to `φ z`. -/
@[simp] theorem hEquivOfSurjective_mk {M : Type*} [AddCommGroup M] [Module k M]
    (φ : K.cycles n →ₗ[k] M) (hφ : Function.Surjective φ)
    (hker : ∀ z : K.cycles n, φ z = 0 ↔ (z : K.cells n → k) ∈ K.boundaries n)
    (z : K.cycles n) : hEquivOfSurjective φ hφ hker (Submodule.Quotient.mk z) = φ z :=
  rfl

/-- Mathlib's homology `K.toChainComplex.homology n` vanishes when `ker ∂ ⧸ range ∂` does. -/
theorem subsingleton_homology (h : Subsingleton (K.H n)) :
    Subsingleton (K.toChainComplex.homology n) :=
  (K.homologyEquiv n).toEquiv.subsingleton

/-- A chain endomorphism acting by `-1` on `ker ∂ ⧸ range ∂` acts by `-1` on Mathlib's
homology. -/
theorem homologyMap_eq_neg {σ : K.ChainMap K} (h : ∀ x, σ.mapH n x = -x)
    (x : K.toChainComplex.homology n) : σ.homologyMap n x = -x := by
  apply (K.homologyEquiv n).injective
  rw [CellComplex.ChainMap.homologyEquiv_naturality, h, map_neg]

/-- A chain endomorphism acting by `1` on `ker ∂ ⧸ range ∂` acts by `1` on Mathlib's
homology. -/
theorem homologyMap_eq_self {σ : K.ChainMap K} (h : ∀ x, σ.mapH n x = x)
    (x : K.toChainComplex.homology n) : σ.homologyMap n x = x := by
  apply (K.homologyEquiv n).injective
  rw [CellComplex.ChainMap.homologyEquiv_naturality, h]

end Helpers

section Degree0

variable {k : Type u} [CommRing k]

/-- The homology in degree `0` of a chain complex of modules indexed by `ℕ` is `M`, given a
surjective linear map `C_0 → M` whose kernel is the image of `d : C_1 → C_0`. Used for the
invariant chains of the doubling example in part (b). -/
noncomputable def homologyZeroEquiv (C : ChainComplex (ModuleCat.{w} k) ℕ) {M : Type*}
    [AddCommGroup M] [Module k M] (φ : C.X 0 →ₗ[k] M) (hφ : Function.Surjective φ)
    (hker : ∀ x, φ x = 0 ↔ x ∈ LinearMap.range (C.d 1 0).hom) : C.homology 0 ≃ₗ[k] M :=
  (moduleCatHomologyEquiv (C.sc 0) (Z := ⊤) (B := LinearMap.range (C.d 1 0).hom)
      (by
        change LinearMap.ker (C.d 0 ((ComplexShape.down ℕ).next 0)).hom = ⊤
        rw [ChainComplex.next_nat_zero, C.shape 0 0 (by simp)]
        simp)
      (by
        change LinearMap.range (C.d ((ComplexShape.down ℕ).prev 0) 0).hom = _
        rw [ChainComplex.prev]
        rfl)).trans
    ((Submodule.quotEquivOfEq _ _ (by ext x; exact (hker x.1).symm)).trans
      ((φ ∘ₗ (⊤ : Submodule k (C.X 0)).subtype).quotKerEquivOfSurjective fun m ↦ by
        obtain ⟨x, rfl⟩ := hφ m
        exact ⟨⟨x, trivial⟩, rfl⟩))

/-- The homology of a complex of modules vanishes in a degree where the module of chains
vanishes. -/
theorem subsingleton_homology_of_subsingleton {ι : Type*} {c : ComplexShape ι}
    (C : HomologicalComplex (ModuleCat.{w} k) c) (i : ι) (h : Subsingleton (C.X i)) :
    Subsingleton (C.homology i) :=
  haveI : Subsingleton (C.sc i).X₂ := h
  (moduleCatHomologyEquiv (C.sc i) rfl rfl).toEquiv.subsingleton

/-- The homology in degree `n + 1` of a chain complex of modules indexed by `ℕ` is `M`, given a
surjective linear map from the cycles `ker (d : C_(n+1) → C_n)` to `M` whose kernel is the image
of `d : C_(n+2) → C_(n+1)`. Used for the `-1` eigen-subcomplex of the doubling example. -/
noncomputable def homologySuccEquiv (C : ChainComplex (ModuleCat.{w} k) ℕ) (n : ℕ) {M : Type*}
    [AddCommGroup M] [Module k M] (φ : LinearMap.ker (C.d (n + 1) n).hom →ₗ[k] M)
    (hφ : Function.Surjective φ)
    (hker : ∀ z, φ z = 0 ↔ z.1 ∈ LinearMap.range (C.d (n + 1 + 1) (n + 1)).hom) :
    C.homology (n + 1) ≃ₗ[k] M :=
  (moduleCatHomologyEquiv (C.sc (n + 1)) (Z := LinearMap.ker (C.d (n + 1) n).hom)
      (B := LinearMap.range (C.d (n + 1 + 1) (n + 1)).hom)
      (by
        change LinearMap.ker (C.d (n + 1) ((ComplexShape.down ℕ).next (n + 1))).hom = _
        rw [ChainComplex.next_nat_succ]
        rfl)
      (by
        change LinearMap.range (C.d ((ComplexShape.down ℕ).prev (n + 1)) (n + 1)).hom = _
        rw [ChainComplex.prev]
        rfl)).trans
    ((Submodule.quotEquivOfEq _ _ (by ext z; exact (hker z).symm)).trans
      (φ.quotKerEquivOfSurjective hφ))

end Degree0

section Eigen

/-! The homology of the eigen-subcomplexes from the action of the involution on homology, through
the splitting `homologyEquivEigen` of `Homology/Basic`. -/

variable {k : Type u} [CommRing k] [Invertible (2 : k)] {ι : Type*} {c : ComplexShape ι}
  {C : HomologicalComplex (ModuleCat.{w} k) c} {σ : C ⟶ C} {i : ι}

/-- If the involution `σ` acts by `-1` on `H_i(C)`, `2` invertible, then `H_i(C⁺) = 0`: the
`+1` eigenspace of `σ` on `H_i(C)` is zero (Proposition `prop:z2_reconstruction`). -/
theorem subsingleton_homology_plus (hσ : σ ≫ σ = 𝟙 C)
    (h : ∀ x, (HomologicalComplex.homologyMap σ i).hom x = -x) :
    Subsingleton ((plusSubcomplex σ).homology i) := by
  have key : ∀ y, (HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom y
      = 0 := by
    intro y
    apply eq_zero_of_eq_neg_of_invertible k
    have := h ((HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom y)
    rwa [homologyMap_hom_comp_apply, plusι_comp] at this
  exact ⟨fun y₁ y₂ ↦ injective_homologyMap_plusι hσ i (by rw [key, key])⟩

/-- If the involution `σ` acts by `1` on `H_i(C)`, `2` invertible, then `H_i(C⁻) = 0`: the `-1`
eigenspace of `σ` on `H_i(C)` is zero (Proposition `prop:z2_reconstruction`). -/
theorem subsingleton_homology_minus (hσ : σ ≫ σ = 𝟙 C)
    (h : ∀ x, (HomologicalComplex.homologyMap σ i).hom x = x) :
    Subsingleton ((minusSubcomplex σ).homology i) := by
  have key : ∀ y, (HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom y
      = 0 := by
    intro y
    apply eq_zero_of_eq_neg_of_invertible k
    have := h ((HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom y)
    rw [homologyMap_hom_comp_apply, minusι_comp, HomologicalComplex.homologyMap_neg,
      ModuleCat.hom_neg, LinearMap.neg_apply] at this
    exact this.symm
  exact ⟨fun y₁ y₂ ↦ injective_homologyMap_minusι hσ i (by rw [key, key])⟩

/-- If the involution `σ` acts by `1` on `H_i(C)`, `2` invertible, then the inclusion
`C⁺ → C` induces `H_i(C⁺) ≃ H_i(C)` (Proposition `prop:z2_reconstruction`). -/
noncomputable def homologyPlusEquiv (hσ : σ ≫ σ = 𝟙 C)
    (h : ∀ x, (HomologicalComplex.homologyMap σ i).hom x = x) :
    (plusSubcomplex σ).homology i ≃ₗ[k] C.homology i :=
  LinearEquiv.ofBijective
    (HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom
    ⟨injective_homologyMap_plusι hσ i, fun x ↦ by
      have hx : x ∈ LinearMap.range
          (HomologicalComplex.homologyMap (subcomplexι : plusSubcomplex σ ⟶ C) i).hom := by
        rw [range_homologyMap_plusι hσ i, LinearMap.mem_ker, LinearMap.sub_apply,
          Module.End.one_apply, h, sub_self]
      exact LinearMap.mem_range.mp hx⟩

/-- If the involution `σ` acts by `-1` on `H_i(C)`, `2` invertible, then the inclusion
`C⁻ → C` induces `H_i(C⁻) ≃ H_i(C)` (Proposition `prop:z2_reconstruction`). -/
noncomputable def homologyMinusEquiv (hσ : σ ≫ σ = 𝟙 C)
    (h : ∀ x, (HomologicalComplex.homologyMap σ i).hom x = -x) :
    (minusSubcomplex σ).homology i ≃ₗ[k] C.homology i :=
  LinearEquiv.ofBijective
    (HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom
    ⟨injective_homologyMap_minusι hσ i, fun x ↦ by
      have hx : x ∈ LinearMap.range
          (HomologicalComplex.homologyMap (subcomplexι : minusSubcomplex σ ⟶ C) i).hom := by
        rw [range_homologyMap_minusι hσ i, LinearMap.mem_ker, LinearMap.add_apply,
          Module.End.one_apply, h, neg_add_cancel]
      exact LinearMap.mem_range.mp hx⟩

end Eigen

/-! ### (a) The doubling map on a mirror -/

section Doubling

variable (k : Type u) [CommRing k]

/-- The number of cells of `N = [-1, 1]` not in `L = [-1, -1/2] ∪ [1/2, 1]` in each degree, on
the grid with vertices at the multiples of `1/2`: the vertex `0` and the edges `[-1/2, 0]` and
`[0, 1/2]`. -/
def doublingDim : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | _ => 0

/-- The relative cellular chain complex `C_*(N, L)` of the doubling example. The `0`-cell is
the vertex `v = 0`; the `1`-cells are `e₋ = [-1/2, 0]` (`0 : Fin 2`), oriented from `-1/2` to `0`,
and `e₊ = [0, 1/2]` (`1 : Fin 2`), oriented from `1/2` to `0`. Relative to `L` both have boundary
`v`, so `[e₋ : v] = [e₊ : v] = 1`; there are no cells in degrees `≥ 2`. -/
def doubling : CellComplex.{u, 0} k where
  cells n := Fin (doublingDim n)
  inc
    | 0 => fun _ _ ↦ 1
    | _ + 1 => fun _ _ ↦ 0
  bd_bd n c := by
    rcases n with _ | n
    · ext j
      simp
    · ext j
      simp

/-- The mirror `x ↦ -x` on the cells of the doubling example: it swaps `e₋` and `e₊` and fixes
`v`. The orientations above are chosen so that it permutes the cells without signs. -/
def doublingCellMap : ∀ n, Fin (doublingDim n) → Fin (doublingDim n)
  | 1 => Fin.rev
  | _ => id

/-- The mirror is an involution of the cells. -/
theorem doublingCellMap_involutive : ∀ n, Function.Involutive (doublingCellMap n)
  | 0 => fun _ ↦ rfl
  | 1 => Fin.rev_rev
  | _ + 2 => fun _ ↦ rfl

/-- The mirror preserves the incidence numbers, `[g e : g e'] = [e : e']`. -/
theorem doublingCellMap_inc : ∀ n (e : (doubling k).cells (n + 1)) (e' : (doubling k).cells n),
    (doubling k).inc n (doublingCellMap (n + 1) e) (doublingCellMap n e') =
      (doubling k).inc n e e'
  | 0, _, _ => rfl
  | _ + 1, _, _ => rfl

/-- The action `c ↦ c ∘ g` of the mirror on `C_*(N, L)`, a chain map. -/
def doublingInvolution : (doubling k).ChainMap (doubling k) :=
  CellComplex.ChainMap.ofInvolutive (K := doubling k) doublingCellMap doublingCellMap_involutive
    (doublingCellMap_inc k)

/-- The action of the mirror on chains is an involution. -/
theorem doublingInvolution_comp_self :
    (doublingInvolution k).comp (doublingInvolution k) = CellComplex.ChainMap.id _ :=
  CellComplex.ChainMap.ofInvolutive_comp_self

variable {k}

/-- The boundary of a `1`-chain: `∂c = (c(e₋) + c(e₊)) v`. -/
theorem doubling_bd_zero_apply (c : (doubling k).cells 1 → k) (j : (doubling k).cells 0) :
    bd ((doubling k).inc 0) c j = c (0 : Fin 2) + c (1 : Fin 2) := by
  change ∑ i : Fin 2, c i * 1 = _
  simp [Fin.sum_univ_two]

/-- The `1`-cycles are the chains with `c(e₋) + c(e₊) = 0`. -/
theorem doubling_mem_cycles_one {c : (doubling k).cells 1 → k} :
    c ∈ (doubling k).cycles 1 ↔ c (0 : Fin 2) + c (1 : Fin 2) = 0 := by
  rw [CellComplex.mem_cycles_succ, funext_iff]
  constructor
  · intro h
    have h0 := h (0 : Fin 1)
    rwa [doubling_bd_zero_apply] at h0
  · intro h j
    rw [doubling_bd_zero_apply, h]
    rfl

/-- Every `0`-chain is a boundary. -/
theorem doubling_boundaries_zero : (doubling k).boundaries 0 = ⊤ := by
  rw [eq_top_iff]
  intro c _
  refine LinearMap.mem_range.mpr ⟨(![c (0 : Fin 1), 0] : Fin 2 → k),
    funext fun (j : Fin 1) ↦ ?_⟩
  rw [doubling_bd_zero_apply, Fin.fin_one_eq_zero j]
  simp

/-- There are no `1`-boundaries. -/
theorem doubling_boundaries_one : (doubling k).boundaries 1 = ⊥ :=
  boundaries_eq_bot_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- `H_0(N, L) = 0` over any commutative ring. -/
theorem subsingleton_doubling_H_zero : Subsingleton ((doubling k).H 0) :=
  subsingleton_H_of_le (by rw [doubling_boundaries_zero]; exact le_top)

/-- `H_n(N, L) = 0` for `n ≥ 2`: there are no cells. -/
theorem subsingleton_doubling_H_add_two (n : ℕ) : Subsingleton ((doubling k).H (n + 2)) :=
  subsingleton_H_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- The cycle `e₋ - e₊`, the interval `[-1/2, 1/2]` oriented from `-1/2` to `1/2`, which
generates `H_1(N, L)`. -/
def doublingCycle : (doubling k).cycles 1 :=
  ⟨(![1, -1] : Fin 2 → k), doubling_mem_cycles_one.mpr (by simp)⟩

/-- Evaluation of a `1`-cycle at `e₋`. -/
def doublingEval : (doubling k).cycles 1 →ₗ[k] k :=
  (LinearMap.proj (R := k) (φ := fun _ : (doubling k).cells 1 ↦ k) (0 : Fin 2)) ∘ₗ
    ((doubling k).cycles 1).subtype

/-- `doublingEval` is the coefficient of `e₋`. -/
theorem doublingEval_apply (z : (doubling k).cycles 1) : doublingEval z = z.1 (0 : Fin 2) :=
  rfl

/-- Every `a ∈ k` is the coefficient of `e₋` of the cycle `a (e₋ - e₊)`. -/
theorem doublingEval_surjective : Function.Surjective (doublingEval (k := k)) := fun a ↦
  ⟨⟨(![a, -a] : Fin 2 → k), doubling_mem_cycles_one.mpr (by simp)⟩, rfl⟩

/-- A `1`-cycle vanishing at `e₋` is zero, which is the only `1`-boundary. -/
theorem doublingEval_eq_zero_iff (z : (doubling k).cycles 1) :
    doublingEval z = 0 ↔ (z : (doubling k).cells 1 → k) ∈ (doubling k).boundaries 1 := by
  rw [doubling_boundaries_one, Submodule.mem_bot, doublingEval_apply]
  have hz := doubling_mem_cycles_one.mp z.2
  constructor
  · intro h
    refine funext fun (e : Fin 2) ↦ ?_
    fin_cases e
    · exact h
    · change z.1 (1 : Fin 2) = 0
      rw [h, zero_add] at hz
      exact hz
  · intro h
    rw [h]
    rfl

/-- `H_1(N, L) ≅ k` over any commutative ring, by the coefficient of `e₋`. -/
noncomputable def doublingH1Equiv : (doubling k).H 1 ≃ₗ[k] k :=
  hEquivOfSurjective doublingEval doublingEval_surjective doublingEval_eq_zero_iff

/-- The class of `e₋ - e₊` corresponds to `1 ∈ k`. -/
theorem doublingH1Equiv_cycle :
    doublingH1Equiv (k := k) (Submodule.Quotient.mk doublingCycle) = 1 :=
  rfl

/-- `H_1(N, L)` is spanned by the class of `e₋ - e₊`: every class is a multiple of it. -/
theorem doubling_H_one_eq_smul (x : (doubling k).H 1) :
    x = doublingH1Equiv (k := k) x • Submodule.Quotient.mk doublingCycle := by
  apply doublingH1Equiv.injective
  rw [map_smul, doublingH1Equiv_cycle, smul_eq_mul, mul_one]

/-- The ranks `(0, 1)` of `H_*(N, L)` for the doubling example, over any commutative ring:
`H_0 = 0`, `H_1 ≅ k`, and `H_n = 0` for `n ≥ 2`. -/
theorem doubling_ranks :
    Subsingleton ((doubling k).H 0) ∧ Nonempty ((doubling k).H 1 ≃ₗ[k] k) ∧
      ∀ n, Subsingleton ((doubling k).H (n + 2)) :=
  ⟨subsingleton_doubling_H_zero, ⟨doublingH1Equiv⟩, subsingleton_doubling_H_add_two⟩

/-- The mirror acts by `-1` on `H_1(N, L)`, since it maps `e₋ - e₊` to `e₊ - e₋`. -/
theorem doublingInvolution_mapH_one (x : (doubling k).H 1) :
    (doublingInvolution k).mapH 1 x = -x := by
  induction x using Submodule.Quotient.induction_on with
  | H z =>
    rw [CellComplex.ChainMap.mapH_mk, ← Submodule.Quotient.mk_neg]
    congr 1
    apply Subtype.ext
    have hz := doubling_mem_cycles_one.mp z.2
    refine funext fun (e : Fin 2) ↦ ?_
    fin_cases e
    · change z.1 (1 : Fin 2) = -z.1 (0 : Fin 2)
      linear_combination hz
    · change z.1 (0 : Fin 2) = -z.1 (1 : Fin 2)
      linear_combination hz

/-- The mirror acts by `-1` on `H_n(N, L)` for every `n`. -/
theorem doublingInvolution_mapH (n : ℕ) (x : (doubling k).H n) :
    (doublingInvolution k).mapH n x = -x := by
  rcases n with _ | _ | n
  · haveI := subsingleton_doubling_H_zero (k := k)
    exact Subsingleton.elim _ _
  · exact doublingInvolution_mapH_one x
  · haveI := subsingleton_doubling_H_add_two (k := k) n
    exact Subsingleton.elim _ _

/-- The mirror acts by `-1` on Mathlib's homology of `C_*(N, L)` in every degree. -/
theorem doublingInvolution_homologyMap (n : ℕ) (x : (doubling k).toChainComplex.homology n) :
    (doublingInvolution k).homologyMap n x = -x :=
  homologyMap_eq_neg (doublingInvolution_mapH n) x

/-- If `2` is invertible, the `+1` eigen-subcomplex of the mirror is acyclic: the trivial
component, which is the orbit-space index, is `0` (Proposition
`prop:z2_reconstruction`). -/
theorem doubling_plus_acyclic [Invertible (2 : k)] (n : ℕ) :
    Subsingleton ((plusSubcomplex (doublingInvolution k).toHom).homology n) :=
  subsingleton_homology_plus
    (CellComplex.ChainMap.toHom_comp_self (doublingInvolution_comp_self k))
    (doublingInvolution_homologyMap n)

/-- If `2` is invertible, the `-1` eigen-subcomplex of the mirror has the homology of `(N, L)`,
through the inclusion (Proposition `prop:z2_reconstruction`). -/
noncomputable def doublingMinusEquiv [Invertible (2 : k)] (n : ℕ) :
    (minusSubcomplex (doublingInvolution k).toHom).homology n ≃ₗ[k]
      (doubling k).toChainComplex.homology n :=
  homologyMinusEquiv (CellComplex.ChainMap.toHom_comp_self (doublingInvolution_comp_self k))
    (doublingInvolution_homologyMap n)

/-! ### (a) The sign component over any ring -/

/-- The `-1` eigen-subcomplex of the mirror consists of the anti-invariant chains,
`c ∘ g = -c`. -/
theorem mem_minus_doubling_iff {n : ℕ} {c : (doubling k).cells n → k} :
    c ∈ LinearMap.ker
        (((doublingInvolution k).toHom + 𝟙 (doubling k).toChainComplex).f n).hom ↔
      ∀ e, c (doublingCellMap n e) = -c e :=
  CellComplex.ChainMap.mem_minus_ofInvolutive_iff

/-- The differential of the `-1` eigen-subcomplex is the boundary of `C_*(N, L)`. -/
theorem doubling_minus_d_apply (n : ℕ)
    (y : (minusSubcomplex (doublingInvolution k).toHom).X (n + 1)) :
    (((minusSubcomplex (doublingInvolution k).toHom).d (n + 1) n).hom y).1 =
      bd ((doubling k).inc n) y.1 :=
  LinearMap.congr_fun (CellComplex.toChainComplex_d_hom (doubling k) n) y.1

/-- There are no `2`-cells, so the boundary of a `2`-chain is `0`. -/
theorem doubling_bd_one (c : (doubling k).cells 2 → k) : bd ((doubling k).inc 1) c = 0 :=
  funext fun j ↦ by
    change ∑ i : Fin 0, c i * 0 = (0 : k)
    simp

/-- The boundary vanishes on the anti-invariant `1`-chains, the multiples of `e₋ - e₊`. -/
theorem doubling_bd_of_mem_minus {c : (doubling k).cells 1 → k}
    (hc : c ∈ LinearMap.ker
      (((doublingInvolution k).toHom + 𝟙 (doubling k).toChainComplex).f 1).hom) :
    bd ((doubling k).inc 0) c = 0 := by
  have h : c (1 : Fin 2) = -c (0 : Fin 2) := mem_minus_doubling_iff.mp hc (0 : Fin 2)
  refine funext fun (j : Fin 1) ↦ ?_
  rw [doubling_bd_zero_apply, h, add_neg_cancel]
  rfl

/-- Evaluation at `e₋` of the anti-invariant `1`-cycles. -/
def doublingMinusEval :
    LinearMap.ker ((minusSubcomplex (doublingInvolution k).toHom).d 1 0).hom →ₗ[k] k :=
  (LinearMap.proj (R := k) (φ := fun _ : (doubling k).cells 1 ↦ k) (0 : Fin 2)) ∘ₗ
    (LinearMap.ker
      (((doublingInvolution k).toHom + 𝟙 (doubling k).toChainComplex).f 1).hom).subtype ∘ₗ
    (LinearMap.ker ((minusSubcomplex (doublingInvolution k).toHom).d 1 0).hom).subtype

/-- `doublingMinusEval` is the coefficient of `e₋`. -/
theorem doublingMinusEval_apply
    (z : LinearMap.ker ((minusSubcomplex (doublingInvolution k).toHom).d 1 0).hom) :
    doublingMinusEval z = z.1.1 (0 : Fin 2) :=
  rfl

/-- Every `a ∈ k` is the coefficient of `e₋` of the anti-invariant cycle `a (e₋ - e₊)`. -/
theorem doublingMinusEval_surjective : Function.Surjective (doublingMinusEval (k := k)) := by
  intro a
  have hy : (![a, -a] : Fin 2 → k) ∈ LinearMap.ker
      (((doublingInvolution k).toHom + 𝟙 (doubling k).toChainComplex).f 1).hom :=
    mem_minus_doubling_iff.mpr fun (e : Fin 2) ↦ by
      fin_cases e
      · rfl
      · exact (neg_neg a).symm
  refine ⟨⟨⟨(![a, -a] : Fin 2 → k), hy⟩, ?_⟩, rfl⟩
  rw [LinearMap.mem_ker]
  exact Subtype.ext ((doubling_minus_d_apply 0 _).trans (doubling_bd_of_mem_minus hy))

/-- An anti-invariant `1`-cycle vanishing at `e₋` is zero, which is the only boundary. -/
theorem doublingMinusEval_eq_zero_iff
    (z : LinearMap.ker ((minusSubcomplex (doublingInvolution k).toHom).d 1 0).hom) :
    doublingMinusEval z = 0 ↔
      z.1 ∈ LinearMap.range ((minusSubcomplex (doublingInvolution k).toHom).d 2 1).hom := by
  rw [doublingMinusEval_apply]
  constructor
  · intro h
    have h1 : z.1.1 (1 : Fin 2) = -z.1.1 (0 : Fin 2) :=
      mem_minus_doubling_iff.mp z.1.2 (0 : Fin 2)
    have hz : z.1 = 0 := Subtype.ext (funext fun (e : Fin 2) ↦ by
      fin_cases e
      · exact h
      · change z.1.1 (1 : Fin 2) = 0
        rw [h1, h, neg_zero])
    rw [hz]
    exact zero_mem _
  · rintro ⟨y, hy⟩
    have hz : z.1.1 = 0 := by
      rw [← hy, doubling_minus_d_apply, doubling_bd_one]
    rw [hz]
    rfl

/-- Over any commutative ring, `H_1` of the `-1` eigen-subcomplex of the mirror is `k`, by the
coefficient of `e₋`. With `2` invertible (or, here, not a zero divisor) the sign component
carries the index, and this is also `doublingMinusEquiv 1`. Without it,
`C⁻` is not a summand of `C` and the computation says nothing about `H_*(N, L)`. -/
noncomputable def doublingMinusH1Equiv :
    (minusSubcomplex (doublingInvolution k).toHom).homology 1 ≃ₗ[k] k :=
  homologySuccEquiv _ 0 doublingMinusEval doublingMinusEval_surjective
    doublingMinusEval_eq_zero_iff

/-- Evaluation at `v` of the anti-invariant `0`-chains, whose values are `2`-torsion. -/
def doublingMinusEval0 : (minusSubcomplex (doublingInvolution k).toHom).X 0 →ₗ[k]
    LinearMap.ker (LinearMap.mulLeft k (2 : k)) :=
  LinearMap.codRestrict _
    ((LinearMap.proj (R := k) (φ := fun _ : (doubling k).cells 0 ↦ k) (0 : Fin 1)) ∘ₗ
      (LinearMap.ker
        (((doublingInvolution k).toHom + 𝟙 (doubling k).toChainComplex).f 0).hom).subtype)
    fun x ↦ by
      have h : x.1 (0 : Fin 1) = -x.1 (0 : Fin 1) := mem_minus_doubling_iff.mp x.2 (0 : Fin 1)
      rw [LinearMap.mem_ker, LinearMap.mulLeft_apply]
      change 2 * x.1 (0 : Fin 1) = 0
      linear_combination h

/-- `doublingMinusEval0` is the value at `v`. -/
theorem coe_doublingMinusEval0_apply (x : (minusSubcomplex (doublingInvolution k).toHom).X 0) :
    (doublingMinusEval0 x : k) = x.1 (0 : Fin 1) :=
  rfl

/-- Every `a ∈ k` with `2a = 0` is the value at `v` of the anti-invariant chain `a v`. -/
theorem doublingMinusEval0_surjective : Function.Surjective (doublingMinusEval0 (k := k)) := by
  rintro ⟨a, ha⟩
  have ha' : a = -a := by
    rw [LinearMap.mem_ker, LinearMap.mulLeft_apply] at ha
    linear_combination ha
  exact ⟨⟨fun _ ↦ a, mem_minus_doubling_iff.mpr fun _ ↦ ha'⟩, rfl⟩

/-- An anti-invariant `0`-chain is a boundary if and only if it is zero. -/
theorem doublingMinusEval0_eq_zero_iff (x : (minusSubcomplex (doublingInvolution k).toHom).X 0) :
    doublingMinusEval0 x = 0 ↔
      x ∈ LinearMap.range ((minusSubcomplex (doublingInvolution k).toHom).d 1 0).hom := by
  rw [Subtype.ext_iff, coe_doublingMinusEval0_apply, ZeroMemClass.coe_zero]
  constructor
  · intro h
    have hx : x = 0 := Subtype.ext (funext fun (j : Fin 1) ↦ by
      rw [Fin.fin_one_eq_zero j]
      exact h)
    rw [hx]
    exact zero_mem _
  · rintro ⟨y, rfl⟩
    rw [doubling_minus_d_apply, doubling_bd_of_mem_minus y.2]
    rfl

/-- Over any commutative ring, `H_0` of the `-1` eigen-subcomplex of the mirror is the
`2`-torsion `{a ∈ k | 2a = 0}`, by the value at the fixed vertex `v`. -/
noncomputable def doublingMinusH0Equiv :
    (minusSubcomplex (doublingInvolution k).toHom).homology 0 ≃ₗ[k]
      LinearMap.ker (LinearMap.mulLeft k (2 : k)) :=
  homologyZeroEquiv _ doublingMinusEval0 doublingMinusEval0_surjective
    doublingMinusEval0_eq_zero_iff

/-- If `2` is not a zero divisor, for instance if `2` is invertible or `k = ℤ`, `H_0` of the `-1`
eigen-subcomplex of the mirror is `0`. -/
theorem subsingleton_doubling_minus_H_zero (h2 : IsLeftRegular (2 : k)) :
    Subsingleton ((minusSubcomplex (doublingInvolution k).toHom).homology 0) := by
  haveI : Subsingleton (LinearMap.ker (LinearMap.mulLeft k (2 : k))) :=
    ⟨fun a b ↦ Subtype.ext (h2 ((LinearMap.mem_ker.mp a.2).trans (LinearMap.mem_ker.mp b.2).symm))⟩
  exact doublingMinusH0Equiv.toEquiv.subsingleton

/-- `H_n` of the `-1` eigen-subcomplex of the mirror is `0` for `n ≥ 2`, over any commutative
ring: there are no cells. -/
theorem subsingleton_doubling_minus_H_add_two (n : ℕ) :
    Subsingleton ((minusSubcomplex (doublingInvolution k).toHom).homology (n + 2)) :=
  subsingleton_homology_of_subsingleton _ _
    ⟨fun _ _ ↦ Subtype.ext (funext fun (j : Fin 0) ↦ j.elim0)⟩

/-! ### (b) The invariant chains of the doubling example over any ring -/

/-- The `+1` eigen-subcomplex of the mirror consists of the invariant chains, `c ∘ g = c`. -/
theorem mem_plus_doubling_iff {n : ℕ} {c : (doubling k).cells n → k} :
    c ∈ LinearMap.ker
        (((doublingInvolution k).toHom - 𝟙 (doubling k).toChainComplex).f n).hom ↔
      ∀ e, c (doublingCellMap n e) = c e :=
  CellComplex.ChainMap.mem_plus_ofInvolutive_iff

/-- The boundary of an invariant `1`-chain `c (e₋ + e₊)` is `2c v`: in the coordinates of the
invariant chains the incidence at the fixed vertex is `[e₋ : v] + [e₋ : g v] = 2`
(`IsRepresentatives.bd_apply_symChains`), while in the orbit complex it is `[e₋ : v] = 1`. -/
theorem doubling_bd_of_mem_plus {c : (doubling k).cells 1 → k}
    (hc : c ∈ LinearMap.ker
      (((doublingInvolution k).toHom - 𝟙 (doubling k).toChainComplex).f 1).hom)
    (j : (doubling k).cells 0) : bd ((doubling k).inc 0) c j = 2 * c (0 : Fin 2) := by
  have h := mem_plus_doubling_iff.mp hc (0 : Fin 2)
  rw [doubling_bd_zero_apply, two_mul]
  exact congrArg (c (0 : Fin 2) + ·) h

/-- Evaluation at `v` of the invariant `0`-chains, modulo `2`. -/
def doublingPlusEval :
    (plusSubcomplex (doublingInvolution k).toHom).X 0 →ₗ[k] k ⧸ Ideal.span {(2 : k)} :=
  (Ideal.span {(2 : k)}).mkQ ∘ₗ
    (LinearMap.proj (R := k) (φ := fun _ : (doubling k).cells 0 ↦ k) (0 : Fin 1)) ∘ₗ
      (LinearMap.ker
        (((doublingInvolution k).toHom - 𝟙 (doubling k).toChainComplex).f 0).hom).subtype

/-- `doublingPlusEval` is the value at `v`, modulo `2`. -/
theorem doublingPlusEval_apply (x : (plusSubcomplex (doublingInvolution k).toHom).X 0) :
    doublingPlusEval x = Ideal.Quotient.mk _ (x.1 (0 : Fin 1)) :=
  rfl

/-- `doublingPlusEval` is surjective, since every `0`-chain is invariant. -/
theorem doublingPlusEval_surjective : Function.Surjective (doublingPlusEval (k := k)) := by
  intro m
  obtain ⟨a, rfl⟩ := Ideal.Quotient.mk_surjective m
  exact ⟨⟨fun _ ↦ a, mem_plus_doubling_iff.mpr fun _ ↦ rfl⟩, rfl⟩

/-- The differential of the `+1` eigen-subcomplex is the boundary of `C_*(N, L)`. -/
theorem doubling_plus_d_apply (y : (plusSubcomplex (doublingInvolution k).toHom).X 1) :
    (((plusSubcomplex (doublingInvolution k).toHom).d 1 0).hom y).1 =
      bd ((doubling k).inc 0) y.1 :=
  LinearMap.congr_fun (CellComplex.toChainComplex_d_hom (doubling k) 0) y.1

/-- An invariant `0`-chain is a boundary of an invariant `1`-chain if and only if its value at
`v` is divisible by `2`. -/
theorem doublingPlusEval_eq_zero_iff (x : (plusSubcomplex (doublingInvolution k).toHom).X 0) :
    doublingPlusEval x = 0 ↔
      x ∈ LinearMap.range ((plusSubcomplex (doublingInvolution k).toHom).d 1 0).hom := by
  rw [doublingPlusEval_apply, Ideal.Quotient.eq_zero_iff_mem, Ideal.mem_span_singleton',
    LinearMap.mem_range]
  constructor
  · rintro ⟨a, ha⟩
    refine ⟨⟨fun _ ↦ a, mem_plus_doubling_iff.mpr fun _ ↦ rfl⟩, Subtype.ext ?_⟩
    rw [doubling_plus_d_apply]
    refine funext fun (j : Fin 1) ↦ ?_
    rw [doubling_bd_zero_apply, Fin.fin_one_eq_zero j, ← ha, mul_two]
  · rintro ⟨y, rfl⟩
    refine ⟨y.1 (0 : Fin 2), ?_⟩
    rw [doubling_plus_d_apply, doubling_bd_of_mem_plus y.2, mul_comm]

/-- Over any commutative ring, `H_0` of the invariant chains (the `+1` eigen-subcomplex of the
mirror) is `k ⧸ 2k`, by the value at `v` (compare the reconstruction from the orbit
space, which assumes `char F ∤ |G|`). -/
noncomputable def doublingPlusH0Equiv :
    (plusSubcomplex (doublingInvolution k).toHom).homology 0 ≃ₗ[k] k ⧸ Ideal.span {(2 : k)} :=
  homologyZeroEquiv _ doublingPlusEval doublingPlusEval_surjective doublingPlusEval_eq_zero_iff

/-- The `+1` eigen-subcomplex of the mirror has `H_0 = 0` if and only if `2` is a unit, so the
hypothesis of `doubling_plus_acyclic` cannot be weakened. -/
theorem subsingleton_doubling_plus_H_zero_iff :
    Subsingleton ((plusSubcomplex (doublingInvolution k).toHom).homology 0) ↔ IsUnit (2 : k) := by
  rw [doublingPlusH0Equiv.toEquiv.subsingleton_congr, Submodule.Quotient.subsingleton_iff,
    Ideal.span_singleton_eq_top]

/-- Over `ℤ`, `H_0` of the invariant chains of the doubling example is `ℤ/2ℤ`. -/
noncomputable def doublingPlusH0EquivInt :
    (plusSubcomplex (doublingInvolution ℤ).toHom).homology 0 ≃ₗ[ℤ] ZMod 2 :=
  doublingPlusH0Equiv.trans (Int.quotientSpanEquivZMod 2).toAddEquiv.toIntLinearEquiv

/-- Over `ℤ`, `H_0` of the invariant chains of the doubling example is nonzero. -/
theorem nontrivial_doubling_plus_H_zero :
    Nontrivial ((plusSubcomplex (doublingInvolution ℤ).toHom).homology 0) :=
  doublingPlusH0EquivInt.toEquiv.nontrivial

end Doubling

section DoublingOrbit

variable (k : Type u) [CommRing k]

/-- The number of cells of the orbit complex in each degree: one vertex and one edge. -/
def doublingOrbitDim : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | _ => 0

/-- The orbit complex of the doubling example: the cells of `(N/G, L/G) = ([0, 1], [1/2, 1])` not
in `L/G`, the vertex `[v]` and the edge `[e₋] = [e₊]`, with incidence
`[[e₋] : [v]] = [e₋ : v] = 1` (the incidences `orbitInc` of `OrbitChains`). -/
def doublingOrbit : CellComplex.{u, 0} k where
  cells n := Fin (doublingOrbitDim n)
  inc
    | 0 => fun _ _ ↦ 1
    | _ + 1 => fun _ _ ↦ 0
  bd_bd n c := by
    rcases n with _ | n
    · ext j
      simp
    · ext j
      simp

variable {k}

/-- The boundary of the orbit complex is the identity `C_1 = k → k = C_0`. -/
theorem doublingOrbit_bd_zero_apply (c : (doublingOrbit k).cells 1 → k)
    (j : (doublingOrbit k).cells 0) : bd ((doublingOrbit k).inc 0) c j = c (0 : Fin 1) := by
  change ∑ i : Fin 1, c i * 1 = _
  simp

/-- The orbit complex of the doubling example is acyclic over any commutative ring: the
orbit-space index is `0`. -/
theorem doublingOrbit_acyclic (n : ℕ) : Subsingleton ((doublingOrbit k).H n) := by
  rcases n with _ | _ | n
  · refine subsingleton_H_of_le fun c _ ↦ LinearMap.mem_range.mpr
      ⟨(fun _ ↦ c (0 : Fin 1) : Fin 1 → k), funext fun (j : Fin 1) ↦ ?_⟩
    rw [doublingOrbit_bd_zero_apply, Fin.fin_one_eq_zero j]
  · refine subsingleton_H_of_le fun c hc ↦ ?_
    have hc0 := congr_fun ((CellComplex.mem_cycles_succ _).mp hc) (0 : Fin 1)
    rw [doublingOrbit_bd_zero_apply] at hc0
    have hc' : c = 0 := funext fun (j : Fin 1) ↦ by rw [Fin.fin_one_eq_zero j]; exact hc0
    rw [hc']
    exact zero_mem _
  · exact subsingleton_H_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- Evaluation at the representatives, `e₋ + e₊ ↦ [e]` and `v ↦ [v]`, does not commute with the
boundaries over a nontrivial ring: the invariant chain `e₋ + e₊` has boundary `2v`, while the
orbit edge `[e]` has boundary `[v]`. So the identification of the invariant chains with the orbit
complex in the proof of Proposition `prop:z2_reconstruction` is `π_#` (`push` of `OrbitChains`),
which maps `e₋ + e₊` to `2[e]`, and not evaluation at the representatives. -/
theorem doubling_bd_ne_doublingOrbit_bd [Nontrivial k] :
    bd ((doubling k).inc 0) (fun _ ↦ 1) (0 : Fin 1) ≠
      bd ((doublingOrbit k).inc 0) (fun _ ↦ 1) (0 : Fin 1) := by
  rw [doubling_bd_zero_apply, doublingOrbit_bd_zero_apply]
  intro h
  exact one_ne_zero (left_eq_add.mp h.symm)

/-- `H_0` of the invariant chains of the doubling example (`k ⧸ 2k`) is isomorphic to `H_0` of
the orbit complex (`0`) if and only if `2` is a unit. So if `2` is not a unit, for instance over
`ℤ` or `ZMod 2`, the identification of the invariant chains with the orbit complex
(`transferEquiv` of `OrbitChains`) fails on homology, not only on
chains. -/
theorem isEmpty_plus_homology_equiv_orbit_iff :
    IsEmpty ((plusSubcomplex (doublingInvolution k).toHom).homology 0 ≃ₗ[k]
      (doublingOrbit k).toChainComplex.homology 0) ↔ ¬ IsUnit (2 : k) := by
  haveI := subsingleton_homology (doublingOrbit_acyclic (k := k) 0)
  rw [← subsingleton_doubling_plus_H_zero_iff]
  constructor
  · intro h hs
    exact h.false (LinearEquiv.ofSubsingleton _ _)
  · intro h
    exact ⟨fun e ↦ h e.toEquiv.subsingleton⟩

/-- If `2` is not a unit, the invariant chains of the doubling example are not isomorphic, as a
complex, to the orbit complex. -/
theorem isEmpty_plus_iso_orbit_of_not_isUnit (h : ¬ IsUnit (2 : k)) :
    IsEmpty (plusSubcomplex (doublingInvolution k).toHom ≅ (doublingOrbit k).toChainComplex) :=
  ⟨fun e ↦ (isEmpty_plus_homology_equiv_orbit_iff.mpr h).false
    (HomologicalComplex.homologyMapIso e 0).toLinearEquiv⟩

/-- `2` is not a unit of `ℤ`. -/
theorem not_isUnit_two_int : ¬ IsUnit (2 : ℤ) := by
  rw [Int.isUnit_iff]
  decide

/-- Over `ℤ`, `H_0` of the invariant chains of the doubling example (`ℤ/2ℤ`) is not isomorphic to
`H_0` of the orbit complex (`0`): with `2` not invertible, the identification of the invariant
chains with the orbit complex fails on homology, not only on chains. -/
theorem isEmpty_plus_homology_equiv_orbit :
    IsEmpty ((plusSubcomplex (doublingInvolution ℤ).toHom).homology 0 ≃ₗ[ℤ]
      (doublingOrbit ℤ).toChainComplex.homology 0) :=
  isEmpty_plus_homology_equiv_orbit_iff.mpr not_isUnit_two_int

/-- Over `ℤ`, the invariant chains of the doubling example are not isomorphic, as a complex, to
the orbit complex. -/
theorem isEmpty_plus_iso_orbit :
    IsEmpty (plusSubcomplex (doublingInvolution ℤ).toHom ≅ (doublingOrbit ℤ).toChainComplex) :=
  isEmpty_plus_iso_orbit_of_not_isUnit not_isUnit_two_int

/-- If `2` is not a unit, the splitting `H_*(C) ≅ H_*(C⁺) × H_*(C⁻)` of `homologyEquivEigen`,
the chain-level step of the proof of Proposition `prop:z2_reconstruction`, fails for the doubling
example: `H_0(N, L) = 0`, while `H_0` of the `+1` eigen-subcomplex is `k ⧸ 2k ≠ 0`, so
`H_0(N, L)` is not `H_0(C⁺) × M` for any module `M`. The proposition's formula with orbit-space
terms holds for this example over every ring (`doubling_formula`). -/
theorem doubling_splitting_fails (h : ¬ IsUnit (2 : k)) (M : Type*) [AddCommGroup M]
    [Module k M] :
    IsEmpty ((doubling k).toChainComplex.homology 0 ≃ₗ[k]
      (plusSubcomplex (doublingInvolution k).toHom).homology 0 × M) := by
  refine ⟨fun e ↦ h ?_⟩
  haveI := subsingleton_homology (subsingleton_doubling_H_zero (k := k))
  haveI : Subsingleton ((plusSubcomplex (doublingInvolution k).toHom).homology 0 × M) :=
    e.symm.toEquiv.subsingleton
  haveI : Subsingleton ((plusSubcomplex (doublingInvolution k).toHom).homology 0) :=
    (Prod.mk_left_injective (0 : M)).subsingleton
  exact subsingleton_doubling_plus_H_zero_iff.mp ‹_›

/-- The formula of Proposition `prop:z2_reconstruction` holds for the doubling example over every
commutative ring, with no hypothesis on `2`: the orbit complex is acyclic, and the twisted term
has one cell, the free pair `{e₋, e₊}` in degree `1` (the fixed vertex `v` is dropped), so it is
`k` in degree `1` and `0` in degree `0`. Hence `H_1(N, L) ≅ H_1(N/G, L/G) × k` and
`H_0(N, L) ≅ H_0(N/G, L/G)`. -/
theorem doubling_formula :
    Nonempty ((doubling k).H 1 ≃ₗ[k] (doublingOrbit k).H 1 × k) ∧
      Nonempty ((doubling k).H 0 ≃ₗ[k] (doublingOrbit k).H 0) := by
  haveI := doublingOrbit_acyclic (k := k) 1
  haveI := doublingOrbit_acyclic (k := k) 0
  haveI := subsingleton_doubling_H_zero (k := k)
  exact ⟨⟨doublingH1Equiv.trans (LinearEquiv.ofBijective (LinearMap.inr k _ k)
    ⟨LinearMap.inr_injective, fun p ↦ ⟨p.2, Prod.ext (Subsingleton.elim _ _) rfl⟩⟩)⟩,
    ⟨LinearEquiv.ofSubsingleton _ _⟩⟩

end DoublingOrbit

/-! ### (c) `-I` on a square -/

section Square

variable (k : Type u) [CommRing k]

/-- The number of cells of `N = [-1, 1]²` not in `L = ∂N` in each degree, on the grid of unit
squares: the origin, four edges and four squares (the example after
Proposition `prop:z2_reconstruction`). -/
def squareDim : ℕ → ℕ
  | 0 => 1
  | 1 => 4
  | 2 => 4
  | _ => 0

/-- The incidence `[Q_i : e_j]` of the squares `Q1, Q2, Q3, Q4` (`i = 0, 1, 2, 3`) and the edges
`a, b, a', b'` (`j = 0, 1, 2, 3`): `1` if `j = i`, `-1` if `j = i + 1`, and `0` otherwise. So
`∂Q1 = a - b`, `∂Q2 = b - a'`, `∂Q3 = a' - b'`, `∂Q4 = b' - a` relative to `∂N`. -/
def squareInc (i j : Fin 4) : k := if j = i then 1 else if j = i + 1 then -1 else 0

/-- The incidence matrix of the squares (rows `Q1, Q2, Q3, Q4`) and the edges (columns
`a, b, a', b'`): `∂Q1 = a - b`, `∂Q2 = b - a'`, `∂Q3 = a' - b'`, `∂Q4 = b' - a`. -/
theorem squareInc_eq :
    squareInc k = ![![1, -1, 0, 0], ![0, 1, -1, 0], ![0, 0, 1, -1], ![-1, 0, 0, 1]] := by
  funext i j
  fin_cases i <;> fin_cases j <;> rfl

/-- The relative cellular chain complex `C_*(N, L)` of `N = [-1, 1]²` and `L = ∂N` on the grid of
unit squares (the example after Proposition `prop:z2_reconstruction`). The
`0`-cell is the origin `O`; the `1`-cells are `a = O → (1, 0)`, `b = O → (0, 1)`,
`a' = O → (-1, 0)`, `b' = O → (0, -1)`, which are `0, 1, 2, 3 : Fin 4`; the `2`-cells are
`Q1 = [0, 1]²`, `Q2 = [-1, 0] × [0, 1]`, `Q3 = [-1, 0]²`, `Q4 = [0, 1] × [-1, 0]`, which are
`0, 1, 2, 3 : Fin 4`, with the standard orientation. Relative to `∂N`, `∂e = -O` for every edge
and `∂Q` is given by `squareInc`. The field `bd_bd` is `∂ ∘ ∂ = 0`. -/
def square : CellComplex.{u, 0} k where
  cells n := Fin (squareDim n)
  inc
    | 0 => fun _ _ ↦ -1
    | 1 => squareInc k
    | _ + 2 => fun _ _ ↦ 0
  bd_bd n c := by
    rcases n with _ | _ | n
    · funext j
      change ∑ e : Fin 4, (∑ q : Fin 4, c q * squareInc k q e) * (-1) = 0
      simp only [Fin.sum_univ_four, squareInc]
      simp
      ring
    · funext j
      change ∑ e : Fin 4, (∑ q : Fin 0, c q * 0) * squareInc k e j = 0
      simp
    · funext j
      simp

/-- The half-turn `-I` on the edges `a ↔ a'`, `b ↔ b'` and on the squares `Q1 ↔ Q3`,
`Q2 ↔ Q4`: `i ↦ i + 2` on `Fin 4`. -/
def halfTurn (i : Fin 4) : Fin 4 := i + 2

/-- The half-turn is an involution. -/
theorem halfTurn_halfTurn (i : Fin 4) : halfTurn (halfTurn i) = i := by
  fin_cases i <;> rfl

/-- The half-turn fixes no edge and no square: they form the free orbits. -/
theorem halfTurn_ne_self (i : Fin 4) : halfTurn i ≠ i := by
  fin_cases i <;> decide

/-- The half-turn preserves the incidences of squares and edges, `[-Q : -e] = [Q : e]`. -/
theorem squareInc_halfTurn (i j : Fin 4) :
    squareInc k (halfTurn i) (halfTurn j) = squareInc k i j := by
  simp only [squareInc, halfTurn, add_right_comm i 2 1, add_left_inj]

/-- The action of `-I` on the cells of the square: the half-turn on edges and squares, and the
identity on the fixed vertex `O`. It permutes the cells without signs. -/
def squareCellMap : ∀ n, Fin (squareDim n) → Fin (squareDim n)
  | 1 => halfTurn
  | 2 => halfTurn
  | _ => id

/-- `-I` is an involution of the cells. -/
theorem squareCellMap_involutive : ∀ n, Function.Involutive (squareCellMap n)
  | 0 => fun _ ↦ rfl
  | 1 => halfTurn_halfTurn
  | 2 => halfTurn_halfTurn
  | _ + 3 => fun _ ↦ rfl

/-- `-I` preserves all incidence numbers, `[g e : g e'] = [e : e']`. -/
theorem squareCellMap_inc : ∀ n (e : (square k).cells (n + 1)) (e' : (square k).cells n),
    (square k).inc n (squareCellMap (n + 1) e) (squareCellMap n e') = (square k).inc n e e'
  | 0, _, _ => rfl
  | 1, e, e' => squareInc_halfTurn k e e'
  | _ + 2, _, _ => rfl

/-- The action `c ↦ c ∘ g` of `-I` on `C_*(N, L)`, a chain map. -/
def squareInvolution : (square k).ChainMap (square k) :=
  CellComplex.ChainMap.ofInvolutive (K := square k) squareCellMap squareCellMap_involutive
    (squareCellMap_inc k)

/-- The action of `-I` on chains is an involution. -/
theorem squareInvolution_comp_self :
    (squareInvolution k).comp (squareInvolution k) = CellComplex.ChainMap.id _ :=
  CellComplex.ChainMap.ofInvolutive_comp_self

variable {k}

/-- The boundary of a `1`-chain: `∂c = -(c(a) + c(b) + c(a') + c(b')) O`. -/
theorem square_bd_zero_apply (c : (square k).cells 1 → k) (j : (square k).cells 0) :
    bd ((square k).inc 0) c j =
      -(c (0 : Fin 4) + c (1 : Fin 4) + c (2 : Fin 4) + c (3 : Fin 4)) := by
  change ∑ i : Fin 4, c i * (-1) = _
  simp only [Fin.sum_univ_four]
  ring

/-- The boundary of a `2`-chain, by its coefficients on `a, b, a', b'`:
`(c(Q1) - c(Q4), c(Q2) - c(Q1), c(Q3) - c(Q2), c(Q4) - c(Q3))`. -/
theorem square_bd_one (c : (square k).cells 2 → k) :
    bd ((square k).inc 1) c = ![c (0 : Fin 4) - c (3 : Fin 4), c (1 : Fin 4) - c (0 : Fin 4),
      c (2 : Fin 4) - c (1 : Fin 4), c (3 : Fin 4) - c (2 : Fin 4)] := by
  refine funext fun (j : Fin 4) ↦ ?_
  change ∑ i : Fin 4, c i * squareInc k i j = _
  fin_cases j <;> simp [Fin.sum_univ_four, squareInc] <;> ring

/-- The `1`-cycles are the chains whose coefficients sum to `0`. -/
theorem square_mem_cycles_one {c : (square k).cells 1 → k} :
    c ∈ (square k).cycles 1 ↔
      c (0 : Fin 4) + c (1 : Fin 4) + c (2 : Fin 4) + c (3 : Fin 4) = 0 := by
  rw [CellComplex.mem_cycles_succ, funext_iff]
  constructor
  · intro h
    exact neg_eq_zero.mp ((square_bd_zero_apply c (0 : Fin 1)).symm.trans (h (0 : Fin 1)))
  · intro h j
    rw [square_bd_zero_apply, h, neg_zero]
    rfl

/-- The `2`-cycles are the constant chains, the multiples of `Q1 + Q2 + Q3 + Q4`. -/
theorem square_mem_cycles_two {c : (square k).cells 2 → k} :
    c ∈ (square k).cycles 2 ↔ ∀ j : Fin 4, c j = c (0 : Fin 4) := by
  rw [CellComplex.mem_cycles_succ, square_bd_one]
  constructor
  · intro h j
    have h1 : c (1 : Fin 4) - c (0 : Fin 4) = 0 := congr_fun h (1 : Fin 4)
    have h2 : c (2 : Fin 4) - c (1 : Fin 4) = 0 := congr_fun h (2 : Fin 4)
    have h3 : c (3 : Fin 4) - c (2 : Fin 4) = 0 := congr_fun h (3 : Fin 4)
    fin_cases j
    · rfl
    · exact sub_eq_zero.mp h1
    · change c (2 : Fin 4) = c (0 : Fin 4)
      linear_combination h2 + h1
    · change c (3 : Fin 4) = c (0 : Fin 4)
      linear_combination h3 + h2 + h1
  · intro h
    refine funext fun (j : Fin 4) ↦ ?_
    fin_cases j
    · exact sub_eq_zero.mpr (h 3).symm
    · exact sub_eq_zero.mpr (h 1)
    · exact sub_eq_zero.mpr ((h 2).trans (h 1).symm)
    · exact sub_eq_zero.mpr ((h 3).trans (h 2).symm)

/-- Every `0`-chain is a boundary. -/
theorem square_boundaries_zero : (square k).boundaries 0 = ⊤ := by
  rw [eq_top_iff]
  intro c _
  refine LinearMap.mem_range.mpr ⟨(![-c (0 : Fin 1), 0, 0, 0] : Fin 4 → k),
    funext fun (j : Fin 1) ↦ ?_⟩
  rw [square_bd_zero_apply, Fin.fin_one_eq_zero j]
  simp

/-- Every `1`-cycle `z` is a boundary, of the chain
`z(a) Q1 + (z(a) + z(b)) Q2 + (z(a) + z(b) + z(a')) Q3`. -/
theorem square_cycles_one_le : (square k).cycles 1 ≤ (square k).boundaries 1 := by
  intro z hz
  rw [square_mem_cycles_one] at hz
  refine LinearMap.mem_range.mpr ⟨(![z (0 : Fin 4), z (0 : Fin 4) + z (1 : Fin 4),
    z (0 : Fin 4) + z (1 : Fin 4) + z (2 : Fin 4), 0] : Fin 4 → k), ?_⟩
  rw [square_bd_one]
  refine funext fun (j : Fin 4) ↦ ?_
  fin_cases j
  · simp
  · simp
  · simp
  · simp only [Fin.reduceFinMk, Matrix.cons_val, zero_sub]
    linear_combination -hz

/-- There are no `2`-boundaries. -/
theorem square_boundaries_two : (square k).boundaries 2 = ⊥ :=
  boundaries_eq_bot_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- `H_0(N, L) = 0` for the square. -/
theorem subsingleton_square_H_zero : Subsingleton ((square k).H 0) :=
  subsingleton_H_of_le (by rw [square_boundaries_zero]; exact le_top)

/-- `H_1(N, L) = 0` for the square. -/
theorem subsingleton_square_H_one : Subsingleton ((square k).H 1) :=
  subsingleton_H_of_le square_cycles_one_le

/-- `H_n(N, L) = 0` for `n ≥ 3`: there are no cells. -/
theorem subsingleton_square_H_add_three (n : ℕ) : Subsingleton ((square k).H (n + 3)) :=
  subsingleton_H_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- The fundamental cycle `Q1 + Q2 + Q3 + Q4`. -/
def squareCycle : (square k).cycles 2 :=
  ⟨fun _ ↦ 1, square_mem_cycles_two.mpr fun _ ↦ rfl⟩

/-- Evaluation of a `2`-cycle at `Q1`. -/
def squareEval : (square k).cycles 2 →ₗ[k] k :=
  (LinearMap.proj (R := k) (φ := fun _ : (square k).cells 2 ↦ k) (0 : Fin 4)) ∘ₗ
    ((square k).cycles 2).subtype

/-- `squareEval` is the coefficient of `Q1`. -/
theorem squareEval_apply (z : (square k).cycles 2) : squareEval z = z.1 (0 : Fin 4) :=
  rfl

/-- Every `a ∈ k` is the coefficient of `Q1` of the cycle `a (Q1 + Q2 + Q3 + Q4)`. -/
theorem squareEval_surjective : Function.Surjective (squareEval (k := k)) := fun a ↦
  ⟨⟨fun _ ↦ a, square_mem_cycles_two.mpr fun _ ↦ rfl⟩, rfl⟩

/-- A `2`-cycle vanishing at `Q1` is zero, which is the only `2`-boundary. -/
theorem squareEval_eq_zero_iff (z : (square k).cycles 2) :
    squareEval z = 0 ↔ (z : (square k).cells 2 → k) ∈ (square k).boundaries 2 := by
  rw [square_boundaries_two, Submodule.mem_bot, squareEval_apply]
  have hz := square_mem_cycles_two.mp z.2
  constructor
  · intro h
    exact funext fun (j : Fin 4) ↦ (hz j).trans h
  · intro h
    rw [h]
    rfl

/-- `H_2(N, L) ≅ k` for the square, over any commutative ring, by the coefficient of `Q1`. -/
noncomputable def squareH2Equiv : (square k).H 2 ≃ₗ[k] k :=
  hEquivOfSurjective squareEval squareEval_surjective squareEval_eq_zero_iff

/-- The class of `Q1 + Q2 + Q3 + Q4` corresponds to `1 ∈ k`. -/
theorem squareH2Equiv_cycle :
    squareH2Equiv (k := k) (Submodule.Quotient.mk squareCycle) = 1 :=
  rfl

/-- `H_2(N, L)` is spanned by the class of `Q1 + Q2 + Q3 + Q4`. -/
theorem square_H_two_eq_smul (x : (square k).H 2) :
    x = squareH2Equiv (k := k) x • Submodule.Quotient.mk squareCycle := by
  apply squareH2Equiv.injective
  rw [map_smul, squareH2Equiv_cycle, smul_eq_mul, mul_one]

/-- `H_*(N, L)` of the square has ranks `(0, 0, 1)` over any commutative ring
(the example after Proposition `prop:z2_reconstruction`). -/
theorem square_ranks :
    Subsingleton ((square k).H 0) ∧ Subsingleton ((square k).H 1) ∧
      Nonempty ((square k).H 2 ≃ₗ[k] k) ∧ ∀ n, Subsingleton ((square k).H (n + 3)) :=
  ⟨subsingleton_square_H_zero, subsingleton_square_H_one, ⟨squareH2Equiv⟩,
    subsingleton_square_H_add_three⟩

/-- `-I` acts by `+1` on `H_2(N, L)`: it permutes the squares, so it fixes the constant
cycles. -/
theorem squareInvolution_mapH_two (x : (square k).H 2) : (squareInvolution k).mapH 2 x = x := by
  induction x using Submodule.Quotient.induction_on with
  | H z =>
    rw [CellComplex.ChainMap.mapH_mk]
    congr 1
    apply Subtype.ext
    have hz := square_mem_cycles_two.mp z.2
    exact funext fun (e : Fin 4) ↦ (hz (halfTurn e)).trans (hz e).symm

/-- `-I` acts by `+1` on `H_n(N, L)` for every `n` (the example after
Proposition `prop:z2_reconstruction`). -/
theorem squareInvolution_mapH (n : ℕ) (x : (square k).H n) :
    (squareInvolution k).mapH n x = x := by
  rcases n with _ | _ | _ | n
  · haveI := subsingleton_square_H_zero (k := k)
    exact Subsingleton.elim _ _
  · haveI := subsingleton_square_H_one (k := k)
    exact Subsingleton.elim _ _
  · exact squareInvolution_mapH_two x
  · haveI := subsingleton_square_H_add_three (k := k) n
    exact Subsingleton.elim _ _

/-- `-I` acts by `+1` on Mathlib's homology of `C_*(N, L)` in every degree
(the example after Proposition `prop:z2_reconstruction`). -/
theorem squareInvolution_homologyMap (n : ℕ) (x : (square k).toChainComplex.homology n) :
    (squareInvolution k).homologyMap n x = x :=
  homologyMap_eq_self (squareInvolution_mapH n) x

/-- If `2` is invertible, the `-1` eigen-subcomplex of `-I` on `C_*(N, L)` is acyclic: the `-1`
eigenspace is zero (the example after Proposition
`prop:z2_reconstruction`). -/
theorem square_minus_acyclic [Invertible (2 : k)] (n : ℕ) :
    Subsingleton ((minusSubcomplex (squareInvolution k).toHom).homology n) :=
  subsingleton_homology_minus
    (CellComplex.ChainMap.toHom_comp_self (squareInvolution_comp_self k))
    (squareInvolution_homologyMap n)

/-- If `2` is invertible, the `+1` eigen-subcomplex of `-I` has the homology of `(N, L)`, through
the inclusion: it is the trivial component, of ranks `(0, 0, 1)`. -/
noncomputable def squarePlusEquiv [Invertible (2 : k)] (n : ℕ) :
    (plusSubcomplex (squareInvolution k).toHom).homology n ≃ₗ[k]
      (square k).toChainComplex.homology n :=
  homologyPlusEquiv (CellComplex.ChainMap.toHom_comp_self (squareInvolution_comp_self k))
    (squareInvolution_homologyMap n)

/-- If `2` is invertible, `H_2` of the `+1` eigen-subcomplex of `-I` is `k`. -/
noncomputable def squarePlusH2Equiv [Invertible (2 : k)] :
    (plusSubcomplex (squareInvolution k).toHom).homology 2 ≃ₗ[k] k :=
  (squarePlusEquiv 2).trans (((square k).homologyEquiv 2).trans squareH2Equiv)

end Square

/-! ### (c) The twisted and the untwisted orbit-space complexes of the square -/

section Twisted

variable (k : Type u) [CommRing k]

/-- The number of free orbits of cells of the square in each degree: `[a], [b]` and
`[Q1], [Q2]`. The fixed vertex `O` lies in `N^G` and is dropped. -/
def squareRepDim : ℕ → ℕ
  | 1 => 2
  | 2 => 2
  | _ => 0

/-- The representatives: `a, b` among the edges and `Q1, Q2` among the squares, both
`0, 1 : Fin 4`. -/
def squareRep : Fin 2 → Fin 4 := ![0, 1]

/-- `a, b` (and `Q1, Q2`) are a set of representatives of the free orbits of the half-turn, in the
sense of `SignChains`. -/
theorem isRepresentatives_squareRep :
    IsRepresentatives halfTurn (Finset.univ.image squareRep) :=
  ⟨by decide, by decide⟩

/-- The twisted complex of the square: the cellular chains of `(N/G, (L ∪ N^G)/G)` with
coefficients in the sign representation of the double cover, in the coordinates of the
representatives `{a, b}` and `{Q1, Q2}`. The incidences are `twInc`,
`[r : q] - [r : g q]` (`antiChainsEquiv_comp_bd` of `SignChains`), so `∂[Q1] = [a] - [b]` and
`∂[Q2] = [a] + [b]`; the fixed vertex `O` is dropped. Proposition `prop:z2_reconstruction`. -/
def twisted : CellComplex.{u, 0} k where
  cells n := Fin (squareRepDim n)
  inc
    | 1 => fun r q ↦ twInc (squareInc k) halfTurn (squareRep r) (squareRep q)
    | _ => fun _ _ ↦ 0
  bd_bd n c := by
    rcases n with _ | _ | n <;> ext j <;> simp

/-- The untwisted orbit-space complex of the square relative to the fixed cell: the free orbits
`[a], [b]` and `[Q1], [Q2]` with the orbit-space incidences `[r : q] + [r : g q]`
(`IsRepresentatives.bd_apply_symChains` of `SignChains`, `orbitInc` of `OrbitChains`), so
`∂[Q1] = [a] - [b]` and `∂[Q2] = [b] - [a]`. It is the orbit complex with the fixed vertex `[O]`
removed, the chains of `(N/G, (L ∪ N^G)/G)` with untwisted coefficients
(the example after Proposition `prop:z2_reconstruction`). -/
def untwisted : CellComplex.{u, 0} k where
  cells n := Fin (squareRepDim n)
  inc
    | 1 => fun r q ↦
      squareInc k (squareRep r) (squareRep q) + squareInc k (squareRep r) (halfTurn (squareRep q))
    | _ => fun _ _ ↦ 0
  bd_bd n c := by
    rcases n with _ | _ | n <;> ext j <;> simp

variable {k}

/-- The boundary of the twisted complex:
`∂(c₁ [Q1] + c₂ [Q2]) = (c₁ + c₂) [a] + (c₂ - c₁) [b]`. -/
theorem twisted_bd_one (c : (twisted k).cells 2 → k) :
    bd ((twisted k).inc 1) c =
      ![c (0 : Fin 2) + c (1 : Fin 2), c (1 : Fin 2) - c (0 : Fin 2)] := by
  refine funext fun (q : Fin 2) ↦ ?_
  change ∑ r : Fin 2, c r * twInc (squareInc k) halfTurn (squareRep r) (squareRep q) = _
  fin_cases q
  · simp [Fin.sum_univ_two, twInc, squareInc, halfTurn, squareRep]
  · simp [Fin.sum_univ_two, twInc, squareInc, halfTurn, squareRep]
    ring

/-- The boundary of the untwisted complex:
`∂(c₁ [Q1] + c₂ [Q2]) = (c₁ - c₂) [a] + (c₂ - c₁) [b]`. -/
theorem untwisted_bd_one (c : (untwisted k).cells 2 → k) :
    bd ((untwisted k).inc 1) c =
      ![c (0 : Fin 2) - c (1 : Fin 2), c (1 : Fin 2) - c (0 : Fin 2)] := by
  refine funext fun (q : Fin 2) ↦ ?_
  change ∑ r : Fin 2, c r * (squareInc k (squareRep r) (squareRep q) +
    squareInc k (squareRep r) (halfTurn (squareRep q))) = _
  fin_cases q <;> simp [Fin.sum_univ_two, squareInc, halfTurn, squareRep] <;> ring

/-- If `2` is invertible, the twisted complex of the square is acyclic
(the example after Proposition `prop:z2_reconstruction`: the `-1`
eigenspace is zero). -/
theorem twisted_acyclic [Invertible (2 : k)] (n : ℕ) : Subsingleton ((twisted k).H n) := by
  have h2 : (⅟2 : k) * 2 = 1 := invOf_mul_self 2
  rcases n with _ | _ | _ | n
  · exact subsingleton_H_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)
  · refine subsingleton_H_of_le fun z _ ↦ LinearMap.mem_range.mpr
      ⟨(![⅟2 * (z (0 : Fin 2) - z (1 : Fin 2)), ⅟2 * (z (0 : Fin 2) + z (1 : Fin 2))] :
        Fin 2 → k), ?_⟩
    rw [twisted_bd_one]
    refine funext fun (q : Fin 2) ↦ ?_
    fin_cases q
    · change ⅟2 * (z (0 : Fin 2) - z (1 : Fin 2)) + ⅟2 * (z (0 : Fin 2) + z (1 : Fin 2)) =
        z (0 : Fin 2)
      linear_combination z (0 : Fin 2) * h2
    · change ⅟2 * (z (0 : Fin 2) + z (1 : Fin 2)) - ⅟2 * (z (0 : Fin 2) - z (1 : Fin 2)) =
        z (1 : Fin 2)
      linear_combination z (1 : Fin 2) * h2
  · refine subsingleton_H_of_le fun c hc ↦ ?_
    rw [CellComplex.mem_cycles_succ, twisted_bd_one] at hc
    have h0 : c (0 : Fin 2) + c (1 : Fin 2) = 0 := congr_fun hc (0 : Fin 2)
    have h1 : c (1 : Fin 2) - c (0 : Fin 2) = 0 := congr_fun hc (1 : Fin 2)
    have hc1 : c (1 : Fin 2) = 0 := by
      linear_combination (⅟2 : k) * h0 + (⅟2 : k) * h1 - c (1 : Fin 2) * h2
    have hc0 : c (0 : Fin 2) = 0 := by linear_combination h0 - hc1
    have hc' : c = 0 := funext fun (j : Fin 2) ↦ by
      fin_cases j
      · exact hc0
      · exact hc1
    rw [hc']
    exact zero_mem _
  · exact subsingleton_H_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- Every `1`-chain of the twisted complex is a cycle: there are no `0`-cells. -/
theorem twisted_mem_cycles_one (z : (twisted k).cells 1 → k) : z ∈ (twisted k).cycles 1 :=
  funext fun (j : Fin 0) ↦ j.elim0

/-- The sum of the coefficients of a `1`-cycle of the twisted complex, modulo `2`. -/
def twistedEval : (twisted k).cycles 1 →ₗ[k] k ⧸ Ideal.span {(2 : k)} :=
  (Ideal.span {(2 : k)}).mkQ ∘ₗ
    ((LinearMap.proj (R := k) (φ := fun _ : (twisted k).cells 1 ↦ k) (0 : Fin 2)) +
      (LinearMap.proj (R := k) (φ := fun _ : (twisted k).cells 1 ↦ k) (1 : Fin 2))) ∘ₗ
    ((twisted k).cycles 1).subtype

/-- `twistedEval` is the sum of the coefficients, modulo `2`. -/
theorem twistedEval_apply (z : (twisted k).cycles 1) :
    twistedEval z = Ideal.Quotient.mk _ (z.1 (0 : Fin 2) + z.1 (1 : Fin 2)) :=
  rfl

/-- `twistedEval` is surjective, since every `1`-chain is a cycle. -/
theorem twistedEval_surjective : Function.Surjective (twistedEval (k := k)) := by
  intro m
  obtain ⟨a, rfl⟩ := Ideal.Quotient.mk_surjective m
  exact ⟨⟨(![a, 0] : Fin 2 → k), twisted_mem_cycles_one _⟩, by
    rw [twistedEval_apply]
    congr 1
    change a + 0 = a
    rw [add_zero]⟩

/-- A `1`-chain `x [a] + y [b]` of the twisted complex is a boundary if and only if `x + y` is
divisible by `2`. -/
theorem twistedEval_eq_zero_iff (z : (twisted k).cycles 1) :
    twistedEval z = 0 ↔ (z : (twisted k).cells 1 → k) ∈ (twisted k).boundaries 1 := by
  rw [twistedEval_apply, Ideal.Quotient.eq_zero_iff_mem, Ideal.mem_span_singleton']
  constructor
  · rintro ⟨a, ha⟩
    refine LinearMap.mem_range.mpr ⟨(![z.1 (0 : Fin 2) - a, a] : Fin 2 → k), ?_⟩
    rw [twisted_bd_one]
    refine funext fun (q : Fin 2) ↦ ?_
    fin_cases q
    · change z.1 (0 : Fin 2) - a + a = z.1 (0 : Fin 2)
      ring
    · change a - (z.1 (0 : Fin 2) - a) = z.1 (1 : Fin 2)
      linear_combination ha
  · intro hz
    obtain ⟨c, hc⟩ := LinearMap.mem_range.mp hz
    rw [twisted_bd_one] at hc
    refine ⟨c (1 : Fin 2), ?_⟩
    have h0 : c (0 : Fin 2) + c (1 : Fin 2) = z.1 (0 : Fin 2) := congr_fun hc (0 : Fin 2)
    have h1 : c (1 : Fin 2) - c (0 : Fin 2) = z.1 (1 : Fin 2) := congr_fun hc (1 : Fin 2)
    linear_combination h0 + h1

/-- Over any commutative ring, `H_1` of the twisted complex of the square is `k ⧸ 2k`. -/
noncomputable def twistedH1Equiv : (twisted k).H 1 ≃ₗ[k] k ⧸ Ideal.span {(2 : k)} :=
  hEquivOfSurjective twistedEval twistedEval_surjective twistedEval_eq_zero_iff

/-- `H_1` of the twisted complex vanishes if and only if `2` is a unit, so the hypothesis of
`twisted_acyclic` cannot be weakened. -/
theorem subsingleton_twisted_H_one_iff : Subsingleton ((twisted k).H 1) ↔ IsUnit (2 : k) := by
  rw [twistedH1Equiv.toEquiv.subsingleton_congr, Submodule.Quotient.subsingleton_iff,
    Ideal.span_singleton_eq_top]

/-- Over `ℤ`, `H_1` of the twisted complex of the square is `ℤ/2ℤ`. -/
noncomputable def twistedH1EquivInt : (twisted ℤ).H 1 ≃ₗ[ℤ] ZMod 2 :=
  twistedH1Equiv.trans (Int.quotientSpanEquivZMod 2).toAddEquiv.toIntLinearEquiv

/-- Every `1`-chain of the untwisted complex is a cycle: there are no `0`-cells. -/
theorem untwisted_mem_cycles_one (z : (untwisted k).cells 1 → k) :
    z ∈ (untwisted k).cycles 1 :=
  funext fun (j : Fin 0) ↦ j.elim0

/-- The `2`-cycles of the untwisted complex are the multiples of `[Q1] + [Q2]`. -/
theorem untwisted_mem_cycles_two {c : (untwisted k).cells 2 → k} :
    c ∈ (untwisted k).cycles 2 ↔ c (1 : Fin 2) = c (0 : Fin 2) := by
  rw [CellComplex.mem_cycles_succ, untwisted_bd_one]
  constructor
  · intro h
    exact sub_eq_zero.mp (congr_fun h (1 : Fin 2) : c (1 : Fin 2) - c (0 : Fin 2) = 0)
  · intro h
    refine funext fun (q : Fin 2) ↦ ?_
    fin_cases q
    · exact sub_eq_zero.mpr h.symm
    · exact sub_eq_zero.mpr h

/-- The sum of the coefficients of a `1`-cycle of the untwisted complex. -/
def untwistedEval1 : (untwisted k).cycles 1 →ₗ[k] k :=
  ((LinearMap.proj (R := k) (φ := fun _ : (untwisted k).cells 1 ↦ k) (0 : Fin 2)) +
      (LinearMap.proj (R := k) (φ := fun _ : (untwisted k).cells 1 ↦ k) (1 : Fin 2))) ∘ₗ
    ((untwisted k).cycles 1).subtype

/-- `untwistedEval1` is the sum of the coefficients. -/
theorem untwistedEval1_apply (z : (untwisted k).cycles 1) :
    untwistedEval1 z = z.1 (0 : Fin 2) + z.1 (1 : Fin 2) :=
  rfl

/-- Every `a ∈ k` is the sum of the coefficients of the cycle `a [a]`. -/
theorem untwistedEval1_surjective : Function.Surjective (untwistedEval1 (k := k)) := fun a ↦
  ⟨⟨(![a, 0] : Fin 2 → k), untwisted_mem_cycles_one _⟩, by
    rw [untwistedEval1_apply]
    change a + 0 = a
    rw [add_zero]⟩

/-- A `1`-chain `x [a] + y [b]` of the untwisted complex is a boundary if and only if
`x + y = 0`. -/
theorem untwistedEval1_eq_zero_iff (z : (untwisted k).cycles 1) :
    untwistedEval1 z = 0 ↔ (z : (untwisted k).cells 1 → k) ∈ (untwisted k).boundaries 1 := by
  rw [untwistedEval1_apply]
  constructor
  · intro h
    refine LinearMap.mem_range.mpr ⟨(![z.1 (0 : Fin 2), 0] : Fin 2 → k), ?_⟩
    rw [untwisted_bd_one]
    refine funext fun (q : Fin 2) ↦ ?_
    fin_cases q
    · change z.1 (0 : Fin 2) - 0 = z.1 (0 : Fin 2)
      rw [sub_zero]
    · change 0 - z.1 (0 : Fin 2) = z.1 (1 : Fin 2)
      linear_combination -h
  · intro hz
    obtain ⟨c, hc⟩ := LinearMap.mem_range.mp hz
    rw [untwisted_bd_one] at hc
    have h0 : c (0 : Fin 2) - c (1 : Fin 2) = z.1 (0 : Fin 2) := congr_fun hc (0 : Fin 2)
    have h1 : c (1 : Fin 2) - c (0 : Fin 2) = z.1 (1 : Fin 2) := congr_fun hc (1 : Fin 2)
    linear_combination -h0 - h1

/-- `H_1` of the untwisted complex is `k`, over any commutative ring, by the sum of the
coefficients. -/
noncomputable def untwistedH1Equiv : (untwisted k).H 1 ≃ₗ[k] k :=
  hEquivOfSurjective untwistedEval1 untwistedEval1_surjective untwistedEval1_eq_zero_iff

/-- The cycle `[a]` of the untwisted complex. -/
def untwistedCycle1 : (untwisted k).cycles 1 :=
  ⟨(![1, 0] : Fin 2 → k), untwisted_mem_cycles_one _⟩

/-- The class of `[a]` corresponds to `1 ∈ k`, so it spans `H_1` of the untwisted complex. -/
theorem untwistedH1Equiv_cycle :
    untwistedH1Equiv (k := k) (Submodule.Quotient.mk untwistedCycle1) = 1 :=
  add_zero (1 : k)

/-- Evaluation of a `2`-cycle of the untwisted complex at `[Q1]`. -/
def untwistedEval2 : (untwisted k).cycles 2 →ₗ[k] k :=
  (LinearMap.proj (R := k) (φ := fun _ : (untwisted k).cells 2 ↦ k) (0 : Fin 2)) ∘ₗ
    ((untwisted k).cycles 2).subtype

/-- `untwistedEval2` is the coefficient of `[Q1]`. -/
theorem untwistedEval2_apply (z : (untwisted k).cycles 2) :
    untwistedEval2 z = z.1 (0 : Fin 2) :=
  rfl

/-- Every `a ∈ k` is the coefficient of `[Q1]` of the cycle `a ([Q1] + [Q2])`. -/
theorem untwistedEval2_surjective : Function.Surjective (untwistedEval2 (k := k)) := fun a ↦
  ⟨⟨fun _ ↦ a, untwisted_mem_cycles_two.mpr rfl⟩, rfl⟩

/-- There are no `2`-boundaries in the untwisted complex. -/
theorem untwisted_boundaries_two : (untwisted k).boundaries 2 = ⊥ :=
  boundaries_eq_bot_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- A `2`-cycle of the untwisted complex vanishing at `[Q1]` is zero. -/
theorem untwistedEval2_eq_zero_iff (z : (untwisted k).cycles 2) :
    untwistedEval2 z = 0 ↔ (z : (untwisted k).cells 2 → k) ∈ (untwisted k).boundaries 2 := by
  rw [untwisted_boundaries_two, Submodule.mem_bot, untwistedEval2_apply]
  have hz := untwisted_mem_cycles_two.mp z.2
  constructor
  · intro h
    refine funext fun (j : Fin 2) ↦ ?_
    fin_cases j
    · exact h
    · exact hz.trans h
  · intro h
    rw [h]
    rfl

/-- `H_2` of the untwisted complex is `k`, over any commutative ring, by the coefficient of
`[Q1]`. -/
noncomputable def untwistedH2Equiv : (untwisted k).H 2 ≃ₗ[k] k :=
  hEquivOfSurjective untwistedEval2 untwistedEval2_surjective untwistedEval2_eq_zero_iff

/-- The cycle `[Q1] + [Q2]` of the untwisted complex. -/
def untwistedCycle2 : (untwisted k).cycles 2 :=
  ⟨fun _ ↦ 1, untwisted_mem_cycles_two.mpr rfl⟩

/-- The class of `[Q1] + [Q2]` corresponds to `1 ∈ k`, so it spans `H_2` of the untwisted
complex. -/
theorem untwistedH2Equiv_cycle :
    untwistedH2Equiv (k := k) (Submodule.Quotient.mk untwistedCycle2) = 1 :=
  rfl

/-- `H_0` of the untwisted complex is `0`: there are no `0`-cells. -/
theorem subsingleton_untwisted_H_zero : Subsingleton ((untwisted k).H 0) :=
  subsingleton_H_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- `H_n` of the untwisted complex is `0` for `n ≥ 3`. -/
theorem subsingleton_untwisted_H_add_three (n : ℕ) : Subsingleton ((untwisted k).H (n + 3)) :=
  subsingleton_H_of_isEmpty (show IsEmpty (Fin 0) from inferInstance)

/-- The untwisted orbit-space complex of the square relative to the fixed cell has homology of
ranks `(0, 1, 1)` over any commutative ring (the example after Proposition
`prop:z2_reconstruction`). -/
theorem untwisted_ranks :
    Subsingleton ((untwisted k).H 0) ∧ Nonempty ((untwisted k).H 1 ≃ₗ[k] k) ∧
      Nonempty ((untwisted k).H 2 ≃ₗ[k] k) ∧ ∀ n, Subsingleton ((untwisted k).H (n + 3)) :=
  ⟨subsingleton_untwisted_H_zero, ⟨untwistedH1Equiv⟩, ⟨untwistedH2Equiv⟩,
    subsingleton_untwisted_H_add_three⟩

/-- The untwisted formula fails for `-I` on the square (the example after
Proposition `prop:z2_reconstruction`): over a nontrivial commutative ring, `H_1(N, L) = 0` is
not `M × H_1` of the untwisted complex for any module `M`, since the latter is `k`. So
`H_*(N, L)` is not the sum of the trivial component and the untwisted orbit-space term. -/
theorem untwisted_formula_fails [Nontrivial k] (M : Type*) [AddCommGroup M] [Module k M] :
    IsEmpty ((square k).H 1 ≃ₗ[k] M × (untwisted k).H 1) := by
  refine ⟨fun e ↦ ?_⟩
  haveI := subsingleton_square_H_one (k := k)
  haveI : Subsingleton (M × (untwisted k).H 1) := e.symm.toEquiv.subsingleton
  haveI : Subsingleton ((untwisted k).H 1) := (Prod.mk_right_injective (0 : M)).subsingleton
  haveI : Subsingleton k := untwistedH1Equiv.symm.toEquiv.subsingleton
  exact not_subsingleton k ‹_›

/-- The formula of Proposition `prop:z2_reconstruction` needs `2` invertible: if `2` is not a
unit, `H_1(N, L) = 0` for `-I` on the square is not `M × H_1` of the twisted complex for any
module `M`, since the latter is `k ⧸ 2k ≠ 0` (`subsingleton_twisted_H_one_iff`). Over `ℤ` the
twisted `H_1` is `ZMod 2` (`twistedH1EquivInt`). -/
theorem twisted_formula_fails (h : ¬ IsUnit (2 : k)) (M : Type*) [AddCommGroup M] [Module k M] :
    IsEmpty ((square k).H 1 ≃ₗ[k] M × (twisted k).H 1) := by
  refine ⟨fun e ↦ h ?_⟩
  haveI := subsingleton_square_H_one (k := k)
  haveI : Subsingleton (M × (twisted k).H 1) := e.symm.toEquiv.subsingleton
  haveI : Subsingleton ((twisted k).H 1) := (Prod.mk_right_injective (0 : M)).subsingleton
  exact subsingleton_twisted_H_one_iff.mp ‹_›

end Twisted

end LatentEquivariant.Cellular
