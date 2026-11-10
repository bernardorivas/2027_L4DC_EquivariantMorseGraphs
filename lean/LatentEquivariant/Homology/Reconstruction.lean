import LatentEquivariant.Homology.Basic
import LatentEquivariant.Algebra.OrbitChains
import LatentEquivariant.Algebra.SignChains

/-!
# Reconstruction of the index from the orbit space for a group of order two

Proposition `prop:z2_reconstruction`, assembled from the
splitting of `LatentEquivariant.Homology.Basic` (`homologyEquivEigen`), the transfer of
`LatentEquivariant.Algebra.OrbitChains` (`transferEquiv`) and the twisted coordinates of
`LatentEquivariant.Algebra.SignChains` (`antiChainsEquiv`).

## Model

`G = {1, g}` acts on a finite cell complex `K` over a commutative ring `k`, and `g` permutes the
cells of each degree by an involution `s n` with `[s e : s e'] = [e : e']`
(`CellComplex.CellularInvolution`). This is the unsigned model of
`LatentEquivariant.Algebra.Chains`, which the grid condition of the paper provides after orienting
cells along orbits (`exists_reorient` of `LatentEquivariant.Algebra.Reorient`). A cell is fixed if
`s e = e` and free otherwise. A relative complex `C_*(N, L)` of subcomplexes with `N \ L`
invariant is again a cell complex with an involution (`CellularInvolution.relative`), so the
absolute statements cover the relative ones.

## Main definitions and statements

* `CellularInvolution.Cells`: the cells with the action of `ℤˣ = {1, -1}` in which `-1` acts by
  `s`, the bridge to `OrbitChains`: the invariant chains are `symChains s`
  (`invariantChains_eq_symChains`) and the orbits have one or two cells (`ncard_orbit_of_fixed`,
  `ncard_orbit_of_free`).
* `orbitComplex`: cells the orbits, incidences `orbitInc`. `∂ ∘ ∂ = 0` over any ring, because
  `π_#` is onto (`push_surjective`) and a chain map (`pushChainMap`).
* `twistedComplex R hR`: cells the representatives `R n` of the free pairs, incidences
  `twInc`, `[r : q] - [r : g q]`. `∂ ∘ ∂ = 0` over any ring, because `antiExtend` is an injective
  chain map into `C_*(K)` (`antiExtendChainMap`).
* `plusIso`: the `+1` eigen-subcomplex `C⁺` of the action of `g` is isomorphic to the orbit complex
  if `2` is a unit. `minusIso`: the `-1` eigen-subcomplex `C⁻` is isomorphic to the twisted complex
  if `2` is not a zero divisor, for instance over `ℤ`. Over any commutative ring the twisted
  complex does not depend on the representatives, up to isomorphism (`twistedChainEquiv`;
  `twistedIso` is the form through `minusIso`).
* Exactness over any commutative ring: `ker π_# = range antiExtend`
  (`push_eq_zero_iff_mem_range_antiExtend`), so `0 → C^tw → C_*(K) → C^orb → 0` is a short exact
  sequence of complexes, with `antiExtend` injective and `π_#` onto. Over `ℤ` or `𝔽₂`, where the
  splitting below fails, this still gives a long exact sequence; it is not derived here.
* `reconstructionEquiv` (Proposition `prop:z2_reconstruction`): for `2` invertible,
  `H_n(K) ≃ H_n(orbit complex) × H_n(twisted complex)`, and `reconstructionEquivH` in the form
  `ker ∂ ⧸ range ∂`. The summands are the `+1` and `-1` eigenspaces of `g` on `H_n(K)`
  (`range_reconstructionEquiv_symm_inl`, `range_reconstructionEquiv_symm_inr`).
  `relativeReconstructionEquiv` is the relative form `H_n(N, L) ≃ ...`.
* The index map preserves the summands. For a chain endomorphism `T` commuting with `g`:
  `orbitEnd`, `twistedEnd`, `reconstructionEquiv_naturality` and
  `isNilpotent_homologyMap_iff_orbit_twisted`. For an endomorphism of `H_n(K)` commuting with
  `g_*`, as for an index map defined on homology only: `orbitPart`, `twistedPart`,
  `reconstructionEquiv_apply_end` and `isNilpotent_iff_orbitPart_twistedPart`.
* Untwisting (last claim of the proposition): if the representatives lie in a subcomplex `D`
  whose cells with image in `D` are fixed, the twisted incidences are the orbit incidences
  (`twistedComplex_inc_eq_add`, `twistedComplex_inc_eq_orbitComplex_inc`). If moreover the faces
  of fixed cells are fixed (`FacesFixed`), the twisted complex is isomorphic to
  `untwistedComplex`, the orbit complex relative to the orbits of fixed cells
  (`untwistChainEquiv`, `untwistedReconstructionEquiv`). Under the paper's covering hypothesis,
  `N = D ∪ gD` with `D ∩ gD ⊆ L ∪ N^G`, `FacesFixed` follows (`facesFixed_of_cover`), so the
  paper's form needs nothing more (`nonempty_untwisted`, `nonempty_relative_untwisted`). The
  relative form is stated as `N \ L ⊆ D ∪ gD` and `(N \ L) ∩ D ∩ gD ⊆ N^G`, which for invariant
  `L ⊆ N` is equivalent to the paper's hypothesis (take `(D ∩ N) ∪ L`). Without the covering
  hypothesis, `FacesFixed` cannot be dropped from `untwistChainEquiv`: under the hypotheses of
  `twistedComplex_inc_eq_add` alone the orbits of fixed cells need not form a subcomplex
  (`exists_untwisting_not_isSubcomplex_fixedOrbits`).

## Not formalized

The grid condition and the identification of these combinatorial complexes with spaces: the
orbit complex of the cells of `N` not in `L` with the cellular chains of `(N/G, L/G)`, the twisted
complex with the cellular chains of `(N/G, (L ∪ N^G)/G)` with coefficients in the sign local
system `F^w`, and the untwisted complex with those of `(N/G, (L ∪ N^G)/G)`. The local system `F^w`
itself is not defined. Cellular homology is not compared with singular homology, and the index
map of an equivariant `f` on an index pair, which commutes with `g_*`, is not constructed: it
enters as an endomorphism commuting with `g`. The abelian case after the proposition is not
treated.
-/

universe u v w

namespace LatentEquivariant

open CategoryTheory MulAction

section Push

variable {G k I : Type*} [Group G] [CommRing k] [MulAction G I] [Fintype I]

open scoped Classical in
/-- The map `π_#` of the orbit map is onto, over any commutative ring: an orbit chain `d` is
`π_#` of the chain equal to `d(π(i))` on the chosen cell `i = π(i).out` of each orbit and to `0`
elsewhere. Used for `∂ ∘ ∂ = 0` on the orbit complex (Proposition `prop:z2_reconstruction`). -/
theorem push_surjective : Function.Surjective (push G (k := k) (I := I)) := by
  intro d
  refine ⟨fun i ↦ if (orbitMap G I i).out = i then d (orbitMap G I i) else 0, ?_⟩
  funext q
  rw [push_apply, Finset.sum_congr rfl fun i hi ↦ by rw [(Finset.mem_filter.mp hi).2],
    Finset.sum_ite_eq, if_pos]
  simp

end Push

section Conj

variable {k : Type u} [CommRing k] {ι : Type*} {c : ComplexShape ι}
  {C D : HomologicalComplex (ModuleCat.{w} k) c}

/-- The homology map of the conjugate `e ∘ φ ∘ e⁻¹` (that is, `e.inv ≫ φ ≫ e.hom`) of a chain
endomorphism `φ`, on the image of a class under `e`. Used to transport index maps along the
isomorphisms of Proposition `prop:z2_reconstruction`. -/
theorem homologyMap_conj_apply (e : C ≅ D) (φ : C ⟶ C) (i : ι) (x : C.homology i) :
    (HomologicalComplex.homologyMap (e.inv ≫ φ ≫ e.hom) i).hom
        ((HomologicalComplex.homologyMap e.hom i).hom x) =
      (HomologicalComplex.homologyMap e.hom i).hom
        ((HomologicalComplex.homologyMap φ i).hom x) := by
  rw [homologyMap_hom_comp_apply, homologyMap_hom_comp_apply, e.hom_inv_id_assoc]

end Conj

namespace CellComplex

variable {k : Type u} [CommRing k]

/-- A cellular involution of a cell complex `K`: an involution `s n` of the cells of each degree
that preserves the incidence numbers, `[s e : s e'] = [e : e']`. This is the action of `g` for
`G = {1, g}` in Proposition `prop:z2_reconstruction`, in the unsigned model of
`LatentEquivariant.Algebra.Chains` (the grid condition lets one orient cells so that `g` permutes
them without signs), and the data of `ChainMap.ofInvolutive`. -/
structure CellularInvolution (K : CellComplex.{u, v} k) where
  /-- The permutation of the cells of degree `n` by `g`. -/
  s : ∀ n, K.cells n → K.cells n
  /-- Each `s n` is an involution. -/
  involutive : ∀ n, Function.Involutive (s n)
  /-- The incidence numbers are invariant. -/
  inc_apply : ∀ n e e', K.inc n (s (n + 1) e) (s n e') = K.inc n e e'

namespace CellularInvolution

variable {K : CellComplex.{u, v} k} (σ : CellularInvolution K)

/-! ### The action of a group of order two -/

/-- The cells of degree `n` of `K`, as a type synonym carrying the action of `ℤˣ = {1, -1}` in
which `-1` acts by `s n`. The synonym keeps the action from becoming an instance on `K.cells n`.
It is the bridge from Proposition `prop:z2_reconstruction` to the group actions of
`LatentEquivariant.Algebra.OrbitChains`. -/
@[nolint unusedArguments]
def Cells (_σ : CellularInvolution K) (n : ℕ) : Type v :=
  K.cells n

/-- The cells of each degree are finite. -/
instance (n : ℕ) : Fintype (σ.Cells n) :=
  K.fintype n

/-- `ℤˣ = {1, -1}` acts on the cells, `1` by the identity and `-1` by `s n`. -/
instance (n : ℕ) : SMul ℤˣ (σ.Cells n) :=
  ⟨fun u e ↦ if u = 1 then e else σ.s n e⟩

/-- The action of `u ∈ ℤˣ` on a cell. -/
theorem smul_def (n : ℕ) (u : ℤˣ) (e : σ.Cells n) : u • e = if u = 1 then e else σ.s n e :=
  rfl

/-- The action of `ℤˣ`, the group `G = {1, g}` of Proposition `prop:z2_reconstruction` with
`g = -1`. It is an action because `s n` is an involution. -/
instance (n : ℕ) : MulAction ℤˣ (σ.Cells n) where
  one_smul _ := if_pos rfl
  mul_smul u v e := by
    rcases Int.units_eq_one_or u with rfl | rfl <;>
      rcases Int.units_eq_one_or v with rfl | rfl <;>
      simp [smul_def, σ.involutive n e, neg_units_ne_self (1 : ℤˣ)]

/-- `-1` acts by the involution. -/
theorem neg_one_smul (n : ℕ) (e : σ.Cells n) : (-1 : ℤˣ) • e = σ.s n e := by
  rw [smul_def, if_neg (by decide)]

/-- The incidence numbers are invariant under the action of `ℤˣ`, the hypothesis `hinc` of
`OrbitChains`. -/
theorem inc_smul (n : ℕ) (g : ℤˣ) (i : σ.Cells (n + 1)) (j : σ.Cells n) :
    K.inc n (g • i) (g • j) = K.inc n i j := by
  rcases Int.units_eq_one_or g with rfl | rfl
  · rfl
  · rw [neg_one_smul, neg_one_smul]
    exact σ.inc_apply n i j

/-- The invariant chains of the action of `ℤˣ` are the chains with `c (s e) = c e`. -/
theorem mem_invariantChains_iff {n : ℕ} {c : σ.Cells n → k} :
    c ∈ invariantChains ℤˣ k (σ.Cells n) ↔ ∀ e, c (σ.s n e) = c e := by
  rw [mem_invariantChains]
  constructor
  · intro h e
    rw [← neg_one_smul]
    exact h (-1) e
  · intro h g e
    rcases Int.units_eq_one_or g with rfl | rfl
    · rfl
    · rw [neg_one_smul]
      exact h e

/-- The invariant chains of `OrbitChains` for the action of `ℤˣ` are the `+1` eigenspace
`symChains` of `SignChains` (Proposition `prop:z2_reconstruction`). -/
theorem invariantChains_eq_symChains (n : ℕ) :
    invariantChains ℤˣ k (σ.Cells n) = symChains (σ.s n) := by
  ext c
  exact σ.mem_invariantChains_iff

/-- The orbit of a cell is `{e, g e}`. -/
theorem orbit_eq (n : ℕ) (e : σ.Cells n) : orbit ℤˣ e = {e, (-1 : ℤˣ) • e} := by
  ext e'
  rw [mem_orbit_iff, Set.mem_insert_iff, Set.mem_singleton_iff]
  constructor
  · rintro ⟨g, rfl⟩
    rcases Int.units_eq_one_or g with rfl | rfl
    · exact Or.inl (one_smul _ _)
    · exact Or.inr rfl
  · rintro (rfl | rfl)
    · exact ⟨1, one_smul _ _⟩
    · exact ⟨-1, rfl⟩

/-- The orbit of a fixed cell has one cell. -/
theorem ncard_orbit_of_fixed {n : ℕ} {e : σ.Cells n} (h : σ.s n e = e) :
    (orbit ℤˣ e).ncard = 1 := by
  rw [σ.orbit_eq, neg_one_smul, h, Set.pair_eq_singleton, Set.ncard_singleton]

/-- The orbit of a free cell has two cells. -/
theorem ncard_orbit_of_free {n : ℕ} {e : σ.Cells n} (h : σ.s n e ≠ e) :
    (orbit ℤˣ e).ncard = 2 := by
  rw [σ.orbit_eq]
  refine Set.ncard_pair fun h' ↦ h ?_
  rw [← neg_one_smul, ← h']

/-- Every orbit has one or two cells. -/
theorem ncard_orbit_eq_one_or_two (n : ℕ) (e : σ.Cells n) :
    (orbit ℤˣ e).ncard = 1 ∨ (orbit ℤˣ e).ncard = 2 := by
  by_cases h : σ.s n e = e
  · exact Or.inl (σ.ncard_orbit_of_fixed h)
  · exact Or.inr (σ.ncard_orbit_of_free h)

/-- If `2` is a unit, the cardinality of every orbit is a unit, the hypothesis of
`transferEquiv`. -/
theorem isUnit_ncard_orbit (h2 : IsUnit (2 : k)) (n : ℕ) (e : σ.Cells n) :
    IsUnit ((orbit ℤˣ e).ncard : k) := by
  by_cases h : σ.s n e = e
  · rw [σ.ncard_orbit_of_fixed h, Nat.cast_one]
    exact isUnit_one
  · rw [σ.ncard_orbit_of_free h]
    exact_mod_cast h2

/-! ### The action on chains and its eigen-subcomplexes -/

/-- The action `c ↦ c ∘ s` of `g` on the chains of `K` (`ChainMap.ofInvolutive`), whose `+1` and
`-1` eigen-subcomplexes are the summands of Proposition `prop:z2_reconstruction`. -/
def toChainMap : K.ChainMap K :=
  ChainMap.ofInvolutive σ.s σ.involutive σ.inc_apply

/-- The action of `g` on chains is `ChainMap.ofInvolutive` of the involution of the cells. -/
theorem toChainMap_eq : σ.toChainMap = ChainMap.ofInvolutive σ.s σ.involutive σ.inc_apply :=
  rfl

/-- The action of `g` on chains is an involution. -/
theorem toChainMap_comp_self : σ.toChainMap.comp σ.toChainMap = ChainMap.id K :=
  ChainMap.ofInvolutive_comp_self

/-- In degree `n`, the `+1` eigen-subcomplex of the action of `g` consists of the invariant
chains of `OrbitChains` (`mem_plus_ofInvolutive_iff`). -/
theorem ker_plus_eq (n : ℕ) :
    LinearMap.ker ((σ.toChainMap.toHom - 𝟙 K.toChainComplex).f n).hom =
      invariantChains ℤˣ k (σ.Cells n) := by
  ext c
  exact ChainMap.mem_plus_ofInvolutive_iff.trans σ.mem_invariantChains_iff.symm

/-- In degree `n`, the `-1` eigen-subcomplex of the action of `g` consists of the anti-invariant
chains `antiChains` of `SignChains` (`mem_minus_ofInvolutive_iff`). -/
theorem ker_minus_eq (n : ℕ) :
    LinearMap.ker ((σ.toChainMap.toHom + 𝟙 K.toChainComplex).f n).hom = antiChains (σ.s n) := by
  ext c
  exact ChainMap.mem_minus_ofInvolutive_iff.trans mem_antiChains.symm

/-! ### The orbit complex -/

open scoped Classical in
/-- The orbit complex `C_*(K/G)`: its cells in degree `n` are the orbits of `G = {1, g}` on the
cells of `K`, and its incidence numbers are `orbitInc` of `OrbitChains`,
`[q : p] = ∑_{π(j) = p} [i : j]` for a cell `i` with `π(i) = q`. Over any commutative ring
`∂ ∘ ∂ = 0`, because `π_#` is onto (`push_surjective`) and a chain map (`push_bd`). For the cells
of `N` not in `L`, for a `G`-invariant pair `(N, L)`, its cells are those of `N/G` not in `L/G`,
and it is the cellular chain complex of `(N/G, L/G)` of Proposition `prop:z2_reconstruction`, an
identification that uses the grid condition and is not formalized. -/
noncomputable def orbitComplex : CellComplex.{u, v} k where
  cells n := orbitRel.Quotient ℤˣ (σ.Cells n)
  fintype _ := inferInstance
  inc n := orbitInc ℤˣ (I := σ.Cells (n + 1)) (J := σ.Cells n) (K.inc n)
  bd_bd n d := by
    obtain ⟨c, rfl⟩ := push_surjective d
    rw [← push_bd (σ.inc_smul (n + 1)), ← push_bd (σ.inc_smul n)]
    exact (congrArg (push ℤˣ (I := σ.Cells n) (k := k)) (K.bd_bd n c)).trans (map_zero _)

/-- The incidence numbers of the orbit complex are `orbitInc` of `OrbitChains`. -/
theorem orbitComplex_inc (n : ℕ) :
    σ.orbitComplex.inc n = orbitInc ℤˣ (I := σ.Cells (n + 1)) (J := σ.Cells n) (K.inc n) :=
  rfl

/-- The chain map `π_# : C_*(K) → C_*(K/G)` of the orbit map, `(π_# c)(q) = ∑_{π(i) = q} c(i)`,
over any commutative ring (`push_bd`). -/
noncomputable def pushChainMap : K.ChainMap σ.orbitComplex where
  app n := push ℤˣ (I := σ.Cells n)
  comm n := LinearMap.ext fun c ↦ (push_bd (σ.inc_smul n) c).symm

/-- Two cells have the same orbit if and only if they are equal or swapped by `g`. -/
theorem orbitMap_eq_orbitMap_iff {n : ℕ} {e e' : σ.Cells n} :
    orbitMap ℤˣ (σ.Cells n) e = orbitMap ℤˣ (σ.Cells n) e' ↔ e = e' ∨ e = σ.s n e' := by
  rw [orbitMap_eq_iff]
  constructor
  · rintro ⟨g, rfl⟩
    rcases Int.units_eq_one_or g with rfl | rfl
    · exact Or.inl (one_smul _ _)
    · exact Or.inr (σ.neg_one_smul n e')
  · rintro (rfl | rfl)
    · exact ⟨1, one_smul _ _⟩
    · exact ⟨-1, σ.neg_one_smul n e'⟩

/-- The incidence of orbits at a free face `q` is `[r : q] + [r : g q]`, the orbit-space
incidence of `IsRepresentatives.bd_apply_symChains`. -/
theorem orbitComplex_inc_orbitMap_of_free (n : ℕ) (r : σ.Cells (n + 1)) {q : σ.Cells n}
    (hq : σ.s n q ≠ q) :
    σ.orbitComplex.inc n (orbitMap ℤˣ (σ.Cells (n + 1)) r) (orbitMap ℤˣ (σ.Cells n) q) =
      K.inc n r q + K.inc n r (σ.s n q) := by
  change orbitInc ℤˣ (I := σ.Cells (n + 1)) (J := σ.Cells n) (K.inc n) _ _ = _
  rw [orbitInc_orbitMap (σ.inc_smul n), Finset.sum_filter, Fintype.sum_eq_add q ((-1 : ℤˣ) • q)]
  · rw [if_pos rfl, if_pos (orbitMap_smul _ _), σ.neg_one_smul]
  · rw [σ.neg_one_smul]
    exact Ne.symm hq
  · rintro j ⟨hjq, hjq'⟩
    rw [if_neg]
    rw [σ.orbitMap_eq_orbitMap_iff]
    rintro (h | h)
    · exact hjq h
    · exact hjq' (h.trans (σ.neg_one_smul n q).symm)

/-- The incidence of orbits at a fixed face `q` is `[r : q]`. -/
theorem orbitComplex_inc_orbitMap_of_fixed (n : ℕ) (r : σ.Cells (n + 1)) {q : σ.Cells n}
    (hq : σ.s n q = q) :
    σ.orbitComplex.inc n (orbitMap ℤˣ (σ.Cells (n + 1)) r) (orbitMap ℤˣ (σ.Cells n) q) =
      K.inc n r q := by
  change orbitInc ℤˣ (I := σ.Cells (n + 1)) (J := σ.Cells n) (K.inc n) _ _ = _
  rw [orbitInc_orbitMap (σ.inc_smul n), Finset.sum_filter, Fintype.sum_eq_single q, if_pos rfl]
  intro j hj
  rw [if_neg]
  rw [σ.orbitMap_eq_orbitMap_iff, hq]
  exact fun h ↦ hj (h.elim id id)

/-! ### The twisted complex -/

section Twisted

variable (R : ∀ n, Finset (K.cells n)) (hR : ∀ n, IsRepresentatives (σ.s n) (R n))

/-- The twisted complex: its cells in degree `n` are the representatives `R n` of the free pairs
(`IsRepresentatives`: one cell of each pair `{e, g e}` with `g e ≠ e`), and its incidence numbers
are the twisted incidences `twInc`, `[r : q] - [r : g q]`. Over any commutative ring
`∂ ∘ ∂ = 0`, because `antiExtend` is an injective chain map into `C_*(K)`
(`bd_comp_antiExtend`, `IsRepresentatives.antiExtend_injective`). For the cells of `N` not in `L`,
for a `G`-invariant pair `(N, L)`, there are no fixed cells among its cells: the free pairs are
the cells of `N/G` not in `(L ∪ N^G)/G`, which is the "relative to `(L ∪ N^G)/G`" of Proposition
`prop:z2_reconstruction`, and the sign records whether a face is a representative or its image.
So it is the cellular chain complex of `(N/G, (L ∪ N^G)/G)` with coefficients in the sign local
system `F^w`, an identification that uses the grid condition and is not formalized. -/
noncomputable def twistedComplex : CellComplex.{u, v} k where
  cells n := R n
  fintype _ := inferInstance
  inc n r q := twInc (K.inc n) (σ.s n) r q
  bd_bd n a := by
    classical
    apply (hR n).antiExtend_injective (k := k)
    have e1 := LinearMap.congr_fun (bd_comp_antiExtend (R := R (n + 1)) (σ.involutive (n + 1))
      (σ.involutive n) (σ.inc_apply n) (hR n))
      (bd (fun (r : R (n + 1 + 1)) (q : R (n + 1)) ↦ twInc (K.inc (n + 1)) (σ.s (n + 1)) r q) a)
    have e2 := LinearMap.congr_fun (bd_comp_antiExtend (R := R (n + 1 + 1))
      (σ.involutive (n + 1 + 1)) (σ.involutive (n + 1)) (σ.inc_apply (n + 1)) (hR (n + 1))) a
    simp only [LinearMap.comp_apply] at e1 e2
    rw [map_zero, ← e1, ← e2, K.bd_bd]

/-- The incidence numbers of the twisted complex are `[r : q] - [r : g q]` (`twInc`). -/
theorem twistedComplex_inc (n : ℕ) (r : R (n + 1)) (q : R n) :
    (σ.twistedComplex R hR).inc n r q = K.inc n r q - K.inc n r (σ.s n q) :=
  rfl

open scoped Classical in
/-- The chain map `antiExtend` from the twisted complex to `C_*(K)`, `a ↦ ∑_{r ∈ R} a(r) (r - g r)`,
over any commutative ring (`bd_comp_antiExtend`). It is injective and its image consists of the
anti-invariant chains that vanish on the fixed cells (`IsRepresentatives.antiExtend_injective`,
`IsRepresentatives.mem_range_antiExtend`). -/
noncomputable def antiExtendChainMap : (σ.twistedComplex R hR).ChainMap K where
  app n := antiExtend (σ.s n) (R n)
  comm n := bd_comp_antiExtend (R := R (n + 1)) (σ.involutive (n + 1)) (σ.involutive n)
    (σ.inc_apply n) (hR n)

open scoped Classical in
/-- `π_#` at the orbit of `e`: `c(e)` if `e` is fixed, `c(e) + c(g e)` if `e` is free. -/
theorem push_apply_orbitMap {n : ℕ} (c : σ.Cells n → k) (e : σ.Cells n) :
    push ℤˣ (k := k) c (orbitMap ℤˣ (σ.Cells n) e) =
      if σ.s n e = e then c e else c e + c (σ.s n e) := by
  rw [push_apply, Finset.sum_filter]
  split_ifs with he
  · rw [Fintype.sum_eq_single e]
    · rw [if_pos rfl]
    · intro j hj
      rw [if_neg]
      rw [σ.orbitMap_eq_orbitMap_iff, he]
      exact fun h ↦ hj (h.elim id id)
  · rw [Fintype.sum_eq_add e (σ.s n e) (Ne.symm he)]
    · rw [if_pos rfl, if_pos (σ.orbitMap_eq_orbitMap_iff.mpr (Or.inr rfl))]
    · rintro j ⟨hje, hjse⟩
      rw [if_neg]
      rw [σ.orbitMap_eq_orbitMap_iff]
      rintro (h | h)
      · exact hje h
      · exact hjse h

open scoped Classical in
/-- The kernel of `π_#`: the anti-invariant chains that vanish on the fixed cells, over any
commutative ring. -/
theorem push_eq_zero_iff (n : ℕ) (c : σ.Cells n → k) :
    push ℤˣ (k := k) c = 0 ↔ (∀ e, c (σ.s n e) = -c e) ∧ ∀ e, σ.s n e = e → c e = 0 := by
  constructor
  · intro h
    have key : ∀ e : σ.Cells n, (if σ.s n e = e then c e else c e + c (σ.s n e)) = 0 :=
      fun e ↦ by rw [← σ.push_apply_orbitMap c e, h, Pi.zero_apply]
    refine ⟨fun e ↦ ?_, fun e he ↦ ?_⟩
    · by_cases he : σ.s n e = e
      · have h1 := key e
        rw [if_pos he] at h1
        rw [he, h1, neg_zero]
      · have h1 := key e
        rw [if_neg he] at h1
        linear_combination h1
    · have h1 := key e
      rwa [if_pos he] at h1
  · rintro ⟨hc, hfix⟩
    funext q
    obtain ⟨e, rfl⟩ := orbitMap_surjective q
    rw [σ.push_apply_orbitMap c e, Pi.zero_apply]
    split_ifs with he
    · exact hfix e he
    · rw [hc e, add_neg_cancel]

include hR in
open scoped Classical in
/-- Exactness in the middle, over any commutative ring: `ker π_# = range antiExtend`. With
`antiExtend` injective (`IsRepresentatives.antiExtend_injective`) and `π_#` onto
(`push_surjective`), `0 → C^tw → C_*(K) → C^orb → 0` is a short exact sequence of complexes, with
no hypothesis on `2` (Proposition `prop:z2_reconstruction`). -/
theorem push_eq_zero_iff_mem_range_antiExtend (n : ℕ) (c : K.cells n → k) :
    push ℤˣ (k := k) (I := σ.Cells n) c = 0 ↔
      c ∈ LinearMap.range (antiExtend (k := k) (σ.s n) (R n)) := by
  rw [(hR n).mem_range_antiExtend (σ.involutive n), mem_antiChains]
  exact σ.push_eq_zero_iff n c

end Twisted

/-- A choice of representatives of the free pairs in each degree (`exists_isRepresentatives`). -/
noncomputable def reps (n : ℕ) : Finset (K.cells n) :=
  (exists_isRepresentatives (σ.involutive n)).choose

/-- The chosen cells `reps n` are representatives of the free pairs. -/
theorem isRepresentatives_reps (n : ℕ) : IsRepresentatives (σ.s n) (σ.reps n) :=
  (exists_isRepresentatives (σ.involutive n)).choose_spec

/-! ### The eigen-subcomplexes as the orbit and twisted complexes -/

/-- The transfer in degree `n`: `π_#` restricted to the `+1` eigen-subcomplex, a linear
equivalence onto the orbit chains if `2` is a unit (`transferEquiv`). -/
noncomputable def plusEquiv (h2 : IsUnit (2 : k)) (n : ℕ) :
    (plusSubcomplex σ.toChainMap.toHom).X n ≃ₗ[k] σ.orbitComplex.toChainComplex.X n :=
  (LinearEquiv.ofEq _ _ (σ.ker_plus_eq n)).trans
    (transferEquiv (I := σ.Cells n) (σ.isUnit_ncard_orbit h2 n))

/-- `plusEquiv` is `π_#`. -/
theorem plusEquiv_apply (h2 : IsUnit (2 : k)) (n : ℕ)
    (x : (plusSubcomplex σ.toChainMap.toHom).X n) :
    σ.plusEquiv h2 n x = push ℤˣ (I := σ.Cells n) (x : K.cells n → k) :=
  rfl

/-- The inverse of `plusEquiv` is `orbitLift`, `d ↦ (e ↦ |G e|⁻¹ d([e]))`. -/
theorem coe_plusEquiv_symm_apply (h2 : IsUnit (2 : k)) (n : ℕ)
    (d : σ.orbitComplex.toChainComplex.X n) :
    ((σ.plusEquiv h2 n).symm d : K.cells n → k) = orbitLift (I := σ.Cells n) d :=
  rfl

/-- The `+1` eigen-subcomplex `C⁺` of the action of `g` on `C_*(K)` is isomorphic to the orbit
complex if `2` is a unit, through `π_#`, which sends `e + g e` to `2 [e]` and a fixed cell `e` to
`[e]` (Proposition `prop:z2_reconstruction`, first summand, chain level). -/
noncomputable def plusIso (h2 : IsUnit (2 : k)) :
    plusSubcomplex σ.toChainMap.toHom ≅ σ.orbitComplex.toChainComplex :=
  isoOfLinearEquivs (σ.plusEquiv h2) (by
    rintro _ n rfl
    refine LinearMap.ext fun x ↦ ?_
    rw [LinearMap.comp_apply, LinearMap.comp_apply, toChainComplex_d_hom]
    change bd (σ.orbitComplex.inc n) (push ℤˣ (I := σ.Cells (n + 1)) (x : K.cells (n + 1) → k)) =
      push ℤˣ (I := σ.Cells n) ((K.toChainComplex.d (n + 1) n).hom x)
    rw [toChainComplex_d_hom]
    exact (push_bd (σ.inc_smul n) _).symm)

section Minus

variable (R : ∀ n, Finset (K.cells n)) (hR : ∀ n, IsRepresentatives (σ.s n) (R n))

open scoped Classical in
/-- The twisted coordinates in degree `n`: restriction to the representatives, a linear
equivalence from the `-1` eigen-subcomplex onto the chains on `R n` if `2` is not a zero divisor
(`antiChainsEquiv`). -/
noncomputable def minusEquiv (h2 : IsLeftRegular (2 : k)) (n : ℕ) :
    (minusSubcomplex σ.toChainMap.toHom).X n ≃ₗ[k] (σ.twistedComplex R hR).toChainComplex.X n :=
  (LinearEquiv.ofEq _ _ (σ.ker_minus_eq n)).trans (antiChainsEquiv (σ.involutive n) (hR n) h2)

/-- `minusEquiv` is the restriction to the representatives. -/
theorem minusEquiv_apply (h2 : IsLeftRegular (2 : k)) (n : ℕ)
    (x : (minusSubcomplex σ.toChainMap.toHom).X n) (r : R n) :
    σ.minusEquiv R hR h2 n x r = (x : K.cells n → k) r :=
  rfl

open scoped Classical in
/-- The inverse of `minusEquiv` is `antiExtend`, `a ↦ ∑_{r ∈ R} a(r) (r - g r)`. -/
theorem coe_minusEquiv_symm_apply (h2 : IsLeftRegular (2 : k)) (n : ℕ) (a : R n → k) :
    ((σ.minusEquiv R hR h2 n).symm a : K.cells n → k) = antiExtend (σ.s n) (R n) a :=
  rfl

/-- The `-1` eigen-subcomplex `C⁻` of the action of `g` on `C_*(K)` is isomorphic to the twisted
complex if `2` is not a zero divisor of `k`, for instance if `2` is invertible or `k = ℤ`
(Proposition `prop:z2_reconstruction`, second summand, chain level; `antiChainsEquiv_comp_bd`). -/
noncomputable def minusIso (h2 : IsLeftRegular (2 : k)) :
    minusSubcomplex σ.toChainMap.toHom ≅ (σ.twistedComplex R hR).toChainComplex :=
  isoOfLinearEquivs (σ.minusEquiv R hR h2) (by
    classical
    rintro _ n rfl
    refine LinearMap.ext fun x ↦ ?_
    rw [LinearMap.comp_apply, LinearMap.comp_apply, toChainComplex_d_hom]
    have hx : (x : K.cells (n + 1) → k) ∈ antiChains (σ.s (n + 1)) := by
      rw [← σ.ker_minus_eq]
      exact x.2
    have h := LinearMap.congr_fun (antiChainsEquiv_comp_bd h2 (σ.involutive (n + 1))
      (σ.involutive n) (σ.inc_apply n) (hR (n + 1)) (hR n)) ⟨x, hx⟩
    change bd _ (fun (r : R (n + 1)) ↦ (x : K.cells (n + 1) → k) r) =
      fun (r : R n) ↦ (K.toChainComplex.d (n + 1) n).hom x r
    rw [toChainComplex_d_hom]
    exact h.symm)

/-- If `2` is not a zero divisor of `k`, the twisted complex does not depend on the choice of
representatives, up to isomorphism: both choices give complexes isomorphic to the `-1`
eigen-subcomplex (`minusIso`). So the twisted term of Proposition `prop:z2_reconstruction` is well
defined. The hypothesis on `2` is not needed for this: `twistedChainEquiv`. -/
noncomputable def twistedIso (h2 : IsLeftRegular (2 : k)) (R' : ∀ n, Finset (K.cells n))
    (hR' : ∀ n, IsRepresentatives (σ.s n) (R' n)) :
    (σ.twistedComplex R hR).toChainComplex ≅ (σ.twistedComplex R' hR').toChainComplex :=
  (σ.minusIso R hR h2).symm ≪≫ σ.minusIso R' hR' h2

open scoped Classical in
/-- Change of representatives in degree `n`, over any commutative ring: `a ↦ (antiExtend a)|_R'`,
which flips the sign of the coordinate of `r` where `R'` contains `g r` instead of `r`. -/
noncomputable def repEquiv (R' : ∀ n, Finset (K.cells n))
    (hR' : ∀ n, IsRepresentatives (σ.s n) (R' n)) (n : ℕ) : (R n → k) ≃ₗ[k] (R' n → k) :=
  LinearEquiv.ofLinear
    (LinearMap.funLeft k k (fun r : R' n ↦ (r : K.cells n)) ∘ₗ antiExtend (σ.s n) (R n))
    (LinearMap.funLeft k k (fun r : R n ↦ (r : K.cells n)) ∘ₗ antiExtend (σ.s n) (R' n))
    (by
      refine LinearMap.ext fun b ↦ funext fun r' ↦ ?_
      change antiExtend (σ.s n) (R n) (fun r ↦ antiExtend (σ.s n) (R' n) b r) r' = b r'
      rw [(hR n).antiExtend_restrict (σ.involutive n)
        (antiExtend_mem_antiChains (σ.involutive n) b)
        (fun i hi ↦ antiExtend_apply_of_fixed b hi)]
      exact (hR' n).antiExtend_apply_coe b r')
    (by
      refine LinearMap.ext fun a ↦ funext fun r ↦ ?_
      change antiExtend (σ.s n) (R' n) (fun r' ↦ antiExtend (σ.s n) (R n) a r') r = a r
      rw [(hR' n).antiExtend_restrict (σ.involutive n)
        (antiExtend_mem_antiChains (σ.involutive n) a)
        (fun i hi ↦ antiExtend_apply_of_fixed a hi)]
      exact (hR n).antiExtend_apply_coe a r)

open scoped Classical in
/-- The twisted complex does not depend on the choice of representatives, up to isomorphism, over
any commutative ring: `repEquiv` commutes with the boundaries. This strengthens `twistedIso`, which
assumes that `2` is not a zero divisor (Proposition `prop:z2_reconstruction`). -/
noncomputable def twistedChainEquiv (R' : ∀ n, Finset (K.cells n))
    (hR' : ∀ n, IsRepresentatives (σ.s n) (R' n)) :
    (σ.twistedComplex R hR).ChainEquiv (σ.twistedComplex R' hR') where
  app n := σ.repEquiv R hR R' hR' n
  comm n := by
    refine LinearMap.ext fun a ↦ funext fun q' ↦ ?_
    change bd (fun (r : R' (n + 1)) (q : R' n) ↦ twInc (K.inc n) (σ.s n) r q)
        (fun r ↦ antiExtend (σ.s (n + 1)) (R (n + 1)) a r) q' =
      antiExtend (σ.s n) (R n) (bd (fun (r : R (n + 1)) (q : R n) ↦
        twInc (K.inc n) (σ.s n) r q) a) q'.1
    have hb : antiExtend (σ.s (n + 1)) (R' (n + 1))
        (fun r ↦ antiExtend (σ.s (n + 1)) (R (n + 1)) a r) =
        antiExtend (σ.s (n + 1)) (R (n + 1)) a :=
      (hR' (n + 1)).antiExtend_restrict (σ.involutive (n + 1))
        (antiExtend_mem_antiChains (σ.involutive (n + 1)) a)
        (fun i hi ↦ antiExtend_apply_of_fixed a hi)
    have e1 := LinearMap.congr_fun (bd_comp_antiExtend (R := R' (n + 1))
      (σ.involutive (n + 1)) (σ.involutive n) (σ.inc_apply n) (hR' n))
      (fun r ↦ antiExtend (σ.s (n + 1)) (R (n + 1)) a r)
    have e2 := LinearMap.congr_fun (bd_comp_antiExtend (R := R (n + 1))
      (σ.involutive (n + 1)) (σ.involutive n) (σ.inc_apply n) (hR n)) a
    simp only [LinearMap.comp_apply] at e1 e2
    rw [hb, e2] at e1
    have e3 := congr_fun e1 q'.1
    rw [(hR' n).antiExtend_apply_coe] at e3
    exact e3.symm

end Minus

/-! ### Proposition `prop:z2_reconstruction` -/

section Main

variable (R : ∀ n, Finset (K.cells n)) (hR : ∀ n, IsRepresentatives (σ.s n) (R n))
  [Invertible (2 : k)]

/-- **Proposition `prop:z2_reconstruction`.** For an involution `g` of a finite cell complex `K`
that permutes the cells and preserves the incidence numbers, over a commutative ring in which `2`
is invertible, `H_n(K) ≃ H_n(orbit complex) × H_n(twisted complex)` for every `n`. The twisted
complex is built on any choice `R` of representatives of the free pairs. The summands are the
`+1` and `-1` eigenspaces of `g` (`range_reconstructionEquiv_symm_inl`,
`range_reconstructionEquiv_symm_inr`). The map is the splitting `homologyEquivEigen` followed by
the homology maps of `plusIso` and `minusIso`. For the relative form see
`relativeReconstructionEquiv`. -/
noncomputable def reconstructionEquiv (n : ℕ) :
    K.toChainComplex.homology n ≃ₗ[k]
      σ.orbitComplex.toChainComplex.homology n ×
        (σ.twistedComplex R hR).toChainComplex.homology n :=
  (CellComplex.homologyEquivEigen σ.toChainMap_comp_self n).trans
    (LinearEquiv.prodCongr
      (HomologicalComplex.homologyMapIso (σ.plusIso (isUnit_of_invertible 2)) n).toLinearEquiv
      (HomologicalComplex.homologyMapIso (σ.minusIso R hR isLeftRegular_two_of_invertible)
        n).toLinearEquiv)

/-- **Proposition `prop:z2_reconstruction`** for cellular homology in the form
`ker ∂_n ⧸ range ∂_(n + 1)` (`CellComplex.H`):
`H_n(K) ≃ H_n(orbit complex) × H_n(twisted complex)`, for `2` invertible. -/
noncomputable def reconstructionEquivH (n : ℕ) :
    K.H n ≃ₗ[k] σ.orbitComplex.H n × (σ.twistedComplex R hR).H n :=
  (K.homologyEquiv n).symm.trans ((σ.reconstructionEquiv R hR n).trans
    (LinearEquiv.prodCongr (σ.orbitComplex.homologyEquiv n)
      ((σ.twistedComplex R hR).homologyEquiv n)))

/-- The orbit summand enters `H_n(K)` through the inclusion of the `+1` eigen-subcomplex. -/
theorem reconstructionEquiv_symm_inl (n : ℕ) :
    (σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inl k _ _ =
      (HomologicalComplex.homologyMap
          (subcomplexι : plusSubcomplex σ.toChainMap.toHom ⟶ _) n).hom ∘ₗ
        (HomologicalComplex.homologyMapIso (σ.plusIso (isUnit_of_invertible 2))
          n).toLinearEquiv.symm.toLinearMap := by
  ext y
  simp only [reconstructionEquiv, LinearMap.comp_apply, LinearMap.inl_apply, LinearEquiv.coe_coe,
    LinearEquiv.trans_symm, LinearEquiv.trans_apply, LinearEquiv.prodCongr_symm,
    LinearEquiv.prodCongr_apply, CellComplex.homologyEquivEigen,
    LatentEquivariant.homologyEquivEigen]
  rw [homologyEquivProd_symm_apply]
  dsimp only
  simp only [map_zero, add_zero]

/-- The twisted summand enters `H_n(K)` through the inclusion of the `-1` eigen-subcomplex. -/
theorem reconstructionEquiv_symm_inr (n : ℕ) :
    (σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inr k _ _ =
      (HomologicalComplex.homologyMap
          (subcomplexι : minusSubcomplex σ.toChainMap.toHom ⟶ _) n).hom ∘ₗ
        (HomologicalComplex.homologyMapIso (σ.minusIso R hR isLeftRegular_two_of_invertible)
          n).toLinearEquiv.symm.toLinearMap := by
  ext y
  simp only [reconstructionEquiv, LinearMap.comp_apply, LinearMap.inr_apply, LinearEquiv.coe_coe,
    LinearEquiv.trans_symm, LinearEquiv.trans_apply, LinearEquiv.prodCongr_symm,
    LinearEquiv.prodCongr_apply, CellComplex.homologyEquivEigen,
    LatentEquivariant.homologyEquivEigen]
  rw [homologyEquivProd_symm_apply]
  dsimp only
  simp only [map_zero, zero_add]

/-- The orbit summand is the `+1` eigenspace of `g` on `H_n(K)` (Proposition
`prop:z2_reconstruction`: "the summands are the eigenspaces of `g`"). -/
theorem range_reconstructionEquiv_symm_inl (n : ℕ) :
    LinearMap.range ((σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inl k _ _) =
      LinearMap.ker (σ.toChainMap.homologyMap n - 1) := by
  rw [reconstructionEquiv_symm_inl, LinearMap.range_comp_of_range_eq_top _ (LinearEquiv.range _)]
  exact range_homologyMap_plusι (ChainMap.toHom_comp_self σ.toChainMap_comp_self) n

/-- The twisted summand is the `-1` eigenspace of `g` on `H_n(K)` (Proposition
`prop:z2_reconstruction`). -/
theorem range_reconstructionEquiv_symm_inr (n : ℕ) :
    LinearMap.range ((σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inr k _ _) =
      LinearMap.ker (σ.toChainMap.homologyMap n + 1) := by
  rw [reconstructionEquiv_symm_inr, LinearMap.range_comp_of_range_eq_top _ (LinearEquiv.range _)]
  exact range_homologyMap_minusι (ChainMap.toHom_comp_self σ.toChainMap_comp_self) n

end Main

/-! ### The relative form -/

section Relative

variable (N L : ∀ n, Set (K.cells n)) [∀ n, DecidablePred (· ∈ N n)]
  [∀ n, DecidablePred (· ∈ L n)] (hN : K.IsSubcomplex N) (hL : K.IsSubcomplex L)
  (hsNL : ∀ n, ∀ e ∈ N n \ L n, σ.s n e ∈ N n \ L n)

/-- The restriction of a cellular involution to the relative complex `C_*(N, L)`, the cells of
`N` not in `L`, through `restrictCells`. It needs only that `g` preserves the cells of `N` not in
`L`; for `G`-invariant subcomplexes `N` and `L` this is `mapsTo_diff_of_involutive`. -/
def relative : CellularInvolution (K.relative N L hN hL) where
  s := restrictCells σ.s hsNL
  involutive := involutive_restrictCells hsNL σ.involutive
  inc_apply := inc_restrictCells hsNL σ.inc_apply

/-- The restricted involution acts on a relative cell as `g` acts on the cell of `K`. -/
theorem val_relative_s_apply (n : ℕ) (e : (K.relative N L hN hL).cells n) :
    Subtype.val ((σ.relative N L hN hL hsNL).s n e) = σ.s n (Subtype.val e) :=
  rfl

variable (R : ∀ n, Finset ((K.relative N L hN hL).cells n))
  (hR : ∀ n, IsRepresentatives ((σ.relative N L hN hL hsNL).s n) (R n)) [Invertible (2 : k)]

/-- **Proposition `prop:z2_reconstruction`**, relative form: for subcomplexes `N` and `L` of `K`
whose difference `N \ L` is invariant under `g`, and `2` invertible,
`H_n(N, L) ≃ H_n(orbit complex) × H_n(twisted complex)` of the relative cells. For `G`-invariant
subcomplexes `L ⊆ N`, the paper's setting, the orbit complex of the cells of `N` not in `L` has
for cells the cells of `N/G` not in `L/G`, so it is the cellular chain complex of `(N/G, L/G)`.
The twisted complex has no fixed cells: its cells correspond to the cells of `N/G` not in
`(L ∪ N^G)/G`, which is the "relative to `(L ∪ N^G)/G`" of the proposition, with the twisted
incidences of the sign local system `F^w`. Both identifications with spaces use the grid
condition and are not formalized. Assuming only that `N \ L` is invariant adds no relative
complexes: `N ∪ gN` and `(L ∪ gL) ∩ (N ∪ gN)` are invariant with the same difference. -/
noncomputable def relativeReconstructionEquiv (n : ℕ) :
    (K.relative N L hN hL).toChainComplex.homology n ≃ₗ[k]
      (σ.relative N L hN hL hsNL).orbitComplex.toChainComplex.homology n ×
        ((σ.relative N L hN hL hsNL).twistedComplex R hR).toChainComplex.homology n :=
  (σ.relative N L hN hL hsNL).reconstructionEquiv R hR n

end Relative

/-! ### The index map preserves the summands -/

section Naturality

variable (R : ∀ n, Finset (K.cells n)) (hR : ∀ n, IsRepresentatives (σ.s n) (R n))
  {T : K.ChainMap K} (hT : T.comp σ.toChainMap = σ.toChainMap.comp T)

/-- The endomorphism of the orbit complex induced by a chain endomorphism `T` of `K` commuting
with `g`: its restriction to the `+1` eigen-subcomplex, transported by `plusIso`. In coordinates
it lifts an orbit chain, applies `T` and pushes to orbits (`orbitEnd_apply`). Proposition
`prop:z2_reconstruction`: the index map preserves the first summand. -/
noncomputable def orbitEnd (h2 : IsUnit (2 : k)) :
    σ.orbitComplex.toChainComplex ⟶ σ.orbitComplex.toChainComplex :=
  (σ.plusIso h2).inv ≫ plusMap T.toHom (ChainMap.toHom_comm hT) ≫ (σ.plusIso h2).hom

/-- The endomorphism of the twisted complex induced by a chain endomorphism `T` of `K` commuting
with `g`: its restriction to the `-1` eigen-subcomplex, transported by `minusIso`. In coordinates
it extends a chain by `antiExtend`, applies `T` and restricts to the representatives
(`twistedEnd_apply`). Proposition `prop:z2_reconstruction`: the index map preserves the second
summand. -/
noncomputable def twistedEnd (h2 : IsLeftRegular (2 : k)) :
    (σ.twistedComplex R hR).toChainComplex ⟶ (σ.twistedComplex R hR).toChainComplex :=
  (σ.minusIso R hR h2).inv ≫ minusMap T.toHom (ChainMap.toHom_comm hT) ≫ (σ.minusIso R hR h2).hom

/-- `orbitEnd` in coordinates: `d ↦ π_# (T (orbitLift d))`. -/
theorem orbitEnd_apply (h2 : IsUnit (2 : k)) (n : ℕ) (d : σ.orbitComplex.cells n → k) :
    ((σ.orbitEnd hT h2).f n).hom d =
      push ℤˣ (I := σ.Cells n) (T.app n (orbitLift (I := σ.Cells n) d)) :=
  rfl

open scoped Classical in
/-- `twistedEnd` in coordinates: `a ↦ T (antiExtend a)` restricted to the representatives. -/
theorem twistedEnd_apply (h2 : IsLeftRegular (2 : k)) (n : ℕ) (a : R n → k) (r : R n) :
    ((σ.twistedEnd R hR hT h2).f n).hom a r = T.app n (antiExtend (σ.s n) (R n) a) r :=
  rfl

variable [Invertible (2 : k)]

/-- **Proposition `prop:z2_reconstruction`**, the index map preserves both summands: under
`reconstructionEquiv`, a chain endomorphism `T` of `K` commuting with `g` acts on
`H_n(orbit complex) × H_n(twisted complex)` by `orbitEnd` and `twistedEnd`
(`homologyEquivEigen_naturality`). -/
theorem reconstructionEquiv_naturality (n : ℕ) (x : K.toChainComplex.homology n) :
    σ.reconstructionEquiv R hR n (T.homologyMap n x) =
      Prod.map (HomologicalComplex.homologyMap (σ.orbitEnd hT (isUnit_of_invertible 2)) n).hom
        (HomologicalComplex.homologyMap
          (σ.twistedEnd R hR hT isLeftRegular_two_of_invertible) n).hom
        (σ.reconstructionEquiv R hR n x) := by
  have hσ := ChainMap.toHom_comp_self σ.toChainMap_comp_self
  have key := homologyEquivEigen_naturality hσ hσ T.toHom (ChainMap.toHom_comm hT) n x
  simp only [reconstructionEquiv, LinearEquiv.trans_apply, CellComplex.homologyEquivEigen,
    ChainMap.homologyMap]
  rw [key]
  refine Prod.ext ?_ ?_
  · simp only [LinearEquiv.prodCongr_apply, Prod.map_fst, Iso.toLinearEquiv_apply,
      HomologicalComplex.homologyMapIso_hom, orbitEnd]
    exact (homologyMap_conj_apply _ _ n _).symm
  · simp only [LinearEquiv.prodCongr_apply, Prod.map_snd, Iso.toLinearEquiv_apply,
      HomologicalComplex.homologyMapIso_hom, twistedEnd]
    exact (homologyMap_conj_apply _ _ n _).symm

/-- The index is trivial exactly when it is trivial on both summands (Proposition
`prop:z2_reconstruction`; over a field the index is trivial if and
only if the index map is nilpotent): for a chain endomorphism `T` of `K` commuting with `g`,
`T_*` is nilpotent on `H_n(K)` if and only if it is nilpotent on `H_n(orbit complex)` and on
`H_n(twisted complex)`. -/
theorem isNilpotent_homologyMap_iff_orbit_twisted (n : ℕ) :
    IsNilpotent (T.homologyMap n) ↔
      IsNilpotent (HomologicalComplex.homologyMap (σ.orbitEnd hT (isUnit_of_invertible 2)) n).hom ∧
        IsNilpotent (HomologicalComplex.homologyMap
          (σ.twistedEnd R hR hT isLeftRegular_two_of_invertible) n).hom := by
  rw [isNilpotent_iff_of_linearEquiv (σ.reconstructionEquiv R hR n)
    (g := (HomologicalComplex.homologyMap (σ.orbitEnd hT (isUnit_of_invertible 2)) n).hom.prodMap
      (HomologicalComplex.homologyMap
        (σ.twistedEnd R hR hT isLeftRegular_two_of_invertible) n).hom)
    (σ.reconstructionEquiv_naturality R hR hT n), isNilpotent_prodMap_iff]

end Naturality

section IndexMap

variable (R : ∀ n, Finset (K.cells n)) (hR : ∀ n, IsRepresentatives (σ.s n) (R n))
  [Invertible (2 : k)] (n : ℕ)

/-- A class fixed by `g` has no component in the twisted summand. -/
theorem reconstructionEquiv_snd_eq_zero {x : K.toChainComplex.homology n}
    (hx : σ.toChainMap.homologyMap n x = x) : (σ.reconstructionEquiv R hR n x).2 = 0 := by
  have hx' : x ∈ LinearMap.range
      ((σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inl k _ _) := by
    rw [range_reconstructionEquiv_symm_inl, LinearMap.mem_ker, LinearMap.sub_apply, hx,
      Module.End.one_apply, sub_self]
  obtain ⟨a, rfl⟩ := hx'
  simp

/-- A class negated by `g` has no component in the orbit summand. -/
theorem reconstructionEquiv_fst_eq_zero {x : K.toChainComplex.homology n}
    (hx : σ.toChainMap.homologyMap n x = -x) : (σ.reconstructionEquiv R hR n x).1 = 0 := by
  have hx' : x ∈ LinearMap.range
      ((σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inr k _ _) := by
    rw [range_reconstructionEquiv_symm_inr, LinearMap.mem_ker, LinearMap.add_apply, hx,
      Module.End.one_apply, neg_add_cancel]
  obtain ⟨a, rfl⟩ := hx'
  simp

variable (F : Module.End k (K.toChainComplex.homology n))

/-- The component on `H_n(orbit complex)` of an endomorphism `F` of `H_n(K)`, through
`reconstructionEquiv`. If `F` commutes with `g_*`, as the index map of an equivariant map does,
`F` acts on the orbit summand by `orbitPart` (`reconstructionEquiv_apply_end`). Proposition
`prop:z2_reconstruction`. -/
noncomputable def orbitPart : Module.End k (σ.orbitComplex.toChainComplex.homology n) :=
  LinearMap.fst k _ _ ∘ₗ (σ.reconstructionEquiv R hR n).toLinearMap ∘ₗ F ∘ₗ
    (σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inl k _ _

/-- The component on `H_n(twisted complex)` of an endomorphism `F` of `H_n(K)`, through
`reconstructionEquiv`. If `F` commutes with `g_*`, `F` acts on the twisted summand by
`twistedPart` (`reconstructionEquiv_apply_end`). Proposition `prop:z2_reconstruction`. -/
noncomputable def twistedPart :
    Module.End k ((σ.twistedComplex R hR).toChainComplex.homology n) :=
  LinearMap.snd k _ _ ∘ₗ (σ.reconstructionEquiv R hR n).toLinearMap ∘ₗ F ∘ₗ
    (σ.reconstructionEquiv R hR n).symm.toLinearMap ∘ₗ LinearMap.inr k _ _

variable {F}

/-- **Proposition `prop:z2_reconstruction`**, the index map preserves both summands, for an index
map given on homology only: an endomorphism `F` of `H_n(K)` that commutes with `g_*` is diagonal
under `reconstructionEquiv`, with components `orbitPart` and `twistedPart`. -/
theorem reconstructionEquiv_apply_end (hF : Commute F (σ.toChainMap.homologyMap n))
    (x : K.toChainComplex.homology n) :
    σ.reconstructionEquiv R hR n (F x) =
      Prod.map (σ.orbitPart R hR n F) (σ.twistedPart R hR n F)
        (σ.reconstructionEquiv R hR n x) := by
  set e := σ.reconstructionEquiv R hR n
  have key : ∀ y, e (F (e.symm y)) =
      Prod.map (σ.orbitPart R hR n F) (σ.twistedPart R hR n F) y := by
    rintro ⟨a, b⟩
    have hu : σ.toChainMap.homologyMap n (e.symm (a, 0)) = e.symm (a, 0) := by
      have h : e.symm (a, 0) ∈ LinearMap.ker (σ.toChainMap.homologyMap n - 1) := by
        rw [← σ.range_reconstructionEquiv_symm_inl R hR]
        exact ⟨a, rfl⟩
      rwa [LinearMap.mem_ker, LinearMap.sub_apply, Module.End.one_apply, sub_eq_zero] at h
    have hv : σ.toChainMap.homologyMap n (e.symm (0, b)) = -e.symm (0, b) := by
      have h : e.symm (0, b) ∈ LinearMap.ker (σ.toChainMap.homologyMap n + 1) := by
        rw [← σ.range_reconstructionEquiv_symm_inr R hR]
        exact ⟨b, rfl⟩
      rwa [LinearMap.mem_ker, LinearMap.add_apply, Module.End.one_apply,
        add_eq_zero_iff_eq_neg] at h
    have hFu : σ.toChainMap.homologyMap n (F (e.symm (a, 0))) = F (e.symm (a, 0)) := by
      rw [← Module.End.mul_apply, ← hF.eq, Module.End.mul_apply, hu]
    have hFv : σ.toChainMap.homologyMap n (F (e.symm (0, b))) = -F (e.symm (0, b)) := by
      rw [← Module.End.mul_apply, ← hF.eq, Module.End.mul_apply, hv, map_neg]
    have h1 := σ.reconstructionEquiv_snd_eq_zero R hR n hFu
    have h2 := σ.reconstructionEquiv_fst_eq_zero R hR n hFv
    have hab : ((a, b) : _ × _) = (a, 0) + (0, b) := by simp
    rw [hab, map_add, map_add, map_add]
    refine Prod.ext ?_ ?_
    · simp [orbitPart, h2, e]
    · simp [twistedPart, h1, e]
  have h := key (e x)
  rwa [e.symm_apply_apply] at h

/-- An endomorphism `F` of `H_n(K)` commuting with `g_*`, `2` invertible, is nilpotent if and only
if its components on the orbit summand and on the twisted summand are nilpotent (Proposition
`prop:z2_reconstruction`). -/
theorem isNilpotent_iff_orbitPart_twistedPart (hF : Commute F (σ.toChainMap.homologyMap n)) :
    IsNilpotent F ↔ IsNilpotent (σ.orbitPart R hR n F) ∧ IsNilpotent (σ.twistedPart R hR n F) := by
  rw [isNilpotent_iff_of_linearEquiv (σ.reconstructionEquiv R hR n)
    (g := (σ.orbitPart R hR n F).prodMap (σ.twistedPart R hR n F))
    (σ.reconstructionEquiv_apply_end R hR n hF), isNilpotent_prodMap_iff]

end IndexMap

/-! ### Untwisting -/

section Untwisting

variable (R : ∀ n, Finset (K.cells n)) (hR : ∀ n, IsRepresentatives (σ.s n) (R n))
  {D : ∀ n, Set (K.cells n)}

/-- Untwisting (Proposition `prop:z2_reconstruction`, last claim): if the representatives lie in
a subcomplex `D` whose cells with image in `D` are fixed, then every twisted incidence is
`[r : q] + [r : g q]`, so the coefficients `F^w` can be replaced by `F`
(`IsRepresentatives.bd_twInc_eq` in every degree). -/
theorem twistedComplex_inc_eq_add (hD : K.IsSubcomplex D)
    (hfix : ∀ n, ∀ e ∈ D n, σ.s n e ∈ D n → σ.s n e = e) (hRD : ∀ n, ↑(R n) ⊆ D n) (n : ℕ)
    (r : R (n + 1)) (q : R n) :
    (σ.twistedComplex R hR).inc n r q = K.inc n r q + K.inc n r (σ.s n q) :=
  twInc_eq_add_of_mem (fun i hi j hj ↦ hD n i j hi hj) (hfix n) (hRD (n + 1) r.2) (hRD n q.2)
    ((hR n).free q q.2)

/-- Untwisting, in terms of the orbit complex: under the hypotheses of
`twistedComplex_inc_eq_add`, the twisted incidence of representatives is the incidence of their
orbits in the orbit complex. So the twisted complex is the orbit complex restricted to the free
orbits (Proposition `prop:z2_reconstruction`, last claim). -/
theorem twistedComplex_inc_eq_orbitComplex_inc (hD : K.IsSubcomplex D)
    (hfix : ∀ n, ∀ e ∈ D n, σ.s n e ∈ D n → σ.s n e = e) (hRD : ∀ n, ↑(R n) ⊆ D n) (n : ℕ)
    (r : R (n + 1)) (q : R n) :
    (σ.twistedComplex R hR).inc n r q =
      σ.orbitComplex.inc n (orbitMap ℤˣ (σ.Cells (n + 1)) r.1) (orbitMap ℤˣ (σ.Cells n) q.1) := by
  rw [σ.twistedComplex_inc_eq_add R hR hD hfix hRD,
    σ.orbitComplex_inc_orbitMap_of_free n r.1 ((hR n).free q q.2)]

/-- The orbits of the fixed cells, the cells of `N^G/G` for the relative cells of a pair
`(N, L)`. -/
def fixedOrbits (n : ℕ) : Set (σ.orbitComplex.cells n) :=
  {p | ∀ e : σ.Cells n, orbitMap ℤˣ (σ.Cells n) e = p → σ.s n e = e}

/-- The orbit of a cell is an orbit of fixed cells if and only if the cell is fixed. -/
theorem orbitMap_mem_fixedOrbits_iff {n : ℕ} {e : σ.Cells n} :
    orbitMap ℤˣ (σ.Cells n) e ∈ σ.fixedOrbits n ↔ σ.s n e = e := by
  refine ⟨fun h ↦ h e rfl, fun h e' he' ↦ ?_⟩
  rcases σ.orbitMap_eq_orbitMap_iff.mp he' with rfl | rfl
  · exact h
  · rw [h]
    exact h

/-- The faces of a fixed cell are fixed. This is a consequence of the grid condition (a cell
mapped to itself is fixed pointwise, and then so is its closure), the hypothesis `hfixface` of
`IsRepresentatives.bd_apply_antiChains_of_free`, and it makes the fixed cells a subcomplex. -/
def FacesFixed : Prop :=
  ∀ n e e', σ.s (n + 1) e = e → K.inc n e e' ≠ 0 → σ.s n e' = e'

/-- All cells of the orbit complex form a subcomplex. -/
theorem isSubcomplex_orbitComplex_univ : σ.orbitComplex.IsSubcomplex fun _ ↦ Set.univ :=
  fun _ _ _ _ _ ↦ Set.mem_univ _

/-- If the faces of fixed cells are fixed, the orbits of the fixed cells form a subcomplex of the
orbit complex. -/
theorem isSubcomplex_fixedOrbits (hF : σ.FacesFixed) :
    σ.orbitComplex.IsSubcomplex σ.fixedOrbits := by
  classical
  intro n p p' hp hpp'
  obtain ⟨e, rfl⟩ := orbitMap_surjective p
  change orbitInc ℤˣ (I := σ.Cells (n + 1)) (J := σ.Cells n) (K.inc n) _ _ ≠ 0 at hpp'
  rw [orbitInc_orbitMap (σ.inc_smul n)] at hpp'
  obtain ⟨j, hj, hj0⟩ := Finset.exists_ne_zero_of_sum_ne_zero hpp'
  rw [← (Finset.mem_filter.mp hj).2]
  exact σ.orbitMap_mem_fixedOrbits_iff.mpr (hF n e j (hp e rfl) hj0)

variable (hF : σ.FacesFixed)

open scoped Classical in
/-- The untwisted complex: the orbit complex relative to the orbits of the fixed cells. For the
relative cells of a `G`-invariant pair `(N, L)` it is the cellular chain complex of
`(N/G, (L ∪ N^G)/G)` with untwisted coefficients `F` (Proposition `prop:z2_reconstruction`, last
claim), an identification that uses the grid condition and is not formalized. -/
noncomputable def untwistedComplex : CellComplex.{u, v} k :=
  σ.orbitComplex.relative (fun _ ↦ Set.univ) σ.fixedOrbits σ.isSubcomplex_orbitComplex_univ
    (σ.isSubcomplex_fixedOrbits hF)

/-- The orbit of a representative, a cell of the untwisted complex. -/
def repOrbit (n : ℕ) (r : (σ.twistedComplex R hR).cells n) :
    (σ.untwistedComplex hF).cells n :=
  ⟨orbitMap ℤˣ (σ.Cells n) r.1, Set.mem_univ _,
    fun h ↦ (hR n).free r.1 r.2 (σ.orbitMap_mem_fixedOrbits_iff.mp h)⟩

/-- Each free orbit contains exactly one representative. -/
theorem repOrbit_bijective (n : ℕ) : Function.Bijective (σ.repOrbit R hR hF n) := by
  refine ⟨fun r r' h ↦ ?_, ?_⟩
  · have h' : orbitMap ℤˣ (σ.Cells n) r.1 = orbitMap ℤˣ (σ.Cells n) r'.1 :=
      congrArg Subtype.val h
    rcases σ.orbitMap_eq_orbitMap_iff.mp h' with h'' | h''
    · exact Subtype.ext h''
    · exact absurd (h'' ▸ r.2) ((hR n).apply_notMem r'.2)
  · rintro ⟨p, -, hp⟩
    obtain ⟨e, rfl⟩ := orbitMap_surjective p
    have he : σ.s n e ≠ e := fun h ↦ hp (σ.orbitMap_mem_fixedOrbits_iff.mpr h)
    by_cases heR : e ∈ R n
    · exact ⟨⟨e, heR⟩, rfl⟩
    · refine ⟨⟨σ.s n e, (hR n).apply_mem he heR⟩, Subtype.ext ?_⟩
      change orbitMap ℤˣ (σ.Cells n) (σ.s n e) = orbitMap ℤˣ (σ.Cells n) e
      rw [← σ.neg_one_smul, orbitMap_smul]

/-- The cells of the twisted complex correspond to those of the untwisted complex: representatives
to free orbits. -/
noncomputable def repOrbitEquiv (n : ℕ) :
    (σ.twistedComplex R hR).cells n ≃ (σ.untwistedComplex hF).cells n :=
  Equiv.ofBijective _ (σ.repOrbit_bijective R hR hF n)

/-- **Untwisting** (Proposition `prop:z2_reconstruction`, last claim): if the faces of fixed cells
are fixed and the representatives lie in a subcomplex `D` whose cells with image in `D` are
fixed, the twisted complex is isomorphic to the untwisted complex, the orbit complex relative to
the orbits of fixed cells, by `r ↦ [r]`. -/
noncomputable def untwistChainEquiv (hD : K.IsSubcomplex D)
    (hfix : ∀ n, ∀ e ∈ D n, σ.s n e ∈ D n → σ.s n e = e) (hRD : ∀ n, ↑(R n) ⊆ D n) :
    (σ.twistedComplex R hR).ChainEquiv (σ.untwistedComplex hF) where
  app n := LinearEquiv.funCongrLeft k k (σ.repOrbitEquiv R hR hF n).symm
  comm n := by
    refine LinearMap.ext fun a ↦ funext fun p ↦ ?_
    obtain ⟨q, rfl⟩ := (σ.repOrbitEquiv R hR hF n).surjective p
    simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.funCongrLeft_apply,
      LinearMap.funLeft_apply, bd_apply, Equiv.symm_apply_apply]
    rw [← (σ.repOrbitEquiv R hR hF (n + 1)).sum_comp]
    refine Finset.sum_congr rfl fun r _ ↦ ?_
    rw [Equiv.symm_apply_apply]
    congr 1
    exact (σ.twistedComplex_inc_eq_orbitComplex_inc R hR hD hfix hRD n r q).symm

variable [Invertible (2 : k)]

/-- **Proposition `prop:z2_reconstruction`**, untwisted form: under the hypotheses of
`untwistChainEquiv`, `H_n(K) ≃ H_n(orbit complex) × H_n(untwisted complex)`, the second factor
being the orbit complex relative to the orbits of fixed cells with untwisted coefficients. -/
noncomputable def untwistedReconstructionEquiv (hD : K.IsSubcomplex D)
    (hfix : ∀ n, ∀ e ∈ D n, σ.s n e ∈ D n → σ.s n e = e) (hRD : ∀ n, ↑(R n) ⊆ D n) (n : ℕ) :
    K.toChainComplex.homology n ≃ₗ[k]
      σ.orbitComplex.toChainComplex.homology n ×
        (σ.untwistedComplex hF).toChainComplex.homology n :=
  (σ.reconstructionEquiv R hR n).trans (LinearEquiv.prodCongr (LinearEquiv.refl k _)
    ((σ.untwistChainEquiv R hR hF hD hfix hRD).homologyLinearEquiv n))

end Untwisting

section Cover

variable {D : ∀ n, Set (K.cells n)}

/-- If every cell lies in `D ∪ gD` for a subcomplex `D` whose cells with image in `D` are fixed,
then the faces of fixed cells are fixed: a fixed cell lies in `D`, and with a face `e'` it has the
face `g e'`, both in `D`. -/
theorem facesFixed_of_cover (hD : K.IsSubcomplex D) (hcover : ∀ n e, e ∈ D n ∨ σ.s n e ∈ D n)
    (hfix : ∀ n, ∀ e ∈ D n, σ.s n e ∈ D n → σ.s n e = e) : σ.FacesFixed := by
  intro n e e' he h
  have heD : e ∈ D (n + 1) := (hcover (n + 1) e).elim id fun h' ↦ he ▸ h'
  have h' : K.inc n e (σ.s n e') ≠ 0 := by
    rwa [← σ.inc_apply n e e', he] at h
  exact hfix n e' (hD n e e' heD h) (hD n e (σ.s n e') heD h')

/-- If every cell lies in `D ∪ gD`, representatives can be chosen in `D`
(`exists_isRepresentatives_subset`). -/
theorem exists_isRepresentatives_subset_of_cover (hcover : ∀ n e, e ∈ D n ∨ σ.s n e ∈ D n)
    (n : ℕ) : ∃ R : Finset (K.cells n), IsRepresentatives (σ.s n) R ∧ ↑R ⊆ D n :=
  exists_isRepresentatives_subset (σ.involutive n) (Set.eq_univ_of_forall fun e ↦
    (hcover n e).elim Or.inl fun h ↦ Or.inr ⟨σ.s n e, h, σ.involutive n e⟩)

/-- **Proposition `prop:z2_reconstruction`, last claim**, for a cell complex with an involution:
if every cell lies in `D ∪ gD` for a subcomplex `D` whose cells with image in `D` are fixed, and
`2` is invertible, then `H_n(K) ≃ H_n(orbit complex) × H_n(untwisted complex)`. -/
theorem nonempty_untwisted [Invertible (2 : k)] (hD : K.IsSubcomplex D)
    (hcover : ∀ n e, e ∈ D n ∨ σ.s n e ∈ D n)
    (hfix : ∀ n, ∀ e ∈ D n, σ.s n e ∈ D n → σ.s n e = e) (n : ℕ) :
    Nonempty (K.toChainComplex.homology n ≃ₗ[k]
      σ.orbitComplex.toChainComplex.homology n ×
        (σ.untwistedComplex (σ.facesFixed_of_cover hD hcover hfix)).toChainComplex.homology n) := by
  choose R hR hRD using σ.exists_isRepresentatives_subset_of_cover hcover
  exact ⟨σ.untwistedReconstructionEquiv R hR _ hD hfix hRD n⟩

end Cover

section RelativeCover

variable (N L : ∀ n, Set (K.cells n)) [∀ n, DecidablePred (· ∈ N n)]
  [∀ n, DecidablePred (· ∈ L n)] (hN : K.IsSubcomplex N) (hL : K.IsSubcomplex L)
  (hsNL : ∀ n, ∀ e ∈ N n \ L n, σ.s n e ∈ N n \ L n) {D : ∀ n, Set (K.cells n)}

/-- The cells of a subcomplex `D` among the cells of `N` not in `L` form a subcomplex of the
relative complex. -/
theorem isSubcomplex_relative (hD : K.IsSubcomplex D) :
    (K.relative N L hN hL).IsSubcomplex
      fun n ↦ {e : (K.relative N L hN hL).cells n | Subtype.val e ∈ D n} :=
  fun n e e' he h ↦ hD n e.1 e'.1 he h

/-- `N \ L ⊆ D ∪ gD` gives the covering hypothesis of `nonempty_untwisted` for the relative
complex. -/
theorem relative_cover (hND : ∀ n, ∀ e ∈ N n \ L n, e ∈ D n ∨ σ.s n e ∈ D n) (n : ℕ)
    (e : (K.relative N L hN hL).cells n) :
    e ∈ {e : (K.relative N L hN hL).cells n | Subtype.val e ∈ D n} ∨
      (σ.relative N L hN hL hsNL).s n e ∈
        {e : (K.relative N L hN hL).cells n | Subtype.val e ∈ D n} :=
  hND n e.1 e.2

/-- `(N \ L) ∩ D ∩ gD ⊆ N^G` gives the fixed-cell hypothesis of `nonempty_untwisted` for the
relative complex. -/
theorem relative_fix (hDL : ∀ n, ∀ e ∈ N n \ L n, e ∈ D n → σ.s n e ∈ D n → σ.s n e = e)
    (n : ℕ) (e : (K.relative N L hN hL).cells n)
    (he : e ∈ {e : (K.relative N L hN hL).cells n | Subtype.val e ∈ D n})
    (hse : (σ.relative N L hN hL hsNL).s n e ∈
      {e : (K.relative N L hN hL).cells n | Subtype.val e ∈ D n}) :
    (σ.relative N L hN hL hsNL).s n e = e :=
  Subtype.ext (hDL n e.1 e.2 he hse)

/-- **Proposition `prop:z2_reconstruction`, last claim**, relative form: let `D` be a subcomplex
with `N \ L ⊆ D ∪ gD` and `(N \ L) ∩ D ∩ gD ⊆ N^G`, as for `N = D ∪ gD` with
`D ∩ gD ⊆ L ∪ N^G`. Then, for `2` invertible,
`H_n(N, L) ≃ H_n(orbit complex) × H_n(untwisted complex)` of the relative cells, the second factor
being the orbit complex relative to the orbits of fixed cells: the cellular complex of
`(N/G, (L ∪ N^G)/G)` with untwisted coefficients. -/
theorem nonempty_relative_untwisted [Invertible (2 : k)] (hD : K.IsSubcomplex D)
    (hND : ∀ n, ∀ e ∈ N n \ L n, e ∈ D n ∨ σ.s n e ∈ D n)
    (hDL : ∀ n, ∀ e ∈ N n \ L n, e ∈ D n → σ.s n e ∈ D n → σ.s n e = e) (n : ℕ) :
    Nonempty ((K.relative N L hN hL).toChainComplex.homology n ≃ₗ[k]
      (σ.relative N L hN hL hsNL).orbitComplex.toChainComplex.homology n ×
        ((σ.relative N L hN hL hsNL).untwistedComplex
          ((σ.relative N L hN hL hsNL).facesFixed_of_cover (isSubcomplex_relative N L hN hL hD)
            (σ.relative_cover N L hN hL hsNL hND)
            (σ.relative_fix N L hN hL hsNL hDL))).toChainComplex.homology n) :=
  (σ.relative N L hN hL hsNL).nonempty_untwisted (isSubcomplex_relative N L hN hL hD)
    (σ.relative_cover N L hN hL hsNL hND) (σ.relative_fix N L hN hL hsNL hDL) n

end RelativeCover

/-! ### The hypothesis on the faces of fixed cells is needed -/

section Counterexample

/-- The number of cells of a fixed edge with swapped endpoints: two vertices, one edge. -/
def fixedEdgeCount : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | _ + 2 => 0

/-- A fixed edge whose two endpoints are swapped, over `ℚ`: cells `Fin 2` in degree `0`, `Fin 1`
in degree `1` and none above, with incidence `1` between the edge and each vertex. The grid
condition rules it out, since the edge is fixed and its endpoints are not. -/
def fixedEdgeComplex : CellComplex.{0, 0} ℚ where
  cells n := Fin (fixedEdgeCount n)
  fintype _ := inferInstance
  inc n _ _ := if n = 0 then 1 else 0
  bd_bd n c := by
    funext j
    simp [bd_apply]

/-- The reflection of the fixed edge: `Fin.rev` in every degree, which swaps the vertices and
fixes the edge. -/
def fixedEdgeInvolution : CellularInvolution fixedEdgeComplex where
  s _ := Fin.rev
  involutive _ := Fin.rev_rev
  inc_apply _ _ _ := rfl

/-- The representatives of the free pairs of the fixed edge: the vertex `0` in degree `0`. -/
def fixedEdgeReps : ∀ n, Finset (fixedEdgeComplex.cells n)
  | 0 => ({0} : Finset (Fin 2))
  | _ + 1 => ∅

/-- `fixedEdgeReps` are representatives of the free pairs. -/
theorem isRepresentatives_fixedEdgeReps (n : ℕ) :
    IsRepresentatives (fixedEdgeInvolution.s n) (fixedEdgeReps n) := by
  rcases n with _ | _ | n
  · exact ⟨show ∀ i ∈ ({0} : Finset (Fin 2)), Fin.rev i ≠ i by decide,
      show ∀ i : Fin 2, Fin.rev i ≠ i →
        (i ∈ ({0} : Finset (Fin 2)) ↔ Fin.rev i ∉ ({0} : Finset (Fin 2))) by decide⟩
  · exact ⟨fun i hi ↦ absurd hi (Finset.notMem_empty i),
      fun i hi ↦ absurd (Subsingleton.elim (α := Fin 1) _ _) hi⟩
  · exact ⟨fun i hi ↦ absurd hi (Finset.notMem_empty i), fun i ↦ Fin.elim0 (show Fin 0 from i)⟩

/-- The hypothesis `FacesFixed` of `untwistedComplex` cannot be dropped from the untwisting in
the form "orbit complex relative to the orbits of fixed cells": for the fixed edge with swapped
endpoints, the representatives lie in a subcomplex `D` whose cells with image in `D` are fixed
(the hypotheses of `twistedComplex_inc_eq_add`, so the twisted incidences are still the orbit
incidences), but the orbits of fixed cells do not form a subcomplex of the orbit complex. The
grid condition, and the covering hypothesis of the paper (`facesFixed_of_cover`), rule this out.
Proposition `prop:z2_reconstruction`, last claim. -/
theorem exists_untwisting_not_isSubcomplex_fixedOrbits :
    ∃ (K : CellComplex.{0, 0} ℚ) (σ : CellularInvolution K) (D : ∀ n, Set (K.cells n))
      (R : ∀ n, Finset (K.cells n)), (∀ n, IsRepresentatives (σ.s n) (R n)) ∧
      K.IsSubcomplex D ∧ (∀ n, ∀ e ∈ D n, σ.s n e ∈ D n → σ.s n e = e) ∧
      (∀ n, ↑(R n) ⊆ D n) ∧ ¬ σ.orbitComplex.IsSubcomplex σ.fixedOrbits := by
  refine ⟨fixedEdgeComplex, fixedEdgeInvolution, fun n ↦ ↑(fixedEdgeReps n), fixedEdgeReps,
    isRepresentatives_fixedEdgeReps, ?_, ?_, fun _ ↦ subset_rfl, ?_⟩
  · intro n e e' he _
    exact absurd (Finset.mem_coe.mp he) (Finset.notMem_empty e)
  · intro n e he hse
    rcases n with _ | n
    · revert e
      change ∀ e : Fin 2, e ∈ ({0} : Finset (Fin 2)) → Fin.rev e ∈ ({0} : Finset (Fin 2)) →
        Fin.rev e = e
      decide
    · exact absurd (Finset.mem_coe.mp he) (Finset.notMem_empty e)
  · intro h
    have hfree : fixedEdgeInvolution.s 0 (0 : Fin 2) ≠ (0 : Fin 2) :=
      show Fin.rev (0 : Fin 2) ≠ 0 by decide
    have h1 := h 0 (orbitMap ℤˣ (fixedEdgeInvolution.Cells 1) (0 : Fin 1))
      (orbitMap ℤˣ (fixedEdgeInvolution.Cells 0) (0 : Fin 2))
      (fixedEdgeInvolution.orbitMap_mem_fixedOrbits_iff.mpr
        (show Fin.rev (0 : Fin 1) = 0 by decide))
    rw [fixedEdgeInvolution.orbitComplex_inc_orbitMap_of_free 0 _ hfree] at h1
    have h2 := h1 (show (if 0 = 0 then (1 : ℚ) else 0) + (if 0 = 0 then 1 else 0) ≠ 0 by
      norm_num)
    exact hfree (fixedEdgeInvolution.orbitMap_mem_fixedOrbits_iff.mp h2)

end Counterexample

end CellularInvolution

end CellComplex

end LatentEquivariant
