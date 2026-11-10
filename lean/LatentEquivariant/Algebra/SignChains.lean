import LatentEquivariant.Algebra.Chains
import LatentEquivariant.Algebra.Isotypic

/-!
# The sign component of the chains of a double cover

The chain-level part of Proposition `prop:z2_reconstruction`.
Let `G = {1, g}` act on a cell complex. In two consecutive dimensions the cells are finite types
`I` and `J`, the element `g` permutes them by involutions `s : I → I` and `t : J → J`, and the
incidence numbers `inc i j = [i : j]` in a commutative ring `k` are invariant,
`[s i : t j] = [i : j]`. The grid condition of the paper lets one orient cells so that `g`
permutes them without signs, but this model is weaker than the grid condition (see
`LatentEquivariant.Algebra.Chains`): it only says that a cell mapped to itself keeps its
orientation. The other consequence of the grid condition used here, that the faces of a fixed
cell are fixed, is the explicit hypothesis `hfixface` of
`IsRepresentatives.bd_apply_antiChains_of_free`. Relative chains `C_*(N, L)` are covered by taking
for `I` and `J` the cells of `N` that are not in `L`, and for `inc` the incidences between them.

A cell is fixed if `g` maps it to itself and free otherwise. A set of representatives contains
exactly one cell of each free pair `{i, s i}`. The free pairs of cells not in `L` are the cells
of `N/G` that are not in `(L ∪ N^G)/G`.

* `antiChains s` and `symChains s`: the `-1` and `+1` eigenspaces of `g` on chains.
* `apply_eq_zero_of_mem_antiChains`: an anti-invariant chain vanishes on the fixed cells if `2`
  is not a zero divisor of `k`, for instance if `2` is invertible or `k = ℤ`. This hypothesis is
  sharp (`isLeftRegular_two_iff_forall_antiChains`).
* `bd_mem_antiChains`: the boundary preserves the `-1` eigenspace, over any ring.
* `bd_apply_eq_zero_of_fixed`: fixed faces drop out of the boundary of an anti-invariant chain.
  The hypothesis on `2` is needed (`not_forall_bd_apply_eq_zero_zmod_two`), unless the chain
  vanishes on the fixed cells (`bd_apply_eq_zero_of_fixed_of_vanishing`).
* `antiChainsEquiv`: if `2` is not a zero divisor of `k`, restriction to representatives `R` is
  a linear equivalence from the `-1` eigenspace onto `R → k`. The inverse is `antiExtend`. Over
  any ring, `antiExtend` is injective and its image is the set of anti-invariant chains that
  vanish on the fixed cells
  (`IsRepresentatives.antiExtend_injective`, `IsRepresentatives.mem_range_antiExtend`).
* `antiChainsEquiv_comp_bd`, `IsRepresentatives.bd_apply_antiChains_of_isLeftRegular`: in these
  coordinates the boundary has the twisted incidences `[r : q] - [r : t q]`. This is the cellular
  chain complex of `(N/G, (L ∪ N^G)/G)` with coefficients twisted by the sign representation of
  the double cover. Over any ring, `bd_comp_antiExtend` says that `antiExtend` is a chain map
  from this twisted complex, and `IsRepresentatives.bd_apply_antiChains_of_free` gives the
  formula at free faces when the faces of fixed cells are fixed. Without such a hypothesis the
  formula fails (`not_forall_bd_apply_antiChains_zmod_two`, a configuration that the grid
  condition rules out).
  `IsRepresentatives.bd_apply_symChains` is the formula for the `+1` eigenspace, where the free
  orbits have the orbit-space incidences `[r : q] + [r : t q]`.
* `twInc_eq_of_mem`, `IsRepresentatives.bd_twInc_eq`: if `N = D ∪ gD` with `D ∩ gD ⊂ L ∪ N^G`,
  then for representatives in `D` every sign is `+1`, and the twisted complex is the untwisted
  complex of the orbit space relative to the fixed cells. `exists_isRepresentatives_subset`
  provides such representatives.

Apart from the splitting `isCompl_symChains_antiChains` of the chains into the two eigenspaces,
the hypothesis on `2` is only used through `apply_eq_zero_of_mem_antiChains`, and only in the form
that `2` is not a zero divisor. So the chain-level identification of the `-1` eigenspace with the
twisted complex also holds over `ℤ`. This does not extend to the rest of the proposition over
`ℤ`: the `+1` eigenspace is not the orbit complex (for a reflection of a circle, `H_0` of the
invariant chains is `ℤ ⊕ ℤ/2`, while the orbit space is an interval), and the homology of the
`-1` eigenspace is not the `-1` eigenspace of the homology (for the antipodal map of a circle it
is `(ℤ/2, 0)`, while `g` acts by `+1` on `H_*(S¹; ℤ)`). Both need `2` invertible.
-/

namespace LatentEquivariant

open Function

variable {k I J : Type*} [CommRing k]

section Eigenspaces

/-- The `-1` eigenspace of `g` on the chains `I → k`, where `g` permutes the cells by `s`: the
chains with `c (s i) = -c i`. Proposition `prop:z2_reconstruction`. -/
def antiChains (s : I → I) : Submodule k (I → k) where
  carrier := {c | ∀ i, c (s i) = -c i}
  add_mem' {c d} hc hd i := by
    simp only [Pi.add_apply, hc i, hd i, neg_add]
  zero_mem' _ := by simp
  smul_mem' a c hc i := by
    simp only [Pi.smul_apply, smul_eq_mul, hc i, mul_neg]

/-- The `+1` eigenspace of `g` on the chains `I → k`: the chains with `c (s i) = c i`.
Proposition `prop:z2_reconstruction`. -/
def symChains (s : I → I) : Submodule k (I → k) where
  carrier := {c | ∀ i, c (s i) = c i}
  add_mem' {c d} hc hd i := by
    simp only [Pi.add_apply, hc i, hd i]
  zero_mem' _ := rfl
  smul_mem' a c hc i := by
    simp only [Pi.smul_apply, smul_eq_mul, hc i]

variable {s : I → I} {c : I → k}

/-- Membership in the `-1` eigenspace. -/
theorem mem_antiChains : c ∈ antiChains s ↔ ∀ i, c (s i) = -c i :=
  Iff.rfl

/-- Membership in the `+1` eigenspace. -/
theorem mem_symChains : c ∈ symChains s ↔ ∀ i, c (s i) = c i :=
  Iff.rfl

/-- The `-1` eigenspace is the kernel of `σ + 1`, where `σ c = c ∘ s` is the action of `g` on
chains. Proposition `prop:z2_reconstruction`. -/
theorem antiChains_eq_ker :
    antiChains (k := k) s = LinearMap.ker (LinearMap.funLeft k k s + 1) := by
  ext c
  simp only [mem_antiChains, LinearMap.mem_ker, LinearMap.add_apply, Module.End.one_apply,
    add_eq_zero_iff_eq_neg, funext_iff, LinearMap.funLeft_apply, Pi.neg_apply]

/-- The `+1` eigenspace is the kernel of `σ - 1`, where `σ c = c ∘ s` is the action of `g` on
chains. Proposition `prop:z2_reconstruction`. -/
theorem symChains_eq_ker :
    symChains (k := k) s = LinearMap.ker (LinearMap.funLeft k k s - 1) := by
  ext c
  simp only [mem_symChains, LinearMap.mem_ker, LinearMap.sub_apply, Module.End.one_apply,
    sub_eq_zero, funext_iff, LinearMap.funLeft_apply]

/-- The action `σ c = c ∘ s` of an involution on chains is an involution. -/
theorem funLeft_mul_self (hs : Involutive s) :
    LinearMap.funLeft k k s * LinearMap.funLeft k k s = 1 :=
  LinearMap.ext fun c ↦ funext fun i ↦ by
    rw [Module.End.mul_apply, LinearMap.funLeft_apply, LinearMap.funLeft_apply, hs i,
      Module.End.one_apply]

/-- With `2` invertible the chains are the direct sum of the `+1` and `-1` eigenspaces, the
chain-level splitting of Proposition `prop:z2_reconstruction`. -/
theorem isCompl_symChains_antiChains [Invertible (2 : k)] (hs : Involutive s) :
    IsCompl (symChains (k := k) s) (antiChains s) := by
  rw [symChains_eq_ker, antiChains_eq_ker]
  exact isCompl_ker_sub_one_ker_add_one (funLeft_mul_self hs)

/-- An invertible `2` is not a zero divisor. -/
theorem isLeftRegular_two_of_invertible [Invertible (2 : k)] : IsLeftRegular (2 : k) :=
  (isUnit_of_invertible (2 : k)).isRegular.left

/-- An anti-invariant chain vanishes on the fixed cells if `2` is not a zero divisor of `k`, for
instance if `2` is invertible or `k = ℤ`. Proposition `prop:z2_reconstruction`, draft proof: the
`-1` eigenspace is spanned by the chains `e - ge` of the free cells. -/
theorem apply_eq_zero_of_mem_antiChains (h2 : IsLeftRegular (2 : k)) (hc : c ∈ antiChains s)
    {i : I} (hi : s i = i) : c i = 0 := by
  have h : c i = -c i := by
    have := mem_antiChains.mp hc i
    rwa [hi] at this
  refine h2 ?_
  change 2 * c i = 2 * 0
  linear_combination h

/-- If `2` is invertible, an anti-invariant chain vanishes on the fixed cells. Proposition
`prop:z2_reconstruction`. -/
theorem apply_eq_zero_of_mem_antiChains_of_invertible [Invertible (2 : k)]
    (hc : c ∈ antiChains s) {i : I} (hi : s i = i) : c i = 0 :=
  apply_eq_zero_of_mem_antiChains isLeftRegular_two_of_invertible hc hi

/-- The hypothesis of `apply_eq_zero_of_mem_antiChains` is sharp: anti-invariant chains vanish on
fixed cells for every complex over `k` exactly when `2` is not a zero divisor of `k`. If
`2 x = 0`, the chain `x` on a single fixed cell is anti-invariant. -/
theorem isLeftRegular_two_iff_forall_antiChains :
    IsLeftRegular (2 : k) ↔ ∀ c ∈ antiChains (k := k) (id : Unit → Unit), c () = 0 := by
  refine ⟨fun h2 c hc ↦ apply_eq_zero_of_mem_antiChains h2 hc rfl, fun h a b hab ↦ ?_⟩
  have hab' : 2 * (a - b) = 0 := by
    rw [mul_sub]
    exact sub_eq_zero.mpr hab
  have hx : (fun _ ↦ a - b : Unit → k) ∈ antiChains id := fun _ ↦ by
    change a - b = -(a - b)
    linear_combination hab'
  exact sub_eq_zero.mp (h _ hx)

/-- The hypothesis on `2` in `apply_eq_zero_of_mem_antiChains` is needed: over `ZMod 2` the chain
`1` on a single fixed cell is anti-invariant, because `-1 = 1`. -/
theorem not_forall_apply_eq_zero_zmod_two :
    ¬ ∀ (s : Unit → Unit), Involutive s →
      ∀ c ∈ antiChains (k := ZMod 2) s, ∀ i, s i = i → c i = 0 := by
  intro h
  exact absurd (h id (fun _ ↦ rfl) (fun _ ↦ 1) (fun _ ↦ rfl) () rfl) (by decide)

end Eigenspaces

section Representatives

variable {s : I → I} {R : Finset I}

/-- `R` is a set of representatives of the free cells for the involution `s`: it contains no
fixed cell and exactly one cell of each free pair `{i, s i}`. Proposition
`prop:z2_reconstruction`, draft proof: "choosing a representative cell in each free orbit". -/
structure IsRepresentatives (s : I → I) (R : Finset I) : Prop where
  free : ∀ i ∈ R, s i ≠ i
  mem_iff : ∀ i, s i ≠ i → (i ∈ R ↔ s i ∉ R)

/-- The image of a representative is not a representative. -/
theorem IsRepresentatives.apply_notMem (hR : IsRepresentatives s R) {i : I} (hi : i ∈ R) :
    s i ∉ R :=
  (hR.mem_iff i (hR.free i hi)).mp hi

/-- A free cell that is not a representative is the image of one. -/
theorem IsRepresentatives.apply_mem (hR : IsRepresentatives s R) {i : I} (hi : s i ≠ i)
    (hiR : i ∉ R) : s i ∈ R := by
  by_contra h
  exact hiR ((hR.mem_iff i hi).mpr h)

/-- A cell that is neither a representative nor the image of one is fixed. -/
theorem IsRepresentatives.apply_eq_of_notMem (hR : IsRepresentatives s R) {i : I} (hi : i ∉ R)
    (hsi : s i ∉ R) : s i = i := by
  by_contra h
  exact hi ((hR.mem_iff i h).mpr hsi)

/-- A sum over all cells splits into the free pairs `{r, s r}` with `r ∈ R` and the fixed
cells. -/
theorem IsRepresentatives.sum_eq [Fintype I] [DecidableEq I] {M : Type*} [AddCommMonoid M]
    (hs : Involutive s) (hR : IsRepresentatives s R) (F : I → M) :
    ∑ i, F i = ∑ r ∈ R, (F r + F (s r)) + ∑ i with s i = i, F i := by
  have h1 : Disjoint R (R.image s) := by
    rw [Finset.disjoint_left]
    intro i hi hi'
    obtain ⟨r, hr, rfl⟩ := Finset.mem_image.mp hi'
    exact hR.apply_notMem hr hi
  have h2 : Disjoint (R ∪ R.image s) ({i | s i = i} : Finset I) := by
    rw [Finset.disjoint_left]
    intro i hi hfix
    have hfix' : s i = i := (Finset.mem_filter.mp hfix).2
    rcases Finset.mem_union.mp hi with hi | hi
    · exact hR.free i hi hfix'
    · obtain ⟨r, hr, rfl⟩ := Finset.mem_image.mp hi
      rw [hs r] at hfix'
      exact hR.free r hr hfix'.symm
  have h3 : (R ∪ R.image s) ∪ ({i | s i = i} : Finset I) = Finset.univ := by
    refine Finset.eq_univ_of_forall fun i ↦ ?_
    by_cases hfix : s i = i
    · exact Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨Finset.mem_univ i, hfix⟩)
    · refine Finset.mem_union_left _ ?_
      by_cases hiR : i ∈ R
      · exact Finset.mem_union_left _ hiR
      · exact Finset.mem_union_right _
          (Finset.mem_image.mpr ⟨s i, hR.apply_mem hfix hiR, hs i⟩)
  conv_lhs => rw [← h3]
  rw [Finset.sum_union h2, Finset.sum_union h1, Finset.sum_image hs.injective.injOn,
    Finset.sum_add_distrib]

/-- Representatives inside `D` exist if every cell lies in `D ∪ s(D)`. Proposition
`prop:z2_reconstruction`, last claim: for `N = D ∪ gD` the representatives are chosen in `D`. -/
theorem exists_isRepresentatives_subset [Finite I] (hs : Involutive s) {D : Set I}
    (hD : D ∪ s '' D = Set.univ) : ∃ R : Finset I, IsRepresentatives s R ∧ ↑R ⊆ D := by
  classical
  have := Fintype.ofFinite I
  have hD' : ∀ i, i ∈ D ∨ s i ∈ D := by
    intro i
    have h : i ∈ D ∪ s '' D := hD ▸ Set.mem_univ i
    rwa [hs.image_eq_preimage_symm, Set.mem_union, Set.mem_preimage] at h
  let e := Fintype.equivFin I
  refine ⟨({i | s i ≠ i ∧ i ∈ D ∧ (s i ∈ D → e i < e (s i))} : Finset I), ⟨?_, ?_⟩, ?_⟩
  · intro i hi
    exact (Finset.mem_filter.mp hi).2.1
  · intro i hi
    have hne : e i ≠ e (s i) := fun h ↦ hi (e.injective h).symm
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, hs i]
    constructor
    · rintro ⟨-, hiD, hlt⟩ ⟨-, hsiD, hlt'⟩
      exact lt_asymm (hlt hsiD) (hlt' hiD)
    · intro h
      refine ⟨hi, ?_, ?_⟩
      · rcases hD' i with hiD | hsiD
        · exact hiD
        · by_contra hiD
          exact h ⟨fun h' ↦ hi h'.symm, hsiD, fun h' ↦ absurd h' hiD⟩
      · intro hsiD
        by_contra hlt
        refine h ⟨fun h' ↦ hi h'.symm, hsiD, fun _ ↦ ?_⟩
        exact lt_of_le_of_ne (not_lt.mp hlt) hne.symm
  · intro i hi
    exact (Finset.mem_filter.mp hi).2.2.1

/-- Representatives of the free cells exist. -/
theorem exists_isRepresentatives [Finite I] (hs : Involutive s) :
    ∃ R : Finset I, IsRepresentatives s R :=
  (exists_isRepresentatives_subset hs (D := Set.univ) (Set.univ_union _)).imp fun _ h ↦ h.1

end Representatives

/-- The twisted incidence `[r : q] - [r : t q]` of a representative `r` and a face `q`: the
incidence of the free orbits `[r]` and `[q]` with coefficients in the sign representation of the
double cover, when `r` and `q` are the chosen lifts. Proposition `prop:z2_reconstruction`. -/
def twInc (inc : I → J → k) (t : J → J) (r : I) (q : J) : k :=
  inc r q - inc r (t q)

section Boundary

variable [Fintype I] {s : I → I} {t : J → J} {inc : I → J → k} {c : I → k}

/-- The boundary commutes with the action of `g`: `∂ (c ∘ s) = (∂ c) ∘ t`. Proposition
`prop:z2_reconstruction`, draft proof: `∂(e - ge) = (1 - g) ∂e`. -/
theorem bd_comp_funLeft (hs : Involutive s) (hinc : ∀ i j, inc (s i) (t j) = inc i j) :
    bd inc ∘ₗ LinearMap.funLeft k k s = LinearMap.funLeft k k t ∘ₗ bd inc := by
  refine LinearMap.ext fun c ↦ funext fun j ↦ ?_
  simp only [LinearMap.comp_apply, LinearMap.funLeft_apply, bd_apply]
  exact Fintype.sum_bijective s hs.bijective _ _ fun i ↦ by rw [hinc]

/-- The boundary maps the `-1` eigenspace into the `-1` eigenspace, over any ring. Proposition
`prop:z2_reconstruction`. -/
theorem bd_mem_antiChains (hs : Involutive s) (hinc : ∀ i j, inc (s i) (t j) = inc i j)
    (hc : c ∈ antiChains s) : bd inc c ∈ antiChains t := by
  intro j
  have hc' := mem_antiChains.mp hc
  calc bd inc c (t j) = ∑ i, c (s i) * inc (s i) (t j) := by
        rw [bd_apply]
        exact (hs.bijective.sum_comp fun i ↦ c i * inc i (t j)).symm
    _ = ∑ i, -(c i * inc i j) := by simp only [hc', hinc, neg_mul]
    _ = -bd inc c j := by rw [Finset.sum_neg_distrib, bd_apply]

/-- The boundary maps the `+1` eigenspace into the `+1` eigenspace, over any ring. Proposition
`prop:z2_reconstruction`. -/
theorem bd_mem_symChains (hs : Involutive s) (hinc : ∀ i j, inc (s i) (t j) = inc i j)
    (hc : c ∈ symChains s) : bd inc c ∈ symChains t := by
  intro j
  have hc' := mem_symChains.mp hc
  calc bd inc c (t j) = ∑ i, c (s i) * inc (s i) (t j) := by
        rw [bd_apply]
        exact (hs.bijective.sum_comp fun i ↦ c i * inc i (t j)).symm
    _ = bd inc c j := by simp only [hc', hinc, bd_apply]

/-- Fixed faces drop out of the boundary of an anti-invariant chain if `2` is not a zero divisor
of `k`. Proposition `prop:z2_reconstruction`, draft proof: `∂(e - ge) = (1 - g) ∂e`. -/
theorem bd_apply_eq_zero_of_fixed (h2 : IsLeftRegular (2 : k)) (hs : Involutive s)
    (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hc : c ∈ antiChains s) {q : J} (hq : t q = q) :
    bd inc c q = 0 :=
  apply_eq_zero_of_mem_antiChains h2 (bd_mem_antiChains hs hinc hc) hq

/-- Fixed faces drop out of the boundary of an anti-invariant chain if `2` is invertible.
Proposition `prop:z2_reconstruction`. -/
theorem bd_apply_eq_zero_of_fixed_of_invertible [Invertible (2 : k)] (hs : Involutive s)
    (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hc : c ∈ antiChains s) {q : J} (hq : t q = q) :
    bd inc c q = 0 :=
  bd_apply_eq_zero_of_fixed isLeftRegular_two_of_invertible hs hinc hc hq

/-- The hypothesis on `2` in `bd_apply_eq_zero_of_fixed` is needed. Over `ZMod 2`, take one cell
fixed by `g` with one fixed face of incidence `1`. The chain `1` is anti-invariant, because
`-1 = 1`, and its boundary is `1` on the fixed face. -/
theorem not_forall_bd_apply_eq_zero_zmod_two :
    ¬ ∀ (s t : Unit → Unit) (inc : Unit → Unit → ZMod 2), Involutive s → Involutive t →
      (∀ i j, inc (s i) (t j) = inc i j) →
      ∀ c ∈ antiChains s, ∀ q, t q = q → bd inc c q = 0 := by
  intro h
  have h1 : (fun _ ↦ 1 : Unit → ZMod 2) ∈ antiChains id := fun _ ↦ rfl
  have := h id id (fun _ _ ↦ 1) (fun _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl) _ h1 () rfl
  simp at this

/-- The boundary of an anti-invariant chain over any ring, in terms of the representatives `R`
and the fixed cells: `(∂c)(q) = ∑_{r ∈ R} c(r) ([r : q] - [r : t q]) + ∑_{s i = i} c(i) [i : q]`
for every face `q`. Proposition `prop:z2_reconstruction`, draft proof. -/
theorem IsRepresentatives.bd_apply_antiChains [DecidableEq I] {R : Finset I} (hs : Involutive s)
    (ht : Involutive t) (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hR : IsRepresentatives s R)
    (hc : c ∈ antiChains s) (q : J) :
    bd inc c q = ∑ r ∈ R, c r * twInc inc t r q + ∑ i with s i = i, c i * inc i q := by
  rw [bd_apply, hR.sum_eq hs (fun i ↦ c i * inc i q)]
  congr 1
  refine Finset.sum_congr rfl fun r _ ↦ ?_
  have h1 : inc (s r) q = inc r (t q) := by
    have := hinc r (t q)
    rwa [ht q] at this
  rw [mem_antiChains.mp hc r, h1, twInc]
  ring

/-- The boundary in the `+1` eigenspace: the boundary of an invariant chain over any ring is
`(∂c)(q) = ∑_{r ∈ R} c(r) ([r : q] + [r : t q]) + ∑_{s i = i} c(i) [i : q]` for every face `q`.
For free `r` and `q`, the coefficient `[r : q] + [r : t q]` is the incidence of the orbits `[r]`
and `[q]` in `N/G`. Proposition `prop:z2_reconstruction`, draft proof. -/
theorem IsRepresentatives.bd_apply_symChains [DecidableEq I] {R : Finset I} (hs : Involutive s)
    (ht : Involutive t) (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hR : IsRepresentatives s R)
    (hc : c ∈ symChains s) (q : J) :
    bd inc c q = ∑ r ∈ R, c r * (inc r q + inc r (t q)) + ∑ i with s i = i, c i * inc i q := by
  rw [bd_apply, hR.sum_eq hs (fun i ↦ c i * inc i q)]
  congr 1
  refine Finset.sum_congr rfl fun r _ ↦ ?_
  have h1 : inc (s r) q = inc r (t q) := by
    have := hinc r (t q)
    rwa [ht q] at this
  rw [mem_symChains.mp hc r, h1]
  ring

/-- Over any ring, fixed faces drop out of the boundary of an anti-invariant chain that vanishes
on the fixed cells. Proposition `prop:z2_reconstruction`. -/
theorem bd_apply_eq_zero_of_fixed_of_vanishing (hs : Involutive s) (ht : Involutive t)
    (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hc : c ∈ antiChains s)
    (hfix : ∀ i, s i = i → c i = 0) {q : J} (hq : t q = q) : bd inc c q = 0 := by
  classical
  obtain ⟨R, hR⟩ := exists_isRepresentatives hs
  rw [hR.bd_apply_antiChains hs ht hinc hc q]
  have h1 : ∀ r ∈ R, c r * twInc inc t r q = 0 := fun r _ ↦ by
    rw [twInc, hq, sub_self, mul_zero]
  have h2 : ∀ i ∈ ({i | s i = i} : Finset I), c i * inc i q = 0 := fun i hi ↦ by
    rw [hfix i (Finset.mem_filter.mp hi).2, zero_mul]
  rw [Finset.sum_eq_zero h1, Finset.sum_eq_zero h2, add_zero]

end Boundary

section Coordinates

variable [DecidableEq I] {s : I → I} {R : Finset I}

/-- Extension by zero from the cells of `R` to all cells. -/
def extendByZero (R : Finset I) : (R → k) →ₗ[k] (I → k) where
  toFun a i := if h : i ∈ R then a ⟨i, h⟩ else 0
  map_add' a b := by
    funext i
    by_cases h : i ∈ R <;> simp [h]
  map_smul' x a := by
    funext i
    by_cases h : i ∈ R <;> simp [h]

/-- The values of the extension by zero. -/
theorem extendByZero_apply (a : R → k) (i : I) :
    extendByZero R a i = if h : i ∈ R then a ⟨i, h⟩ else 0 :=
  rfl

/-- The extension by zero agrees with `a` on `R`. -/
@[simp] theorem extendByZero_apply_coe (a : R → k) (r : R) : extendByZero R a r = a r := by
  rw [extendByZero_apply, dif_pos r.2]

/-- The extension by zero vanishes off `R`. -/
theorem extendByZero_apply_of_notMem (a : R → k) {i : I} (hi : i ∉ R) :
    extendByZero R a i = 0 := by
  rw [extendByZero_apply, dif_neg hi]

/-- A sum against the extension by zero is a sum over `R`. -/
theorem sum_extendByZero_mul [Fintype I] (a : R → k) (F : I → k) :
    ∑ i, extendByZero R a i * F i = ∑ r : R, a r * F r := by
  rw [← Finset.sum_subset (Finset.subset_univ R) fun i _ hi ↦ by
    rw [extendByZero_apply_of_notMem a hi, zero_mul], ← Finset.sum_coe_sort R]
  exact Finset.sum_congr rfl fun r _ ↦ by rw [extendByZero_apply_coe]

/-- The chain `â - â ∘ s` defined by `a : R → k`, where `â` extends `a` by zero. It is the chain
`∑_{r ∈ R} a(r) (r - g r)` of the draft proof of Proposition `prop:z2_reconstruction`, and for
representatives `R` it equals `a` on `R`, `-a ∘ s` on `s(R)` and `0` on the fixed cells. -/
def antiExtend (s : I → I) (R : Finset I) : (R → k) →ₗ[k] (I → k) :=
  extendByZero R - LinearMap.funLeft k k s ∘ₗ extendByZero R

/-- The values of `antiExtend s R a`. -/
theorem antiExtend_apply (a : R → k) (i : I) :
    antiExtend s R a i = extendByZero R a i - extendByZero R a (s i) :=
  rfl

/-- `antiExtend s R a` is anti-invariant, for every `R`. -/
theorem antiExtend_mem_antiChains (hs : Involutive s) (a : R → k) :
    antiExtend s R a ∈ antiChains s := fun i ↦ by
  rw [antiExtend_apply, antiExtend_apply, hs i, neg_sub]

/-- `antiExtend s R a` equals `a` on the representatives. -/
theorem IsRepresentatives.antiExtend_apply_coe (hR : IsRepresentatives s R) (a : R → k)
    (r : R) : antiExtend s R a r = a r := by
  rw [antiExtend_apply, extendByZero_apply_coe,
    extendByZero_apply_of_notMem a (hR.apply_notMem r.2), sub_zero]

/-- `antiExtend s R a` equals `-a ∘ s` on the images of the representatives. -/
theorem IsRepresentatives.antiExtend_apply_of_apply_mem (hs : Involutive s)
    (hR : IsRepresentatives s R) (a : R → k) {i : I} (hi : s i ∈ R) :
    antiExtend s R a i = -a ⟨s i, hi⟩ := by
  have hiR : i ∉ R := by
    have := hR.apply_notMem hi
    rwa [hs i] at this
  rw [antiExtend_apply, extendByZero_apply_of_notMem a hiR, zero_sub, extendByZero_apply,
    dif_pos hi]

/-- `antiExtend s R a` vanishes on the fixed cells. -/
theorem antiExtend_apply_of_fixed (a : R → k) {i : I} (hi : s i = i) :
    antiExtend s R a i = 0 := by
  rw [antiExtend_apply, hi, sub_self]

/-- Over any ring, an anti-invariant chain that vanishes on the fixed cells is `antiExtend` of its
restriction to the representatives. -/
theorem IsRepresentatives.antiExtend_restrict (hs : Involutive s) (hR : IsRepresentatives s R)
    {c : I → k} (hc : c ∈ antiChains s) (hfix : ∀ i, s i = i → c i = 0) :
    antiExtend s R (fun r ↦ c r) = c := by
  funext i
  by_cases hi : i ∈ R
  · exact hR.antiExtend_apply_coe _ ⟨i, hi⟩
  · by_cases hsi : s i ∈ R
    · rw [hR.antiExtend_apply_of_apply_mem hs _ hsi]
      change -c (s i) = c i
      rw [mem_antiChains.mp hc i, neg_neg]
    · have h := hR.apply_eq_of_notMem hi hsi
      rw [antiExtend_apply_of_fixed _ h, hfix i h]

/-- `antiExtend s R` is injective for representatives `R`, over any ring. -/
theorem IsRepresentatives.antiExtend_injective (hR : IsRepresentatives s R) :
    Injective (antiExtend (k := k) s R) := fun a b h ↦ funext fun r ↦ by
  rw [← hR.antiExtend_apply_coe a r, ← hR.antiExtend_apply_coe b r, h]

/-- Over any ring, the image of `antiExtend s R` for representatives `R` consists of the
anti-invariant chains that vanish on the fixed cells, that is, the `-1` eigenspace of the relative
chains of `(N, L ∪ N^G)`. Proposition `prop:z2_reconstruction`. -/
theorem IsRepresentatives.mem_range_antiExtend (hs : Involutive s) (hR : IsRepresentatives s R)
    {c : I → k} :
    c ∈ LinearMap.range (antiExtend s R) ↔ c ∈ antiChains s ∧ ∀ i, s i = i → c i = 0 := by
  rw [LinearMap.mem_range]
  constructor
  · rintro ⟨a, rfl⟩
    exact ⟨antiExtend_mem_antiChains hs a, fun i hi ↦ antiExtend_apply_of_fixed a hi⟩
  · rintro ⟨hc, hfix⟩
    exact ⟨fun r ↦ c r, hR.antiExtend_restrict hs hc hfix⟩

/-- Coordinates on the `-1` eigenspace: if `2` is not a zero divisor of `k`, for
instance if `2` is invertible, restriction to the representatives `R` is a linear equivalence
from the anti-invariant chains onto `R → k`. The inverse `antiExtend s R` sends `a` to the chain
equal to `a` on `R`, to `-a ∘ s` on `s(R)` and to `0` on the fixed cells. Proposition
`prop:z2_reconstruction`, draft proof: the chains `e - ge` over the free orbits span the `-1`
eigenspace. -/
def antiChainsEquiv (hs : Involutive s) (hR : IsRepresentatives s R)
    (h2 : IsLeftRegular (2 : k)) : antiChains (k := k) s ≃ₗ[k] (R → k) where
  toFun c r := (c : I → k) r
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun a := ⟨antiExtend s R a, antiExtend_mem_antiChains hs a⟩
  left_inv c := Subtype.ext <|
    hR.antiExtend_restrict hs c.2 fun _ hi ↦ apply_eq_zero_of_mem_antiChains h2 c.2 hi
  right_inv a := funext fun r ↦ hR.antiExtend_apply_coe a r

/-- `antiChainsEquiv` is the restriction to the representatives. -/
@[simp] theorem antiChainsEquiv_apply (hs : Involutive s) (hR : IsRepresentatives s R)
    (h2 : IsLeftRegular (2 : k)) (c : antiChains (k := k) s) (r : R) :
    antiChainsEquiv hs hR h2 c r = (c : I → k) r :=
  rfl

/-- The inverse of `antiChainsEquiv` is `antiExtend`. -/
@[simp] theorem coe_antiChainsEquiv_symm_apply (hs : Involutive s) (hR : IsRepresentatives s R)
    (h2 : IsLeftRegular (2 : k)) (a : R → k) :
    ((antiChainsEquiv hs hR h2).symm a : I → k) = antiExtend s R a :=
  rfl

variable [Fintype I] {t : J → J} {inc : I → J → k}

/-- The boundary of `antiExtend s R a` has the twisted incidences:
`∂(∑_{r ∈ R} a(r) (r - g r)) = ∑_q (∑_{r ∈ R} a(r) ([r : q] - [r : t q])) q`, for every `R`.
Proposition `prop:z2_reconstruction`, draft proof: `∂(e - ge) = (1 - g) ∂e`. -/
theorem bd_antiExtend (hs : Involutive s) (hinc : ∀ i j, inc (s i) (t j) = inc i j)
    (a : R → k) (q : J) : bd inc (antiExtend s R a) q = ∑ r : R, a r * twInc inc t r q := by
  have h3 : ∑ i, extendByZero R a (s i) * inc i q = ∑ i, extendByZero R a i * inc i (t q) :=
    Fintype.sum_bijective s hs.bijective _ _ fun i ↦ by rw [hinc]
  simp only [bd_apply, antiExtend_apply, sub_mul, Finset.sum_sub_distrib, h3, twInc]
  rw [← Finset.sum_sub_distrib]
  simp only [← mul_sub]
  exact sum_extendByZero_mul a _

/-- Over any ring, `antiExtend` is a chain map from the complex on the representatives with the
twisted incidences: `∂ ∘ antiExtend s R = antiExtend t Q ∘ ∂^w`. With
`IsRepresentatives.antiExtend_injective` and `IsRepresentatives.mem_range_antiExtend`, the twisted
complex is isomorphic to the complex of anti-invariant chains vanishing on the fixed cells.
Proposition `prop:z2_reconstruction`. -/
theorem bd_comp_antiExtend [DecidableEq J] {Q : Finset J} (hs : Involutive s)
    (ht : Involutive t) (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hQ : IsRepresentatives t Q) :
    bd inc ∘ₗ antiExtend s R =
      antiExtend t Q ∘ₗ bd (fun (r : R) (q : Q) ↦ twInc inc t r q) := by
  refine LinearMap.ext fun a ↦ funext fun j ↦ ?_
  rw [LinearMap.comp_apply, LinearMap.comp_apply, bd_antiExtend hs hinc]
  by_cases hj : j ∈ Q
  · obtain ⟨q, rfl⟩ : ∃ q : Q, (q : J) = j := ⟨⟨j, hj⟩, rfl⟩
    rw [hQ.antiExtend_apply_coe, bd_apply]
  · by_cases htj : t j ∈ Q
    · rw [hQ.antiExtend_apply_of_apply_mem ht _ htj, bd_apply, ← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun r _ ↦ ?_
      simp only [twInc, ht j]
      ring
    · have hfix := hQ.apply_eq_of_notMem hj htj
      rw [antiExtend_apply_of_fixed _ hfix]
      refine Finset.sum_eq_zero fun r _ ↦ ?_
      rw [twInc, hfix, sub_self, mul_zero]

end Coordinates

section TwistedComplex

variable [Fintype I] {s : I → I} {t : J → J} {inc : I → J → k} {c : I → k} {R : Finset I}

/-- The twisted complex: if `2` is not a zero divisor of `k`, the boundary of an anti-invariant
chain is `(∂c)(q) = ∑_{r ∈ R} c(r) ([r : q] - [r : t q])` for every face `q`, fixed or free.
Proposition `prop:z2_reconstruction`. -/
theorem IsRepresentatives.bd_apply_antiChains_of_isLeftRegular (h2 : IsLeftRegular (2 : k))
    (hs : Involutive s) (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hR : IsRepresentatives s R)
    (hc : c ∈ antiChains s) (q : J) : bd inc c q = ∑ r ∈ R, c r * twInc inc t r q := by
  classical
  calc bd inc c q = bd inc (antiExtend s R (antiChainsEquiv hs hR h2 ⟨c, hc⟩)) q := by
        rw [← coe_antiChainsEquiv_symm_apply hs hR h2, LinearEquiv.symm_apply_apply]
    _ = ∑ r : R, c r * twInc inc t r q := bd_antiExtend hs hinc _ q
    _ = ∑ r ∈ R, c r * twInc inc t r q := Finset.sum_coe_sort R fun r ↦ c r * twInc inc t r q

/-- The twisted complex over any ring, at free faces: if the faces of fixed cells are fixed, then
`(∂c)(q) = ∑_{r ∈ R} c(r) ([r : q] - [r : t q])` for an anti-invariant chain `c` and every free
face `q`. The hypothesis on the faces holds for a complex with the grid condition, where a cell
mapped to itself is fixed pointwise, and then so is its closure. Proposition
`prop:z2_reconstruction`. -/
theorem IsRepresentatives.bd_apply_antiChains_of_free (hs : Involutive s) (ht : Involutive t)
    (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hR : IsRepresentatives s R)
    (hfixface : ∀ i j, s i = i → inc i j ≠ 0 → t j = j) (hc : c ∈ antiChains s) {q : J}
    (hq : t q ≠ q) : bd inc c q = ∑ r ∈ R, c r * twInc inc t r q := by
  classical
  rw [hR.bd_apply_antiChains hs ht hinc hc q, add_eq_left]
  refine Finset.sum_eq_zero fun i hi ↦ ?_
  by_cases h : inc i q = 0
  · rw [h, mul_zero]
  · exact absurd (hfixface i q (Finset.mem_filter.mp hi).2 h) hq

/-- The twisted formula needs a hypothesis: without one on `2` or on the faces of fixed cells, it
fails at free faces. Over `ZMod 2`, take one fixed cell with a free pair of faces, both of
incidence `1`, and no representatives. The grid condition rules this configuration out: the
closure of a cell fixed pointwise is fixed pointwise. -/
theorem not_forall_bd_apply_antiChains_zmod_two :
    ¬ ∀ (s : Unit → Unit) (t : Bool → Bool) (inc : Unit → Bool → ZMod 2) (R : Finset Unit),
      Involutive s → Involutive t → (∀ i j, inc (s i) (t j) = inc i j) →
      IsRepresentatives s R → ∀ c ∈ antiChains s, ∀ q, t q ≠ q →
      bd inc c q = ∑ r ∈ R, c r * twInc inc t r q := by
  intro h
  have hR : IsRepresentatives (id : Unit → Unit) ∅ :=
    ⟨fun i hi ↦ absurd hi (Finset.notMem_empty i), fun _ h ↦ absurd rfl h⟩
  have := h id not (fun _ _ ↦ 1) ∅ (fun _ ↦ rfl) (fun b ↦ by cases b <;> rfl) (fun _ _ ↦ rfl)
    hR (fun _ ↦ 1) (fun _ ↦ rfl) true (by decide)
  simp at this

/-- The twisted complex as a commutative square: in the coordinates `antiChainsEquiv` on the
`-1` eigenspaces in the two dimensions, the boundary becomes the boundary between the
representatives with the twisted incidences `[r : q] - [r : t q]`. This is the cellular chain
complex of `(N/G, (L ∪ N^G)/G)` with coefficients in the sign representation of the double
cover. Proposition `prop:z2_reconstruction`. -/
theorem antiChainsEquiv_comp_bd [DecidableEq I] [DecidableEq J] {Q : Finset J}
    (h2 : IsLeftRegular (2 : k)) (hs : Involutive s) (ht : Involutive t)
    (hinc : ∀ i j, inc (s i) (t j) = inc i j) (hR : IsRepresentatives s R)
    (hQ : IsRepresentatives t Q) :
    (antiChainsEquiv ht hQ h2).toLinearMap ∘ₗ
        (bd inc).restrict (fun _ hc ↦ bd_mem_antiChains hs hinc hc) =
      bd (fun (r : R) (q : Q) ↦ twInc inc t r q) ∘ₗ (antiChainsEquiv hs hR h2).toLinearMap := by
  refine LinearMap.ext fun c ↦ funext fun q ↦ ?_
  change bd inc (c : I → k) q = ∑ r : R, (c : I → k) r * twInc inc t r q
  rw [hR.bd_apply_antiChains_of_isLeftRegular h2 hs hinc c.2 q]
  exact (Finset.sum_coe_sort R _).symm

end TwistedComplex

section Untwisting

variable {t : J → J} {inc : I → J → k} {DI : Set I} {DJ : Set J}

/-- If the faces of the cells of `DI` lie in `DJ` and every `j ∈ DJ` with `t j ∈ DJ` is fixed,
then a cell of `DI` has no face `t q` for a free `q ∈ DJ`. -/
theorem inc_apply_eq_zero_of_mem (hface : ∀ i ∈ DI, ∀ j, inc i j ≠ 0 → j ∈ DJ)
    (hfix : ∀ j ∈ DJ, t j ∈ DJ → t j = j) {r : I} (hr : r ∈ DI) {q : J} (hq : q ∈ DJ)
    (hqt : t q ≠ q) : inc r (t q) = 0 := by
  by_contra h
  exact hqt (hfix q hq (hface r hr (t q) h))

/-- Untwisting: under the hypotheses of `inc_apply_eq_zero_of_mem`, the twisted incidence of
`r ∈ DI` and a free `q ∈ DJ` is `[r : q]`. For `N = D ∪ gD` with `D ∩ gD ⊂ L ∪ N^G`, `DI` and
`DJ` are the cells of `D` not in `L`. Proposition `prop:z2_reconstruction`, last claim. -/
theorem twInc_eq_of_mem (hface : ∀ i ∈ DI, ∀ j, inc i j ≠ 0 → j ∈ DJ)
    (hfix : ∀ j ∈ DJ, t j ∈ DJ → t j = j) {r : I} (hr : r ∈ DI) {q : J} (hq : q ∈ DJ)
    (hqt : t q ≠ q) : twInc inc t r q = inc r q := by
  rw [twInc, inc_apply_eq_zero_of_mem hface hfix hr hq hqt, sub_zero]

/-- Under the hypotheses of `twInc_eq_of_mem`, the twisted incidence equals the orbit-space
incidence `[r : q] + [r : t q]`. Proposition `prop:z2_reconstruction`, last claim. -/
theorem twInc_eq_add_of_mem (hface : ∀ i ∈ DI, ∀ j, inc i j ≠ 0 → j ∈ DJ)
    (hfix : ∀ j ∈ DJ, t j ∈ DJ → t j = j) {r : I} (hr : r ∈ DI) {q : J} (hq : q ∈ DJ)
    (hqt : t q ≠ q) : twInc inc t r q = inc r q + inc r (t q) := by
  rw [twInc, inc_apply_eq_zero_of_mem hface hfix hr hq hqt, sub_zero, add_zero]

variable {R : Finset I} {Q : Finset J}

/-- Untwisting for representatives `R ⊆ DI` and `Q ⊆ DJ`: every sign is `+1`, that is, the
twisted incidence of `r ∈ R` and `q ∈ Q` is `[r : q]`. Proposition `prop:z2_reconstruction`,
last claim: "choosing the representatives in `D` makes every sign `+1`". -/
theorem IsRepresentatives.twInc_eq (hQ : IsRepresentatives t Q) (hRD : ↑R ⊆ DI)
    (hQD : ↑Q ⊆ DJ) (hface : ∀ i ∈ DI, ∀ j, inc i j ≠ 0 → j ∈ DJ)
    (hfix : ∀ j ∈ DJ, t j ∈ DJ → t j = j) {r : I} (hr : r ∈ R) {q : J} (hq : q ∈ Q) :
    twInc inc t r q = inc r q :=
  twInc_eq_of_mem hface hfix (hRD hr) (hQD hq) (hQ.free q hq)

/-- The untwisted complex: for representatives `R ⊆ DI` and `Q ⊆ DJ`, the twisted boundary
between the representatives equals the boundary of the orbit space relative to the fixed cells,
with incidences `[r : q] + [r : t q]`, so the coefficients `k^w` can be replaced by `k`.
Proposition `prop:z2_reconstruction`, last claim. -/
theorem IsRepresentatives.bd_twInc_eq (hQ : IsRepresentatives t Q) (hRD : ↑R ⊆ DI)
    (hQD : ↑Q ⊆ DJ) (hface : ∀ i ∈ DI, ∀ j, inc i j ≠ 0 → j ∈ DJ)
    (hfix : ∀ j ∈ DJ, t j ∈ DJ → t j = j) :
    bd (fun (r : R) (q : Q) ↦ twInc inc t r q) =
      bd (fun (r : R) (q : Q) ↦ inc r q + inc r (t q)) := by
  congr 1
  funext r q
  exact twInc_eq_add_of_mem hface hfix (hRD r.2) (hQD q.2) (hQ.free q q.2)

end Untwisting

end LatentEquivariant
