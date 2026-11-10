import LatentEquivariant.Orbit.Quotient

/-!
# Outer approximations on orbit representatives

Lemma `lem:equivariant_outer`. The top-dimensional cells of
the grid are an abstract type `C` on which `G` acts, with an equivariant realization
`cell : C → Set X`, `cell (g • ξ) = g • cell ξ`. A combinatorial multivalued map is
`F : C → Set C`, and `|S|` is the union of the cells in `S`.

* `exists_equivariant_outerApprox`: an outer approximation on a set `Ξ` of orbit
  representatives whose values are invariant under the stabilizers extends, by
  `F(g ξ) := g F(ξ)`, to an outer approximation on all cells.
* `stabilizerSaturation`: if a value is not invariant under the stabilizer, the union over the
  stabilizer is, and it is still an outer approximation at that cell.

Neither finiteness of `G` nor the grid condition is used, only that `G` acts by homeomorphisms
and maps cells to cells. The paper's "one cell of each `G`-orbit" must be read as exactly one
(`hΞu`): otherwise `F(g ξ) := g F(ξ)` is not well defined. Without uniqueness and the stabilizer
hypothesis an outer approximation on all cells still exists (`exists_outerApprox_of_reps`), but
the formula `F(g ξ) = g F(ξ)` is lost.
-/

namespace LatentEquivariant

open Set Topology Pointwise MulAction

variable {G X C : Type*} [Group G] [MulAction G X] [MulAction G C]

/-- `|S|`, the union of the cells in `S` (Section `sec:combinatorial`). -/
def realize (cell : C → Set X) (S : Set C) : Set X :=
  ⋃ ξ ∈ S, cell ξ

theorem realize_mono {cell : C → Set X} {S T : Set C} (h : S ⊆ T) :
    realize cell S ⊆ realize cell T :=
  biUnion_subset_biUnion_left h

theorem realize_smul {cell : C → Set X} (hcell : ∀ (g : G) (ξ : C), cell (g • ξ) = g • cell ξ)
    (g : G) (S : Set C) : realize cell (g • S) = g • realize cell S := by
  ext x
  simp only [realize, mem_iUnion, exists_prop, mem_smul_set_iff_inv_smul_mem]
  constructor
  · rintro ⟨η, hη, hx⟩
    refine ⟨g⁻¹ • η, hη, ?_⟩
    rw [← mem_smul_set_iff_inv_smul_mem, ← hcell, smul_inv_smul]
    exact hx
  · rintro ⟨ξ, hξ, hx⟩
    refine ⟨g • ξ, by rwa [inv_smul_smul], ?_⟩
    rw [hcell, mem_smul_set_iff_inv_smul_mem]
    exact hx

theorem IsEquivariant.image_smul_set {f : X → X} (hf : IsEquivariant G f) (g : G) (A : Set X) :
    f '' (g • A) = g • (f '' A) := by
  rw [← image_smul, ← image_smul, image_image, image_image]
  simp only [hf g]

variable [TopologicalSpace X]

/-- `F` is an outer approximation of `f` on the cells in `Ξ`: `f(ξ) ⊆ int |F(ξ)|` for `ξ ∈ Ξ`
(Section `sec:combinatorial`). -/
def IsOuterApproxOn (cell : C → Set X) (f : X → X) (F : C → Set C) (Ξ : Set C) : Prop :=
  ∀ ξ ∈ Ξ, f '' cell ξ ⊆ interior (realize cell (F ξ))

/-- Lemma `lem:equivariant_outer`. Let `Ξ` contain exactly one cell of each orbit, and let `F₀` be
an outer approximation of the equivariant map `f` on `Ξ` with `s F₀(ξ) = F₀(ξ)` for `s ∈ G_ξ`.
Then `F(g ξ) := g F₀(ξ)` is well defined and is an outer approximation on all cells. -/
theorem exists_equivariant_outerApprox [ContinuousConstSMul G X] {cell : C → Set X}
    (hcell : ∀ (g : G) (ξ : C), cell (g • ξ) = g • cell ξ) {f : X → X} (hf : IsEquivariant G f)
    {Ξ : Set C} (hΞ : ∀ η : C, ∃ ξ ∈ Ξ, ∃ g : G, g • ξ = η)
    (hΞu : ∀ ξ ∈ Ξ, ∀ ξ' ∈ Ξ, ∀ g : G, g • ξ = ξ' → ξ = ξ')
    {F₀ : C → Set C} (hF₀ : IsOuterApproxOn cell f F₀ Ξ)
    (hstab : ∀ ξ ∈ Ξ, ∀ s : G, s • ξ = ξ → s • F₀ ξ = F₀ ξ) :
    ∃ F : C → Set C, (∀ ξ ∈ Ξ, ∀ g : G, F (g • ξ) = g • F₀ ξ) ∧ IsOuterApproxOn cell f F univ := by
  choose ρ hρΞ γ hγ using hΞ
  refine ⟨fun η ↦ γ η • F₀ (ρ η), ?_, ?_⟩
  · intro ξ hξ g
    have hρ : ρ (g • ξ) = ξ := by
      refine (hΞu ξ hξ (ρ (g • ξ)) (hρΞ _) ((γ (g • ξ))⁻¹ * g) ?_).symm
      rw [mul_smul, inv_smul_eq_iff]
      exact (hγ (g • ξ)).symm
    have hs : (g⁻¹ * γ (g • ξ)) • ξ = ξ := by
      have := hγ (g • ξ)
      rw [hρ] at this
      rw [mul_smul, this, inv_smul_smul]
    show γ (g • ξ) • F₀ (ρ (g • ξ)) = g • F₀ ξ
    rw [hρ]
    conv_lhs => rw [← mul_inv_cancel_left g (γ (g • ξ))]
    rw [mul_smul, hstab ξ hξ _ hs]
  · intro η _
    have h := hF₀ (ρ η) (hρΞ η)
    show f '' cell η ⊆ interior (realize cell (γ η • F₀ (ρ η)))
    conv_lhs => rw [← hγ η]
    rw [hcell, hf.image_smul_set, realize_smul hcell, interior_smul]
    exact smul_set_mono h

/-- Without uniqueness of the representatives and without the stabilizer hypothesis, translating
the values on representatives still gives an outer approximation on all cells, but not the
formula `F(g ξ) = g F(ξ)`. -/
theorem exists_outerApprox_of_reps [ContinuousConstSMul G X] {cell : C → Set X}
    (hcell : ∀ (g : G) (ξ : C), cell (g • ξ) = g • cell ξ) {f : X → X} (hf : IsEquivariant G f)
    {Ξ : Set C} (hΞ : ∀ η : C, ∃ ξ ∈ Ξ, ∃ g : G, g • ξ = η) {F₀ : C → Set C}
    (hF₀ : IsOuterApproxOn cell f F₀ Ξ) : ∃ F : C → Set C, IsOuterApproxOn cell f F univ := by
  choose ρ hρΞ γ hγ using hΞ
  refine ⟨fun η ↦ γ η • F₀ (ρ η), fun η _ ↦ ?_⟩
  have h := hF₀ (ρ η) (hρΞ η)
  show f '' cell η ⊆ interior (realize cell (γ η • F₀ (ρ η)))
  conv_lhs => rw [← hγ η]
  rw [hcell, hf.image_smul_set, realize_smul hcell, interior_smul]
  exact smul_set_mono h

/-- The union of the translates of `S` by the stabilizer of `ξ`. -/
def stabilizerSaturation (G : Type*) [Group G] [MulAction G C] (ξ : C) (S : Set C) : Set C :=
  ⋃ s ∈ stabilizer G ξ, s • S

theorem subset_stabilizerSaturation (ξ : C) (S : Set C) : S ⊆ stabilizerSaturation G ξ S := by
  intro η hη
  exact mem_biUnion (one_mem (stabilizer G ξ)) (by simpa using hη)

theorem smul_stabilizerSaturation {ξ : C} {t : G} (ht : t • ξ = ξ) (S : Set C) :
    t • stabilizerSaturation G ξ S = stabilizerSaturation G ξ S := by
  have htm : t ∈ stabilizer G ξ := ht
  ext η
  simp only [stabilizerSaturation, smul_set_iUnion₂, mem_iUnion, exists_prop, smul_smul]
  constructor
  · rintro ⟨s, hs, hη⟩
    exact ⟨t * s, mul_mem htm hs, hη⟩
  · rintro ⟨s, hs, hη⟩
    refine ⟨t⁻¹ * s, mul_mem (inv_mem htm) hs, ?_⟩
    rwa [mul_inv_cancel_left]

omit [MulAction G X] in
/-- If `F₀(ξ)` is not invariant under `G_ξ`, its saturation over `G_ξ` is, and it is still an
outer approximation at `ξ`. -/
theorem outerApprox_stabilizerSaturation {cell : C → Set X} {f : X → X} {ξ : C} {S : Set C}
    (h : f '' cell ξ ⊆ interior (realize cell S)) :
    f '' cell ξ ⊆ interior (realize cell (stabilizerSaturation G ξ S)) ∧
      ∀ t : G, t • ξ = ξ → t • stabilizerSaturation G ξ S = stabilizerSaturation G ξ S :=
  ⟨h.trans (interior_mono (realize_mono (subset_stabilizerSaturation ξ S))),
    fun _ ht ↦ smul_stabilizerSaturation ht S⟩

end LatentEquivariant

namespace LatentEquivariant

open Set Topology Pointwise MulAction

section QuotientOuterApprox

variable {G X C : Type*} [Group G] [MulAction G X] [MulAction G C]
  [TopologicalSpace X] [ContinuousConstSMul G X]

omit [TopologicalSpace X] [ContinuousConstSMul G X] in
theorem realize_quotientSetMap {cell : C → Set X} (hcell : IsEquivariant G cell) (S : Set C) :
    realize (quotientSetMap cell hcell) (orbitMap G C '' S) =
      orbitMap G X '' realize cell S := by
  ext q
  constructor
  · intro hq
    obtain ⟨c, hc, hqc⟩ := mem_iUnion₂.mp hq
    obtain ⟨ξ, hξ, rfl⟩ := hc
    obtain ⟨x, hx, rfl⟩ := hqc
    exact ⟨x, mem_iUnion₂.mpr ⟨ξ, hξ, hx⟩, rfl⟩
  · rintro ⟨x, hx, rfl⟩
    obtain ⟨ξ, hξ, hx⟩ := mem_iUnion₂.mp hx
    exact mem_iUnion₂.mpr ⟨orbitMap G C ξ, ⟨ξ, hξ, rfl⟩, x, hx, rfl⟩

/-- An equivariant outer approximation descends to the quotient cell realization. -/
theorem IsOuterApproxOn.quotient {cell : C → Set X} (hcell : IsEquivariant G cell)
    {f : X → X} (hf : IsEquivariant G f) {F : C → Set C} (hF : IsEquivariant G F)
    (h : IsOuterApproxOn cell f F univ) :
    IsOuterApproxOn (quotientSetMap cell hcell) (quotientMap f hf)
      (quotientSetMap F hF) univ := by
  intro q _
  obtain ⟨ξ, rfl⟩ := orbitMap_surjective q
  rw [quotientSetMap_orbitMap, quotientSetMap_orbitMap, realize_quotientSetMap hcell,
    image_quotientMap_image]
  exact (image_mono (h ξ (mem_univ ξ))).trans
    (isOpenMap_orbitMap.image_interior_subset _)

end QuotientOuterApprox

end LatentEquivariant

namespace LatentEquivariant

open Set Pointwise MulAction Relation

section GraphQuotient

variable {G X : Type*} [Group G] [MulAction G X] {R : X → X → Prop}

/-- The directed relation induced on vertex orbits. -/
def orbitRelation (G : Type*) [Group G] [MulAction G X] (R : X → X → Prop) :
    orbitRel.Quotient G X → orbitRel.Quotient G X → Prop :=
  Relation.Map R (orbitMap G X) (orbitMap G X)

variable (hR : ∀ (g : G) x y, R x y → R (g • x) (g • y))

include hR

theorem reflTransGen_smul {x y : X} (h : ReflTransGen R x y) (g : G) :
    ReflTransGen R (g • x) (g • y) :=
  h.lift (fun z ↦ g • z) (hR g)

theorem transGen_smul {x y : X} (h : TransGen R x y) (g : G) :
    TransGen R (g • x) (g • y) :=
  h.lift (fun z ↦ g • z) (hR g)

theorem orbitRelation_step_lift (x : X) {q : orbitRel.Quotient G X}
    (h : orbitRelation G R (orbitMap G X x) q) :
    ∃ y, R x y ∧ orbitMap G X y = q := by
  obtain ⟨a, b, hab, ha, hb⟩ := h
  obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp ha
  refine ⟨g⁻¹ • b, ?_, (orbitMap_smul _ _).trans hb⟩
  have h' := hR g⁻¹ a b hab
  simpa only [← hg, inv_smul_smul] using h'

theorem reflTransGen_orbitRelation_lift {x : X} {q : orbitRel.Quotient G X}
    (h : ReflTransGen (orbitRelation G R) (orbitMap G X x) q) :
    ∃ y, ReflTransGen R x y ∧ orbitMap G X y = q := by
  induction h with
  | refl => exact ⟨x, .refl, rfl⟩
  | tail _ hstep ih =>
    obtain ⟨y, hxy, hy⟩ := ih
    rw [← hy] at hstep
    obtain ⟨z, hyz, hz⟩ := orbitRelation_step_lift hR y hstep
    exact ⟨z, hxy.tail hyz, hz⟩

theorem transGen_orbitRelation_lift {x : X} {q : orbitRel.Quotient G X}
    (h : TransGen (orbitRelation G R) (orbitMap G X x) q) :
    ∃ y, TransGen R x y ∧ orbitMap G X y = q := by
  induction h with
  | single hstep =>
    obtain ⟨y, hxy, hy⟩ := orbitRelation_step_lift hR x hstep
    exact ⟨y, .single hxy, hy⟩
  | tail _ hstep ih =>
    obtain ⟨y, hxy, hy⟩ := ih
    rw [← hy] at hstep
    obtain ⟨z, hyz, hz⟩ := orbitRelation_step_lift hR y hstep
    exact ⟨z, hxy.tail hyz, hz⟩

theorem reflTransGen_orbitRelation_iff (x y : X) :
    ReflTransGen (orbitRelation G R) (orbitMap G X x) (orbitMap G X y) ↔
      ∃ g : G, ReflTransGen R x (g • y) := by
  constructor
  · intro h
    obtain ⟨z, hxz, hz⟩ := reflTransGen_orbitRelation_lift hR h
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp hz
    exact ⟨g, hg.symm ▸ hxz⟩
  · rintro ⟨g, hg⟩
    have h : ReflTransGen (orbitRelation G R) (orbitMap G X x)
        (orbitMap G X (g • y)) :=
      hg.lift (orbitMap G X) (fun a b hab ↦ ⟨a, b, hab, rfl, rfl⟩)
    simpa only [orbitMap_smul] using h

theorem transGen_orbitRelation_iff (x y : X) :
    TransGen (orbitRelation G R) (orbitMap G X x) (orbitMap G X y) ↔
      ∃ g : G, TransGen R x (g • y) := by
  constructor
  · intro h
    obtain ⟨z, hxz, hz⟩ := transGen_orbitRelation_lift hR h
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp hz
    exact ⟨g, hg.symm ▸ hxz⟩
  · rintro ⟨g, hg⟩
    have h : TransGen (orbitRelation G R) (orbitMap G X x)
        (orbitMap G X (g • y)) :=
      hg.lift (orbitMap G X) (fun a b hab ↦ ⟨a, b, hab, rfl, rfl⟩)
    simpa only [orbitMap_smul] using h

theorem reflTransGen_smul_pow {x : X} {g : G} (h : ReflTransGen R x (g • x)) (n : ℕ) :
    ReflTransGen R x (g ^ n • x) := by
  induction n with
  | zero => simpa using (ReflTransGen.refl (r := R) (a := x))
  | succ n ih =>
    have h' := reflTransGen_smul hR h (g ^ n)
    simpa only [← mul_smul, ← pow_succ] using ih.trans h'

variable [Finite G]

theorem reflTransGen_smul_reverse {x : X} {g : G} (h : ReflTransGen R x (g • x)) :
    ReflTransGen R (g • x) x := by
  obtain ⟨n, hn⟩ := Nat.exists_eq_succ_of_ne_zero (orderOf_pos g).ne'
  have hn' : n + 1 = orderOf g := hn.symm
  have h' := reflTransGen_smul hR (reflTransGen_smul_pow hR h n) g
  simpa only [← mul_smul, ← pow_succ', hn', pow_orderOf_eq_one, one_smul] using h'

/-- A positive quotient cycle lifts to a positive cycle at every chosen lift of its vertex. -/
theorem transGen_orbitRelation_cycle_iff (x : X) :
    TransGen (orbitRelation G R) (orbitMap G X x) (orbitMap G X x) ↔ TransGen R x x := by
  constructor
  · intro h
    obtain ⟨g, hg⟩ := (transGen_orbitRelation_iff hR x x).mp h
    exact hg.trans_left (reflTransGen_smul_reverse hR hg.to_reflTransGen)
  · intro h
    exact (transGen_orbitRelation_iff hR x x).mpr ⟨1, by simpa using h⟩

end GraphQuotient

section Components

variable {X : Type*} (R : X → X → Prop)

/-- The strongly connected component of a vertex, including a transient singleton. -/
def graphComponent (x : X) : Set X :=
  {y | ReflTransGen R x y ∧ ReflTransGen R y x}

theorem mem_graphComponent_self (x : X) : x ∈ graphComponent R x := ⟨.refl, .refl⟩

theorem graphComponent_eq_of_mem {x y : X} (h : y ∈ graphComponent R x) :
    graphComponent R x = graphComponent R y := by
  ext z
  exact ⟨fun hz ↦ ⟨h.2.trans hz.1, hz.2.trans h.1⟩,
    fun hz ↦ ⟨h.1.trans hz.1, hz.2.trans h.2⟩⟩

variable {R} {G : Type*} [Group G] [MulAction G X]
  (hR : ∀ (g : G) x y, R x y → R (g • x) (g • y))

include hR

theorem graphComponent_smul (g : G) (x : X) :
    graphComponent R (g • x) = g • graphComponent R x := by
  ext y
  constructor
  · intro hy
    refine ⟨g⁻¹ • y, ?_, smul_inv_smul g y⟩
    exact ⟨by simpa using reflTransGen_smul hR hy.1 g⁻¹,
      by simpa using reflTransGen_smul hR hy.2 g⁻¹⟩
  · rintro ⟨y, hy, rfl⟩
    exact ⟨reflTransGen_smul hR hy.1 g, reflTransGen_smul hR hy.2 g⟩

variable [Finite G]

theorem mem_quotient_graphComponent_iff (x y : X) :
    orbitMap G X y ∈ graphComponent (orbitRelation G R) (orbitMap G X x) ↔
      ∃ g : G, g • y ∈ graphComponent R x := by
  constructor
  · rintro ⟨hxy, hyx⟩
    obtain ⟨a, ha⟩ := (reflTransGen_orbitRelation_iff hR x y).mp hxy
    obtain ⟨b, hb⟩ := (reflTransGen_orbitRelation_iff hR y x).mp hyx
    have hba : ReflTransGen R (a • y) ((a * b) • x) := by
      simpa only [mul_smul] using reflTransGen_smul hR hb a
    have hcycle := ha.trans hba
    exact ⟨a, ha, hba.trans (reflTransGen_smul_reverse hR hcycle)⟩
  · rintro ⟨g, hg⟩
    constructor
    · exact (reflTransGen_orbitRelation_iff hR x y).mpr ⟨g, hg.1⟩
    · have h' := reflTransGen_smul hR hg.2 g⁻¹
      exact (reflTransGen_orbitRelation_iff hR y x).mpr
        ⟨g⁻¹, by simpa only [inv_smul_smul] using h'⟩

theorem image_graphComponent (x : X) :
    orbitMap G X '' graphComponent R x =
      graphComponent (orbitRelation G R) (orbitMap G X x) := by
  ext q
  obtain ⟨y, rfl⟩ := orbitMap_surjective q
  constructor
  · rintro ⟨z, hz, hzy⟩
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp hzy
    exact (mem_quotient_graphComponent_iff hR x y).mpr ⟨g, hg.symm ▸ hz⟩
  · intro h
    obtain ⟨g, hg⟩ := (mem_quotient_graphComponent_iff hR x y).mp h
    exact ⟨g • y, hg, orbitMap_smul g y⟩

omit [Finite G] in
theorem image_graphComponent_eq_iff (x y : X) :
    orbitMap G X '' graphComponent R x = orbitMap G X '' graphComponent R y ↔
      ∃ g : G, graphComponent R y = g • graphComponent R x := by
  constructor
  · intro h
    have hy : orbitMap G X y ∈ orbitMap G X '' graphComponent R x := by
      rw [h]
      exact ⟨y, mem_graphComponent_self R y, rfl⟩
    obtain ⟨z, hz, hzy⟩ := hy
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp hzy
    have hc := graphComponent_eq_of_mem R hz
    rw [← hg, graphComponent_smul hR] at hc
    exact ⟨g⁻¹, by rw [hc, inv_smul_smul]⟩
  · rintro ⟨g, hg⟩
    rw [hg, image_orbitMap_smul]

/-- On a recurrent component, positive path endpoints identify its setwise stabilizer. -/
theorem mem_stabilizer_graphComponent_iff {x : X} (hx : TransGen R x x) (g : G) :
    g ∈ stabilizer G (graphComponent R x) ↔ TransGen R x (g • x) := by
  constructor
  · intro hg
    change g • graphComponent R x = graphComponent R x at hg
    have hmem : g • x ∈ graphComponent R x := by
      rw [← hg]
      exact smul_mem_smul_set (mem_graphComponent_self R x)
    rcases reflTransGen_iff_eq_or_transGen.mp hmem.1 with heq | hpath
    · simpa only [heq] using hx
    · exact hpath
  · intro hg
    have hmem : g • x ∈ graphComponent R x :=
      ⟨hg.to_reflTransGen, reflTransGen_smul_reverse hR hg.to_reflTransGen⟩
    change g • graphComponent R x = graphComponent R x
    rw [← graphComponent_smul hR]
    exact (graphComponent_eq_of_mem R hmem).symm

omit [Finite G] in
theorem stabilizer_graphComponent_smul (g : G) (x : X) :
    stabilizer G (graphComponent R (g • x)) =
      (stabilizer G (graphComponent R x)).map (MulAut.conj g).toMonoidHom := by
  rw [graphComponent_smul hR, stabilizer_smul_eq_stabilizer_map_conj]

end Components

section Voltages

variable {G X Q : Type*} [Group G] [MulAction G X]
  (R : X → X → Prop) (rep : Q → X)

/-- An edge from `q` to `r` carries every group element translating `rep r` into the image. -/
abbrev voltageQuiver : Quiver Q where
  Hom q r := {g : G // R (rep q) (g • rep r)}

abbrev VoltagePath (q r : Q) := @Quiver.Path Q (voltageQuiver (G := G) R rep) q r

def voltageLength {q r : Q} (p : VoltagePath (G := G) R rep q r) : ℕ :=
  @Quiver.Path.length Q (voltageQuiver (G := G) R rep) q r p

def voltageComp {q r s : Q} (p : VoltagePath (G := G) R rep q r)
    (p' : VoltagePath (G := G) R rep r s) : VoltagePath (G := G) R rep q s :=
  @Quiver.Path.comp Q (voltageQuiver (G := G) R rep) q r s p p'

/-- Edge voltages multiply in path order. -/
def pathVoltage {q r : Q} (p : VoltagePath (G := G) R rep q r) : G :=
  @Quiver.Path.weight Q (voltageQuiver (G := G) R rep) G _ (fun {_ _} e ↦ e.val) q r p

theorem pathVoltage_comp {q r s : Q} (p : VoltagePath (G := G) R rep q r)
    (p' : VoltagePath (G := G) R rep r s) :
    pathVoltage (G := G) R rep (voltageComp (G := G) R rep p p') =
      pathVoltage (G := G) R rep p * pathVoltage (G := G) R rep p' := by
  letI := voltageQuiver (G := G) R rep
  exact Quiver.Path.weight_comp _ _ _

variable {R rep} (hR : ∀ (g : G) x y, R x y → R (g • x) (g • y))

include hR

theorem voltagePath_lift {q r : Q} (p : VoltagePath (G := G) R rep q r) :
    ReflTransGen R (rep q) (pathVoltage (G := G) R rep p • rep r) := by
  induction p with
  | nil => simpa [pathVoltage, Quiver.Path.weight] using ReflTransGen.refl (r := R)
  | @cons r s p e ih =>
    have h := hR (pathVoltage (G := G) R rep p) (rep r) (e.val • rep s) e.property
    simpa only [pathVoltage, Quiver.Path.weight, mul_smul] using ih.tail h

theorem voltagePath_lift_pos {q r : Q} (p : VoltagePath (G := G) R rep q r)
    (hp : 0 < voltageLength (G := G) R rep p) :
    TransGen R (rep q) (pathVoltage (G := G) R rep p • rep r) := by
  letI := voltageQuiver (G := G) R rep
  cases p with
  | nil => simp [voltageLength, Quiver.Path.length] at hp
  | cons p e =>
    have h := hR (pathVoltage (G := G) R rep p) _ _ e.property
    have h' := TransGen.tail' (voltagePath_lift hR p) h
    simpa only [pathVoltage, Quiver.Path.weight, mul_smul] using h'

end Voltages

section VoltageCompleteness

variable {G X : Type*} [Group G] [MulAction G X] {R : X → X → Prop}
  (hR : ∀ (g : G) x y, R x y → R (g • x) (g • y))
  (rep : orbitRel.Quotient G X → X) (hrep : ∀ q, orbitMap G X (rep q) = q)

include hR hrep

theorem exists_voltagePath_of_transGen {x y : X} (h : TransGen R x y) :
    ∀ q r g, x = rep q → y = g • rep r →
      ∃ p : VoltagePath (G := G) R rep q r,
        0 < voltageLength (G := G) R rep p ∧ pathVoltage (G := G) R rep p = g := by
  letI := voltageQuiver (G := G) R rep
  induction h with
  | single hxy =>
    intro q r g hx hy
    rw [hx, hy] at hxy
    exact ⟨Quiver.Path.nil.cons ⟨g, hxy⟩, by simp [voltageLength, Quiver.Path.length], by
      simp [pathVoltage]⟩
  | @tail y z hxy hyz ih =>
    intro q r g hx hz
    obtain ⟨k, hk⟩ := orbitMap_eq_iff.mp (hrep (orbitMap G X y)).symm
    obtain ⟨p, hp, hpk⟩ := ih q (orbitMap G X y) k hx hk.symm
    have hedge : R (rep (orbitMap G X y)) ((k⁻¹ * g) • rep r) := by
      have h' := hR k⁻¹ y z hyz
      rw [← hk, hz] at h'
      simpa only [inv_smul_smul, mul_smul] using h'
    refine ⟨p.cons ⟨k⁻¹ * g, hedge⟩, by simp [voltageLength, Quiver.Path.length], ?_⟩
    unfold pathVoltage at hpk ⊢
    simp only [Quiver.Path.weight_cons, hpk, mul_inv_cancel_left]

theorem exists_voltagePath_iff (q r : orbitRel.Quotient G X) (g : G) :
    (∃ p : VoltagePath (G := G) R rep q r,
      0 < voltageLength (G := G) R rep p ∧ pathVoltage (G := G) R rep p = g) ↔
      TransGen R (rep q) (g • rep r) := by
  constructor
  · rintro ⟨p, hp, hpg⟩
    simpa only [hpg] using voltagePath_lift_pos hR p hp
  · intro h
    exact exists_voltagePath_of_transGen hR rep hrep h q r g rfl rfl

variable [Finite G]

/-- Closed positive walks carry exactly the elements of the component stabilizer. -/
theorem closed_voltagePath_iff_stabilizer (q : orbitRel.Quotient G X)
    (hq : TransGen R (rep q) (rep q)) (g : G) :
    (∃ p : VoltagePath (G := G) R rep q q,
      0 < voltageLength (G := G) R rep p ∧ pathVoltage (G := G) R rep p = g) ↔
      g ∈ stabilizer G (graphComponent R (rep q)) := by
  rw [exists_voltagePath_iff hR rep hrep, mem_stabilizer_graphComponent_iff hR hq]

end VoltageCompleteness

end LatentEquivariant

namespace LatentEquivariant

open Set Pointwise MulAction

section InvariantImages

variable {G X C : Type*} [Group G] [MulAction G X] [MulAction G C]

theorem isGInvariant_preimage_orbitMap (B : Set (orbitRel.Quotient G X)) :
    IsGInvariant G (orbitMap G X ⁻¹' B) := by
  intro g x hx
  simpa only [mem_preimage, orbitMap_smul] using hx

theorem IsGInvariant.image_inter {A B : Set X} (hA : IsGInvariant G A) :
    orbitMap G X '' (A ∩ B) = orbitMap G X '' A ∩ orbitMap G X '' B := by
  ext q
  constructor
  · rintro ⟨x, hx, rfl⟩
    exact ⟨⟨x, hx.1, rfl⟩, ⟨x, hx.2, rfl⟩⟩
  · rintro ⟨⟨a, ha, haq⟩, b, hb, hbq⟩
    refine ⟨b, ⟨?_, hb⟩, hbq⟩
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp (hbq.trans haq.symm)
    exact hg ▸ hA g a ha

theorem IsGInvariant.realize {cell : C → Set X} (hcell : IsEquivariant G cell)
    {A : Set C} (hA : IsGInvariant G A) : IsGInvariant G (realize cell A) := by
  intro g x hx
  obtain ⟨ξ, hξ, hx⟩ := mem_iUnion₂.mp hx
  refine mem_iUnion₂.mpr ⟨g • ξ, hA g ξ hξ, ?_⟩
  rw [hcell]
  exact smul_mem_smul_set hx

end InvariantImages

section GraphLattice

variable {G X : Type*} [Group G] [MulAction G X] {F : X → Set X}
  (hF : IsEquivariant G F)

theorem realize_preimage_quotientSetMap (B : Set (orbitRel.Quotient G X)) :
    realize F (orbitMap G X ⁻¹' B) =
      orbitMap G X ⁻¹' realize (quotientSetMap F hF) B := by
  have hinv := (isGInvariant_preimage_orbitMap B).realize hF
  rw [← hinv.preimage_image_orbitMap, ← realize_quotientSetMap hF,
    image_preimage_eq B orbitMap_surjective]

theorem realize_iterate_preimage_quotientSetMap (B : Set (orbitRel.Quotient G X)) (n : ℕ) :
    (realize F)^[n] (orbitMap G X ⁻¹' B) =
      orbitMap G X ⁻¹' (realize (quotientSetMap F hF))^[n] B := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ih,
      realize_preimage_quotientSetMap hF]

/-- The omega-limit set of a multivalued map on vertices. -/
def graphOmega (F : X → Set X) (A : Set X) : Set X :=
  ⋂ n : ℕ, ⋃ k ≥ n, (realize F)^[k] A

theorem graphOmega_preimage_quotientSetMap (B : Set (orbitRel.Quotient G X)) :
    graphOmega F (orbitMap G X ⁻¹' B) =
      orbitMap G X ⁻¹' graphOmega (quotientSetMap F hF) B := by
  simp only [graphOmega, preimage_iInter, preimage_iUnion,
    realize_iterate_preimage_quotientSetMap hF]

theorem image_graphOmega {A : Set X} (hA : IsGInvariant G A) :
    orbitMap G X '' graphOmega F A =
      graphOmega (quotientSetMap F hF) (orbitMap G X '' A) := by
  conv_lhs => rw [← hA.preimage_image_orbitMap]
  rw [graphOmega_preimage_quotientSetMap hF, image_preimage_eq _ orbitMap_surjective]

theorem image_graphOmega_inter {A B : Set X} (hA : IsGInvariant G A)
    (hB : IsGInvariant G B) :
    orbitMap G X '' graphOmega F (A ∩ B) =
      graphOmega (quotientSetMap F hF) (orbitMap G X '' A ∩ orbitMap G X '' B) := by
  rw [image_graphOmega hF (hA.inter hB), hA.image_inter]

theorem forwardInvariant_image_iff {A : Set X} (hA : IsGInvariant G A) :
    realize (quotientSetMap F hF) (orbitMap G X '' A) ⊆ orbitMap G X '' A ↔
      realize F A ⊆ A := by
  rw [realize_quotientSetMap hF]
  constructor
  · intro h x hx
    have hx' : x ∈ orbitMap G X ⁻¹' (orbitMap G X '' A) := h ⟨x, hx, rfl⟩
    rwa [hA.preimage_image_orbitMap] at hx'
  · exact image_mono

theorem graphAttractor_image_iff {A : Set X} (hA : IsGInvariant G A) :
    realize (quotientSetMap F hF) (orbitMap G X '' A) = orbitMap G X '' A ↔
      realize F A = A := by
  rw [realize_quotientSetMap hF]
  constructor
  · intro h
    have h' := congrArg (preimage (orbitMap G X)) h
    simpa only [(hA.realize hF).preimage_image_orbitMap, hA.preimage_image_orbitMap] using h'
  · exact congrArg (image (orbitMap G X))

/-- Forward-invariant orbit sets correspond to invariant forward-invariant vertex sets. -/
def forwardInvariantOrbitOrderIso :
    {A : Set X // IsGInvariant G A ∧ realize F A ⊆ A} ≃o
      {B : Set (orbitRel.Quotient G X) // realize (quotientSetMap F hF) B ⊆ B} where
  toFun A := ⟨orbitMap G X '' A, (forwardInvariant_image_iff hF A.property.1).mpr A.property.2⟩
  invFun B := ⟨orbitMap G X ⁻¹' B, isGInvariant_preimage_orbitMap _, by
    rw [realize_preimage_quotientSetMap hF]
    exact preimage_mono B.property⟩
  left_inv A := Subtype.ext A.property.1.preimage_image_orbitMap
  right_inv B := Subtype.ext (image_preimage_eq _ orbitMap_surjective)
  map_rel_iff' := by
    intro A B
    constructor
    · intro h x hx
      have hx' : x ∈ orbitMap G X ⁻¹' (orbitMap G X '' (B : Set X)) := h ⟨x, hx, rfl⟩
      rwa [B.property.1.preimage_image_orbitMap] at hx'
    · exact image_mono

/-- Graph attractors on the quotient correspond to invariant graph attractors upstairs. -/
def graphAttractorOrbitOrderIso :
    {A : Set X // IsGInvariant G A ∧ realize F A = A} ≃o
      {B : Set (orbitRel.Quotient G X) // realize (quotientSetMap F hF) B = B} where
  toFun A := ⟨orbitMap G X '' A, (graphAttractor_image_iff hF A.property.1).mpr A.property.2⟩
  invFun B := ⟨orbitMap G X ⁻¹' B, isGInvariant_preimage_orbitMap _, by
    rw [realize_preimage_quotientSetMap hF, B.property]⟩
  left_inv A := Subtype.ext A.property.1.preimage_image_orbitMap
  right_inv B := Subtype.ext (image_preimage_eq _ orbitMap_surjective)
  map_rel_iff' := by
    intro A B
    constructor
    · intro h x hx
      have hx' : x ∈ orbitMap G X ⁻¹' (orbitMap G X '' (B : Set X)) := h ⟨x, hx, rfl⟩
      rwa [B.property.1.preimage_image_orbitMap] at hx'
    · exact image_mono

end GraphLattice

section GraphOmega

variable {X : Type*} {F : X → Set X} {A B : Set X}

theorem realize_union (F : X → Set X) (A B : Set X) :
    realize F (A ∪ B) = realize F A ∪ realize F B := by
  ext x
  simp only [realize, mem_iUnion, exists_prop, mem_union]
  aesop

theorem realize_iterate_antitone (hA : realize F A ⊆ A) :
    Antitone (fun n : ℕ ↦ (realize F)^[n] A) := by
  apply antitone_nat_of_succ_le
  intro n
  rw [Function.iterate_succ_apply]
  exact (Monotone.iterate (fun _ _ h ↦ realize_mono h) n) hA

theorem graphOmega_eq_iInter_iterate (hA : realize F A ⊆ A) :
    graphOmega F A = ⋂ n : ℕ, (realize F)^[n] A := by
  have hant := realize_iterate_antitone hA
  unfold graphOmega
  congr 1
  funext n
  ext x
  constructor
  · intro hx
    obtain ⟨k, hk, hx⟩ := mem_iUnion₂.mp hx
    exact hant hk hx
  · intro hx
    exact mem_iUnion₂.mpr ⟨n, le_rfl, hx⟩

theorem graphOmega_subset (hA : realize F A ⊆ A) : graphOmega F A ⊆ A := by
  rw [graphOmega_eq_iInter_iterate hA]
  exact iInter_subset (fun n : ℕ ↦ (realize F)^[n] A) 0

theorem graphAttractor_subset_graphOmega (hB : realize F B = B) (hBA : B ⊆ A) :
    B ⊆ graphOmega F A := by
  have hiter : ∀ n : ℕ, B ⊆ (realize F)^[n] A := by
    intro n
    induction n with
    | zero => exact hBA
    | succ n ih =>
      rw [Function.iterate_succ_apply']
      exact hB ▸ realize_mono ih
  intro x hx
  exact mem_iInter.mpr fun n ↦ mem_iUnion₂.mpr ⟨n, le_rfl, hiter n hx⟩

variable [Finite X]

theorem graphOmega_eq_iterate (hA : realize F A ⊆ A) :
    ∃ n : ℕ, graphOmega F A = (realize F)^[n] A ∧
      ∀ m, n ≤ m → (realize F)^[n] A = (realize F)^[m] A := by
  have hant := realize_iterate_antitone hA
  obtain ⟨n, hn⟩ := WellFoundedLT.antitone_chain_condition hant
  refine ⟨n, ?_, hn⟩
  rw [graphOmega_eq_iInter_iterate hA]
  apply Subset.antisymm (iInter_subset _ n)
  intro x hx
  apply mem_iInter.mpr
  intro m
  rcases le_total m n with hmn | hnm
  · exact hant hmn hx
  · rwa [← hn m hnm]

theorem realize_graphOmega (hA : realize F A ⊆ A) :
    realize F (graphOmega F A) = graphOmega F A := by
  obtain ⟨n, hn, hstable⟩ := graphOmega_eq_iterate hA
  rw [hn, ← Function.iterate_succ_apply' (realize F) n A]
  exact (hstable (n + 1) (Nat.le_succ n)).symm

/-- The omega-limit set is the greatest graph attractor contained in a forward-invariant set. -/
theorem graphOmega_isGreatest (hA : realize F A ⊆ A) :
    IsGreatest {B : Set X | realize F B = B ∧ B ⊆ A} (graphOmega F A) :=
  ⟨⟨realize_graphOmega hA, graphOmega_subset hA⟩,
    fun _ hB ↦ graphAttractor_subset_graphOmega hB.1 hB.2⟩

/-- The graph-attractor meet is the omega-limit set of the intersection. -/
theorem graphAttractor_meet_isGreatest (hA : realize F A = A) (hB : realize F B = B) :
    IsGreatest {C : Set X | realize F C = C ∧ C ⊆ A ∩ B} (graphOmega F (A ∩ B)) := by
  apply graphOmega_isGreatest
  intro x hx
  exact ⟨hA ▸ realize_mono inter_subset_left hx, hB ▸ realize_mono inter_subset_right hx⟩

end GraphOmega

end LatentEquivariant
