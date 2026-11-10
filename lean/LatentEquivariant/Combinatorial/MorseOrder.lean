import LatentEquivariant.Combinatorial.OuterApprox

/-!
# The recurrent-component order on an orbit graph

Recurrent strongly connected components are ordered by reverse reachability: `P ≤ Q`
means that `Q` reaches `P`. For a finite group, the orbit poset of these components is
order-isomorphic to the recurrent-component poset of the quotient graph.
-/

namespace LatentEquivariant

open Set Pointwise MulAction Relation

/-- A strongly connected component containing a positive cycle. -/
def RecurrentComponent {X : Type*} (R : X → X → Prop) :=
  {C : Set X // ∃ x, TransGen R x x ∧ graphComponent R x = C}

namespace RecurrentComponent

variable {X : Type*} {R : X → X → Prop}

/-- A chosen recurrent vertex of the component. -/
noncomputable def point (C : RecurrentComponent R) : X := Classical.choose C.property

theorem point_recurrent (C : RecurrentComponent R) : TransGen R C.point C.point :=
  (Classical.choose_spec C.property).1

theorem component_point (C : RecurrentComponent R) : graphComponent R C.point = C.val :=
  (Classical.choose_spec C.property).2

theorem point_mem (C : RecurrentComponent R) : C.point ∈ C.val := by
  rw [← C.component_point]
  exact mem_graphComponent_self R C.point

theorem reachable_of_mem (C : RecurrentComponent R) {x y : X}
    (hx : x ∈ C.val) (hy : y ∈ C.val) : ReflTransGen R x y := by
  rw [← C.component_point] at hx hy
  exact hx.2.trans hy.1

/-- Reverse reachability on recurrent components. -/
noncomputable instance : PartialOrder (RecurrentComponent R) where
  le C D := ReflTransGen R D.point C.point
  le_refl _ := .refl
  le_trans _ _ _ hCD hDE := hDE.trans hCD
  le_antisymm C D hCD hDC := by
    apply Subtype.ext
    rw [← C.component_point, ← D.component_point]
    exact graphComponent_eq_of_mem R ⟨hDC, hCD⟩

theorem le_iff_reachable (C D : RecurrentComponent R) {x y : X}
    (hx : x ∈ C.val) (hy : y ∈ D.val) : C ≤ D ↔ ReflTransGen R y x := by
  constructor
  · intro h
    change ReflTransGen R D.point C.point at h
    exact (D.reachable_of_mem hy D.point_mem).trans
      (h.trans (C.reachable_of_mem C.point_mem hx))
  · intro h
    exact (D.reachable_of_mem D.point_mem hy).trans
      (h.trans (C.reachable_of_mem hx C.point_mem))

/-- The recurrent component of a specified vertex on a positive cycle. -/
def ofVertex (x : X) (hx : TransGen R x x) : RecurrentComponent R :=
  ⟨graphComponent R x, x, hx, rfl⟩

theorem ofVertex_le_ofVertex_iff {x y : X} (hx : TransGen R x x) (hy : TransGen R y y) :
    ofVertex x hx ≤ ofVertex y hy ↔ ReflTransGen R y x :=
  le_iff_reachable _ _ (mem_graphComponent_self R x) (mem_graphComponent_self R y)

end RecurrentComponent

section ComponentAction

variable {G X : Type*} [Group G] [MulAction G X] {R : X → X → Prop}
  (hR : ∀ (g : G) x y, R x y → R (g • x) (g • y))

/-- The given vertex action induces the usual action on recurrent components. -/
noncomputable abbrev recurrentComponentAction : MulAction G (RecurrentComponent R) where
  smul g C := ⟨g • C.val, g • C.point, transGen_smul hR C.point_recurrent g, by
    rw [graphComponent_smul hR, C.component_point]⟩
  one_smul C := Subtype.ext (one_smul G C.val)
  mul_smul g h C := Subtype.ext (mul_smul g h C.val)

end ComponentAction

section ComponentOrbits

variable {G X : Type*} [Group G] [MulAction G X] {R : X → X → Prop}
  [hEquiv : Fact (∀ (g : G) x y, R x y → R (g • x) (g • y))]

local notation "hR" => hEquiv.out

noncomputable local instance : MulAction G (RecurrentComponent R) := recurrentComponentAction hR

theorem recurrentComponent_smul_val (g : G) (C : RecurrentComponent R) :
    (g • C).val = g • C.val := rfl

theorem recurrentComponent_smul_le_smul (g : G) {C D : RecurrentComponent R}
    (h : C ≤ D) : g • C ≤ g • D := by
  apply (RecurrentComponent.le_iff_reachable (g • C) (g • D)
    (smul_mem_smul_set C.point_mem) (smul_mem_smul_set D.point_mem)).mpr
  exact reflTransGen_smul hR h g

variable [Finite G]

/-- The quotient image of a recurrent component is a recurrent quotient component. -/
noncomputable def quotientRecurrentComponent (C : RecurrentComponent R) :
    RecurrentComponent (orbitRelation G R) :=
  ⟨orbitMap G X '' C.val, orbitMap G X C.point,
    (transGen_orbitRelation_cycle_iff hR C.point).mpr C.point_recurrent, by
      rw [← image_graphComponent hR, C.component_point]⟩

@[simp] theorem quotientRecurrentComponent_val (C : RecurrentComponent R) :
    (quotientRecurrentComponent (G := G) (R := R) C).val = orbitMap G X '' C.val := rfl

theorem quotientRecurrentComponent_smul (g : G) (C : RecurrentComponent R) :
    quotientRecurrentComponent (G := G) (R := R) (g • C) =
      quotientRecurrentComponent (G := G) (R := R) C :=
  Subtype.ext (image_orbitMap_smul g C.val)

theorem quotientRecurrentComponent_eq_iff (C D : RecurrentComponent R) :
    quotientRecurrentComponent (G := G) (R := R) C =
      quotientRecurrentComponent (G := G) (R := R) D ↔
      ∃ g : G, D = g • C := by
  have hi : orbitMap G X '' C.val = orbitMap G X '' D.val ↔ ∃ g : G, D = g • C := by
    rw [← C.component_point, ← D.component_point, image_graphComponent_eq_iff hR]
    constructor
    · rintro ⟨g, hg⟩
      refine ⟨g, Subtype.ext ?_⟩
      simpa only [recurrentComponent_smul_val, C.component_point, D.component_point] using hg
    · rintro ⟨g, hg⟩
      refine ⟨g, ?_⟩
      simpa only [recurrentComponent_smul_val, C.component_point, D.component_point] using
        congrArg Subtype.val hg
  constructor
  · intro h
    exact hi.mp (congrArg Subtype.val h)
  · intro h
    exact Subtype.ext (hi.mpr h)

theorem quotientRecurrentComponent_surjective :
    Function.Surjective (quotientRecurrentComponent (G := G) (R := R)) := by
  intro D
  obtain ⟨x, hx⟩ := orbitMap_surjective D.point
  have hxrec : TransGen R x x :=
    (transGen_orbitRelation_cycle_iff hR x).mp (by rw [hx]; exact D.point_recurrent)
  refine ⟨RecurrentComponent.ofVertex x hxrec, Subtype.ext ?_⟩
  change orbitMap G X '' graphComponent R x = D.val
  rw [image_graphComponent hR, hx, D.component_point]

theorem quotientRecurrentComponent_le_iff (C D : RecurrentComponent R) :
    quotientRecurrentComponent (G := G) (R := R) C ≤
      quotientRecurrentComponent (G := G) (R := R) D ↔
      ∃ g : G, C ≤ g • D := by
  rw [RecurrentComponent.le_iff_reachable _ _
    (mem_image_of_mem _ C.point_mem) (mem_image_of_mem _ D.point_mem),
    reflTransGen_orbitRelation_iff hR]
  constructor
  · rintro ⟨g, hg⟩
    refine ⟨g⁻¹, (RecurrentComponent.le_iff_reachable C (g⁻¹ • D)
      C.point_mem (smul_mem_smul_set D.point_mem)).mpr ?_⟩
    simpa only [inv_smul_smul] using reflTransGen_smul hR hg g⁻¹
  · rintro ⟨g, hg⟩
    have hp := (RecurrentComponent.le_iff_reachable C (g • D)
      C.point_mem (smul_mem_smul_set D.point_mem)).mp hg
    exact ⟨g⁻¹, by simpa only [inv_smul_smul] using reflTransGen_smul hR hp g⁻¹⟩

/-- The induced map from component orbits to recurrent quotient components. -/
noncomputable def recurrentOrbitMap :
    orbitRel.Quotient G (RecurrentComponent R) → RecurrentComponent (orbitRelation G R) :=
  Quotient.lift (quotientRecurrentComponent (G := G) (R := R)) fun C D hCD ↦ by
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp (Quotient.sound hCD)
    rw [← hg, quotientRecurrentComponent_smul (G := G) (R := R)]

@[simp] theorem recurrentOrbitMap_orbitMap (C : RecurrentComponent R) :
    recurrentOrbitMap (G := G) (R := R) (orbitMap G (RecurrentComponent R) C) =
      quotientRecurrentComponent (G := G) (R := R) C := rfl

theorem recurrentOrbitMap_injective : Function.Injective (recurrentOrbitMap (G := G) (R := R)) := by
  intro P Q hPQ
  obtain ⟨C, rfl⟩ := orbitMap_surjective P
  obtain ⟨D, rfl⟩ := orbitMap_surjective Q
  obtain ⟨g, hg⟩ := (quotientRecurrentComponent_eq_iff (G := G) (R := R) C D).mp hPQ
  rw [hg, orbitMap_smul]

theorem recurrentOrbitMap_surjective :
    Function.Surjective (recurrentOrbitMap (G := G) (R := R)) := by
  intro D
  obtain ⟨C, hC⟩ := quotientRecurrentComponent_surjective (G := G) (R := R) D
  exact ⟨orbitMap G (RecurrentComponent R) C, hC⟩

/-- The orbit order, identified below with existence of an ordered pair of translates. -/
noncomputable abbrev recurrentComponentOrbitOrder :
    PartialOrder (orbitRel.Quotient G (RecurrentComponent R)) :=
  PartialOrder.lift (recurrentOrbitMap (G := G) (R := R))
    (recurrentOrbitMap_injective (G := G) (R := R))

noncomputable local instance : PartialOrder (orbitRel.Quotient G (RecurrentComponent R)) :=
  recurrentComponentOrbitOrder (G := G) (R := R)

noncomputable local instance : LE (orbitRel.Quotient G (RecurrentComponent R)) :=
  (recurrentComponentOrbitOrder (G := G) (R := R)).toLE

theorem recurrentComponent_orbit_le_iff (C D : RecurrentComponent R) :
    orbitMap G (RecurrentComponent R) C ≤ orbitMap G (RecurrentComponent R) D ↔
      ∃ g : G, C ≤ g • D := by
  change quotientRecurrentComponent (G := G) (R := R) C ≤
    quotientRecurrentComponent (G := G) (R := R) D ↔ _
  exact quotientRecurrentComponent_le_iff (G := G) (R := R) C D

/-- The orbit Morse poset is order-isomorphic to the Morse poset of the quotient graph. -/
noncomputable def recurrentComponentOrbitOrderIso :
    orbitRel.Quotient G (RecurrentComponent R) ≃o RecurrentComponent (orbitRelation G R) where
  toEquiv := Equiv.ofBijective (recurrentOrbitMap (G := G) (R := R))
    ⟨recurrentOrbitMap_injective (G := G) (R := R), recurrentOrbitMap_surjective (G := G) (R := R)⟩
  map_rel_iff' := by
    intro a b
    change recurrentOrbitMap (G := G) (R := R) a ≤ recurrentOrbitMap (G := G) (R := R) b ↔ _
    rfl

@[simp] theorem recurrentComponentOrbitOrderIso_apply (C : RecurrentComponent R) :
    recurrentComponentOrbitOrderIso (G := G) (R := R) (orbitMap G (RecurrentComponent R) C) =
      quotientRecurrentComponent (G := G) (R := R) C := by
  change recurrentOrbitMap (G := G) (R := R) (orbitMap G (RecurrentComponent R) C) = _
  rfl

end ComponentOrbits

end LatentEquivariant
