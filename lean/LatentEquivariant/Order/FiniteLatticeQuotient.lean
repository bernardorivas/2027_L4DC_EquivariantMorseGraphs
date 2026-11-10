import Mathlib.Order.Irreducible
import Mathlib.Order.Hom.BoundedLattice
import Mathlib.Order.Preorder.Finite

/-!
# Join-irreducibles under a finite lattice epimorphism

The surviving join-irreducibles are those whose predecessor has strictly smaller image.
The homomorphism restricts to an order isomorphism from these elements onto the
join-irreducibles of its codomain. It also preserves their predecessors.
-/

namespace LatentEquivariant

section Predecessor

variable {L : Type*} [Fintype L] [SemilatticeSup L] [OrderBot L]

/-- The join of all elements strictly below `p`. For a join-irreducible it is its predecessor. -/
noncomputable def lowerJoin (p : L) : L := by
  classical
  exact (Finset.univ.filter (fun x ↦ x < p)).sup id

theorem lowerJoin_le (p : L) : lowerJoin p ≤ p := by
  classical
  exact Finset.sup_le fun x hx ↦ (Finset.mem_filter.mp hx).2.le

theorem le_lowerJoin_of_lt {x p : L} (h : x < p) : x ≤ lowerJoin p := by
  classical
  change x ≤ (Finset.univ.filter (fun y ↦ y < p)).sup id
  exact Finset.le_sup (s := Finset.univ.filter (fun y : L ↦ y < p)) (f := id) (by simp [h])

theorem lowerJoin_lt {p : L} (hp : SupIrred p) : lowerJoin p < p := by
  classical
  refine lt_of_le_of_ne (lowerJoin_le p) ?_
  intro heq
  obtain ⟨x, hx, hxp⟩ := hp.finset_sup_eq heq
  have hlt : x < p := (Finset.mem_filter.mp hx).2
  exact hlt.ne hxp

theorem lowerJoin_isGreatest {p : L} (hp : SupIrred p) :
    IsGreatest {x : L | x < p} (lowerJoin p) :=
  ⟨lowerJoin_lt hp, fun _ hx ↦ le_lowerJoin_of_lt hx⟩

end Predecessor

section Epimorphism

variable {L A : Type*} [Fintype L] [Lattice L] [Lattice A]
  [BoundedOrder L] [BoundedOrder A]
  (h : BoundedLatticeHom L A)

/-- A join-irreducible survives when the homomorphism does not collapse its predecessor. -/
def SurvivingJoinIrred (p : L) : Prop :=
  SupIrred p ∧ h (lowerJoin p) < h p

theorem survivor_le_of_image_le {p x : L} (hp : SurvivingJoinIrred h p)
    (hx : h p ≤ h x) : p ≤ x := by
  by_contra hnot
  have hlt : p ⊓ x < p := lt_of_le_of_ne inf_le_left (by
    intro heq
    exact hnot (heq ▸ inf_le_right))
  have himage : h (p ⊓ x) = h p := by rw [map_inf, inf_eq_left.mpr hx]
  have hle := (OrderHomClass.monotone h) (le_lowerJoin_of_lt hlt)
  rw [himage] at hle
  exact (not_le_of_gt hp.2) hle

theorem survivor_image_supIrred (hsurj : Function.Surjective h) {p : L}
    (hp : SurvivingJoinIrred h p) : SupIrred (h p) := by
  refine ⟨?_, ?_⟩
  · intro hmin
    exact hp.2.ne (hmin.eq_of_ge ((OrderHomClass.monotone h) (lowerJoin_le p))).symm
  · intro a b hab
    obtain ⟨x, rfl⟩ := hsurj a
    obtain ⟨y, rfl⟩ := hsurj b
    have heq : (p ⊓ x) ⊔ (p ⊓ y) = p := by
      apply le_antisymm
      · exact sup_le inf_le_left inf_le_left
      · apply survivor_le_of_image_le h hp
        rw [map_sup, map_inf, map_inf]
        have hx : h x ≤ h p := hab ▸ le_sup_left
        have hy : h y ≤ h p := hab ▸ le_sup_right
        rw [inf_eq_right.mpr hx, inf_eq_right.mpr hy, hab]
    rcases hp.1.2 heq with hx | hy
    · left
      apply le_antisymm (hab ▸ le_sup_left)
      exact (OrderHomClass.monotone h) (hx ▸ inf_le_right)
    · right
      apply le_antisymm (hab ▸ le_sup_right)
      exact (OrderHomClass.monotone h) (hy ▸ inf_le_right)

theorem exists_least_preimage (hsurj : Function.Surjective h) (a : A) :
    ∃ p : L, h p = a ∧ ∀ x, a ≤ h x → p ≤ x := by
  obtain ⟨x, hx⟩ := hsurj a
  obtain ⟨p, hp⟩ := (Set.toFinite {x : L | h x = a}).exists_minimal ⟨x, hx⟩
  refine ⟨p, hp.1, ?_⟩
  intro y hy
  have hmeet : h (p ⊓ y) = a := by rw [map_inf, hp.1, inf_eq_left.mpr hy]
  have hpeq : p = p ⊓ y := hp.eq_of_ge hmeet inf_le_left
  exact hpeq ▸ inf_le_right

theorem least_preimage_survives {a : A} (ha : SupIrred a) {p : L}
    (hpa : h p = a) (hleast : ∀ x, a ≤ h x → p ≤ x) : SurvivingJoinIrred h p := by
  have hp : SupIrred p := by
    refine ⟨?_, ?_⟩
    · intro hmin
      have hpbot : p = ⊥ := hmin.eq_bot
      exact ha.ne_bot (by rw [← hpa, hpbot, map_bot])
    · intro x y hxy
      have hmap : h x ⊔ h y = a := by rw [← map_sup, hxy, hpa]
      rcases ha.2 hmap with hx | hy
      · left
        exact le_antisymm (hxy ▸ le_sup_left) (hleast x hx.ge)
      · right
        exact le_antisymm (hxy ▸ le_sup_right) (hleast y hy.ge)
  refine ⟨hp, lt_of_le_of_ne ((OrderHomClass.monotone h) (lowerJoin_le p)) ?_⟩
  intro heq
  have hple : p ≤ lowerJoin p := hleast _ (by rw [heq, hpa])
  exact (not_le_of_gt (lowerJoin_lt hp)) hple

theorem survivor_image_lowerJoin [Fintype A] (hsurj : Function.Surjective h) {p : L}
    (hp : SurvivingJoinIrred h p) : h (lowerJoin p) = lowerJoin (h p) := by
  classical
  apply le_antisymm (le_lowerJoin_of_lt hp.2)
  apply Finset.sup_le
  intro a ha
  have hlt : a < h p := (Finset.mem_filter.mp ha).2
  obtain ⟨x, rfl⟩ := hsurj a
  have hmeet : h (x ⊓ p) = h x := by rw [map_inf, inf_eq_left.mpr hlt.le]
  have hxp : x ⊓ p < p := lt_of_le_of_ne inf_le_right (by
    intro heq
    have : h x = h p := by rw [← hmeet, heq]
    exact hlt.ne this)
  have hle := (OrderHomClass.monotone h) (le_lowerJoin_of_lt hxp)
  rwa [hmeet] at hle

/-- The surviving join-irreducibles correspond to all join-irreducibles of the quotient. -/
noncomputable def survivingJoinIrredOrderIso (hsurj : Function.Surjective h) :
    {p : L // SurvivingJoinIrred h p} ≃o {a : A // SupIrred a} := by
  let f : {p : L // SurvivingJoinIrred h p} → {a : A // SupIrred a} :=
    fun p ↦ ⟨h p, survivor_image_supIrred h hsurj p.property⟩
  have hinj : Function.Injective f := by
    intro p q heq
    apply Subtype.ext
    have heq' : h p = h q := congrArg Subtype.val heq
    exact le_antisymm (survivor_le_of_image_le h p.property heq'.le)
      (survivor_le_of_image_le h q.property heq'.ge)
  have hsur : Function.Surjective f := by
    intro a
    obtain ⟨p, hpa, hleast⟩ := exists_least_preimage h hsurj a
    refine ⟨⟨p, least_preimage_survives h a.property hpa hleast⟩, ?_⟩
    exact Subtype.ext hpa
  exact
    { toEquiv := Equiv.ofBijective f ⟨hinj, hsur⟩
      map_rel_iff' := by
        intro p q
        exact ⟨survivor_le_of_image_le h p.property,
          fun hle ↦ (OrderHomClass.monotone h) hle⟩ }

@[simp] theorem survivingJoinIrredOrderIso_apply (hsurj : Function.Surjective h)
    (p : {p : L // SurvivingJoinIrred h p}) :
    (survivingJoinIrredOrderIso h hsurj p : A) = h p := rfl

end Epimorphism

end LatentEquivariant
