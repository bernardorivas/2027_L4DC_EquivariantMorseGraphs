import LatentEquivariant.Dynamics.AttractorLattice
import LatentEquivariant.Dynamics.TrajectoryLimits
import LatentEquivariant.Order.FiniteLatticeQuotient

/-!
# Morse representations from finite block lattices

The nonempty invariant differences associated to join-irreducible blocks form a Morse
representation, including the strict connecting-orbit condition for every full trajectory.
Its labels carry the induced order and correspond to irreducibles of the omega-image lattice.
-/

namespace LatentEquivariant

open Set Topology

section PointLabels

variable {X : Type*} (L : Sublattice (Set X)) [Fintype L] [BoundedOrder L]

/-- The least lattice member containing a point in the top block. -/
noncomputable def pointLabel (x : X) : L := by
  classical
  exact (Finset.univ.filter fun N : L ↦ x ∈ N.val).inf id

theorem mem_pointLabel {x : X} (hx : x ∈ (⊤ : L).val) : x ∈ (pointLabel L x).val := by
  classical
  have hmem : ∀ s : Finset L, (∀ N ∈ s, x ∈ N.val) → x ∈ (s.inf id).val := by
    intro s
    induction s using Finset.induction with
    | empty => simpa using hx
    | insert N s hNs ih =>
      intro hs
      rw [Finset.inf_insert]
      exact ⟨hs N (Finset.mem_insert_self _ _),
        ih (fun M hM ↦ hs M (Finset.mem_insert_of_mem hM))⟩
  exact hmem _ fun N hN ↦ (Finset.mem_filter.mp hN).2

theorem pointLabel_le {x : X} {N : L} (hx : x ∈ N.val) : pointLabel L x ≤ N := by
  classical
  exact Finset.inf_le (s := Finset.univ.filter fun M : L ↦ x ∈ M.val) (f := id)
    (Finset.mem_filter.mpr ⟨Finset.mem_univ N, hx⟩)

theorem pointLabel_le_iff {x : X} (hx : x ∈ (⊤ : L).val) (N : L) :
    pointLabel L x ≤ N ↔ x ∈ N.val :=
  ⟨fun h ↦ h (mem_pointLabel L hx), fun h ↦ pointLabel_le L h⟩

theorem pointLabel_supIrred (hbot : (⊥ : L).val = ∅) {x : X}
    (hx : x ∈ (⊤ : L).val) : SupIrred (pointLabel L x) := by
  have hmem := mem_pointLabel L hx
  refine ⟨?_, ?_⟩
  · intro hmin
    rw [hmin.eq_bot, hbot] at hmem
    exact hmem
  · intro N M hNM
    have hmem' : x ∈ (N ⊔ M : L).val := hNM.symm ▸ hmem
    change x ∈ N.val ∪ M.val at hmem'
    rcases hmem' with hN | hM
    · exact Or.inl (le_antisymm (hNM ▸ le_sup_left) (pointLabel_le L hN))
    · exact Or.inr (le_antisymm (hNM ▸ le_sup_right) (pointLabel_le L hM))

theorem mem_pointLabel_sdiff_lowerJoin (hbot : (⊥ : L).val = ∅) {x : X}
    (hx : x ∈ (⊤ : L).val) :
    x ∈ (pointLabel L x).val \ (lowerJoin (pointLabel L x)).val := by
  refine ⟨mem_pointLabel L hx, fun h ↦ ?_⟩
  exact (not_le_of_gt (lowerJoin_lt (pointLabel_supIrred L hbot hx))) (pointLabel_le L h)

end PointLabels

section FiniteAntitone

variable {P : Type*} [PartialOrder P] [Finite P]

theorem antitone_int_stabilizes {v : ℤ → P} (hv : Antitone v) :
    ∃ p q : P, p ≤ q ∧ (∀ k, p ≤ v k ∧ v k ≤ q) ∧
      (∃ n : ℕ, ∀ k : ℕ, n ≤ k → v k = p) ∧
      (∃ n : ℕ, ∀ k : ℤ, k ≤ -(n : ℤ) → v k = q) := by
  have hp : Antitone (fun n : ℕ ↦ v n) := fun _ _ h ↦ hv (by exact_mod_cast h)
  have hq : Monotone (fun n : ℕ ↦ v (-(n : ℤ))) := fun _ _ h ↦ hv (by omega)
  obtain ⟨n, hn⟩ := WellFoundedLT.antitone_chain_condition hp
  obtain ⟨m, hm⟩ := WellFoundedGT.monotone_chain_condition ⟨_, hq⟩
  change ∀ k : ℕ, m ≤ k → v (-(m : ℤ)) = v (-(k : ℤ)) at hm
  have hbounds : ∀ k : ℤ, v n ≤ v k ∧ v k ≤ v (-(m : ℤ)) := by
    intro k
    constructor
    · let j := max n k.toNat
      have hj : n ≤ j := le_max_left _ _
      rw [hn j hj]
      apply hv
      have : k.toNat ≤ j := le_max_right _ _
      omega
    · let j := max m (-k).toNat
      have hj : m ≤ j := le_max_left _ _
      rw [hm j hj]
      apply hv
      have : (-k).toNat ≤ j := le_max_right _ _
      omega
  refine ⟨v n, v (-(m : ℤ)), (hbounds n).2, hbounds,
    ⟨n, fun k hk ↦ (hn k hk).symm⟩, m, ?_⟩
  intro k hk
  have hkn : k = -((-k).toNat : ℤ) := by omega
  rw [hkn]
  exact (hm (-k).toNat (by omega)).symm

end FiniteAntitone

section Itinerary

variable {X : Type*} [TopologicalSpace X] {f : X → X}
  (L : Sublattice (Set X)) [Fintype L] [BoundedOrder L]
  (hL : ∀ N ∈ L, IsAttractingBlock f N)

include hL

theorem solution_pointLabel_antitone {γ : ℤ → X} (hγ : IsSolution f γ)
    (htop : ∀ k, γ k ∈ (⊤ : L).val) : Antitone (fun k ↦ pointLabel L (γ k)) := by
  apply antitone_int_of_succ_le
  intro k
  apply (pointLabel_le_iff L (htop (k + 1)) _).mpr
  rw [← hγ k]
  exact interior_subset ((hL _ (pointLabel L (γ k)).property).2
    (mem_image_of_mem f (mem_pointLabel L (htop k))))

/-- A full trajectory has ordered, eventually constant block labels at its two ends. -/
theorem exists_morse_itinerary (hbot : (⊥ : L).val = ∅) {γ : ℤ → X}
    (hγ : IsSolution f γ) (htop : ∀ k, γ k ∈ (⊤ : L).val) :
    ∃ p q : L, SupIrred p ∧ SupIrred q ∧ p ≤ q ∧
      (∃ n : ℕ, ∀ k : ℕ, n ≤ k → γ k ∈ p.val \ (lowerJoin p).val) ∧
      (∃ n : ℕ, ∀ k : ℤ, k ≤ -(n : ℤ) → γ k ∈ q.val \ (lowerJoin q).val) ∧
      (p = q → γ 0 ∈ Inv f (closure (p.val \ (lowerJoin p).val))) := by
  obtain ⟨p, q, hpq, hbounds, ⟨n, hn⟩, ⟨m, hm⟩⟩ :=
    antitone_int_stabilizes (solution_pointLabel_antitone L hL hγ htop)
  have hp : SupIrred p := hn n le_rfl ▸ pointLabel_supIrred L hbot (htop n)
  have hq : SupIrred q := hm (-(m : ℤ)) le_rfl ▸
    pointLabel_supIrred L hbot (htop (-(m : ℤ)))
  refine ⟨p, q, hp, hq, hpq, ⟨n, ?_⟩, ⟨m, ?_⟩, ?_⟩
  · intro k hk
    exact hn k hk ▸ mem_pointLabel_sdiff_lowerJoin L hbot (htop k)
  · intro k hk
    exact hm k hk ▸ mem_pointLabel_sdiff_lowerJoin L hbot (htop k)
  · intro hpq
    refine ⟨γ, hγ, rfl, fun k ↦ subset_closure ?_⟩
    have hlabel : pointLabel L (γ k) = p :=
      le_antisymm (hpq ▸ (hbounds k).2) (hbounds k).1
    exact hlabel ▸ mem_pointLabel_sdiff_lowerJoin L hbot (htop k)

end Itinerary

end LatentEquivariant


namespace LatentEquivariant

open Set Topology

section DisjointPieces

variable {X : Type*} (L : Sublattice (Set X)) [Fintype L] [BoundedOrder L]

theorem pointLabel_eq_of_mem_sdiff {p : L} {x : X}
    (hx : x ∈ p.val \ (lowerJoin p).val) : pointLabel L x = p := by
  have hle : pointLabel L x ≤ p := pointLabel_le L hx.1
  by_contra hne
  have hstrict : pointLabel L x < p := lt_of_le_of_ne hle hne
  have hxTop : x ∈ (⊤ : L).val := (le_top : p ≤ ⊤) hx.1
  exact hx.2 ((le_lowerJoin_of_lt hstrict) (mem_pointLabel L hxTop))

theorem disjoint_sdiff_lowerJoin {p q : L} (hpq : p ≠ q) :
    Disjoint (p.val \ (lowerJoin p).val) (q.val \ (lowerJoin q).val) := by
  apply Set.disjoint_left.mpr
  intro x hxp hxq
  exact hpq ((pointLabel_eq_of_mem_sdiff L hxp).symm.trans
    (pointLabel_eq_of_mem_sdiff L hxq))

end DisjointPieces

section MorsePieces

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}
  (L : Sublattice (Set X)) [Fintype L] [BoundedOrder L]
  (hL : ∀ N ∈ L, IsAttractingBlock f N)

/-- The invariant set isolated by a block and the join of its strict predecessors. -/
def morseBlock (f : X → X) (p : L) : Set X :=
  Inv f (closure (p.val \ (lowerJoin p).val))

include hL

theorem morseBlock_subset_sdiff (p : L) :
    morseBlock L f p ⊆ p.val \ (lowerJoin p).val := by
  rw [morseBlock, (hL p p.property).Inv_closure_sdiff
    (hL ((lowerJoin p : L) : Set X) (lowerJoin p).property)]
  exact Inv_subset

theorem morseBlock_isCompact (hf : Continuous f) (p : L) :
    IsCompact (morseBlock L f p) :=
  (hL p p.property).isCompact_Inv_closure_sdiff hf
    (hL ((lowerJoin p : L) : Set X) (lowerJoin p).property)

theorem morseBlock_subset_top_omega (hf : Continuous f) (p : L) :
    morseBlock L f p ⊆ omegaSet f (⊤ : L).val := by
  rw [morseBlock, (hL p p.property).Inv_closure_sdiff_eq_blockDual hf
    (hL ((lowerJoin p : L) : Set X) (lowerJoin p).property)]
  intro x hx
  exact omegaSet_mono (show p.val ⊆ (⊤ : L).val from (le_top : p ≤ ⊤)) hx.1

theorem morseBlock_disjoint {p q : L} (hpq : p ≠ q) :
    Disjoint (morseBlock L f p) (morseBlock L f q) :=
  (disjoint_sdiff_lowerJoin L hpq).mono (morseBlock_subset_sdiff L hL p)
    (morseBlock_subset_sdiff L hL q)

end MorsePieces

end LatentEquivariant


namespace LatentEquivariant

open Set Topology

section Representation

variable {X P : Type*} [TopologicalSpace X] [PartialOrder P]

/-- A finite Morse representation, including the full-trajectory connecting-orbit condition. -/
structure IsMorseRepresentation (f : X → X) (S : Set X) (M : P → Set X) : Prop where
  finite : Finite P
  nonempty : ∀ p, (M p).Nonempty
  compact : ∀ p, IsCompact (M p)
  invariant : ∀ p, f '' M p = M p
  subset : ∀ p, M p ⊆ S
  disjoint : Pairwise fun p q ↦ Disjoint (M p) (M q)
  connecting : ∀ γ : ℤ → X, IsSolution f γ → (∀ k, γ k ∈ S) →
    γ 0 ∉ ⋃ p, M p → ∃ p q, p < q ∧ omegaSet f {γ 0} ⊆ M p ∧ alphaSet γ ⊆ M q

end Representation

section BlockRepresentation

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}
  (L : Sublattice (Set X)) [Fintype L] [BoundedOrder L]
  (hL : ∀ N ∈ L, IsAttractingBlock f N)

omit [TopologicalSpace X] [T2Space X] [Fintype L] in
theorem blockLattice_bot_eq_empty (hempty : (∅ : Set X) ∈ L) : (⊥ : L).val = ∅ := by
  apply Subset.antisymm
  · exact (bot_le : (⊥ : L) ≤ ⟨∅, hempty⟩)
  · exact empty_subset _

/-- The labels of the nonempty invariant block differences, with the inherited lattice order. -/
abbrev MorseLabel (f : X → X) :=
  {p : L // SupIrred p ∧ (morseBlock L f p).Nonempty}

include hL

/-- A finite lattice of attracting blocks gives a Morse representation after empty pieces are
removed. The order is the order inherited from the block lattice. -/
theorem isMorseRepresentation_blocks (hf : Continuous f) (hempty : (∅ : Set X) ∈ L) :
    IsMorseRepresentation f (omegaSet f (⊤ : L).val)
      (fun p : MorseLabel L f ↦ morseBlock L f p.val) where
  finite := inferInstance
  nonempty p := p.property.2
  compact p := morseBlock_isCompact L hL hf p
  invariant p := image_Inv_eq f _
  subset p := morseBlock_subset_top_omega L hL hf p
  disjoint := by
    intro p q hpq
    apply morseBlock_disjoint L hL
    exact fun h ↦ hpq (Subtype.ext h)
  connecting := by
    intro γ hγ hγS hnot
    have htopBlock : IsAttractingBlock f (⊤ : L).val := hL _ (⊤ : L).property
    have htop : ∀ k, γ k ∈ (⊤ : L).val := fun k ↦
      omegaSet_subset hf htopBlock.1 (htopBlock.2.trans interior_subset) (hγS k)
    have hiter : ∀ n : ℕ, f^[n] (γ 0) ∈ (⊤ : L).val := fun n ↦
      (hγ.iterate n).symm ▸ htop n
    obtain ⟨p, q, hp, hq, hpq, hplus, hminus, heq⟩ :=
      exists_morse_itinerary L hL (blockLattice_bot_eq_empty L hempty) hγ htop
    have hplus' : ∃ n : ℕ, ∀ k : ℕ, n ≤ k →
        γ (k : ℤ) ∈ closure (p.val \ (lowerJoin p).val) := by
      obtain ⟨n, hn⟩ := hplus
      exact ⟨n, fun k hk ↦ subset_closure (hn k hk)⟩
    have hminus' : ∃ n : ℕ, ∀ k : ℤ, k ≤ -(n : ℤ) →
        γ k ∈ closure (q.val \ (lowerJoin q).val) := by
      obtain ⟨n, hn⟩ := hminus
      exact ⟨n, fun k hk ↦ subset_closure (hn k hk)⟩
    have homega : omegaSet f {γ 0} ⊆ morseBlock L f p :=
      omegaSet_singleton_subset_Inv_of_eventually hf hγ htopBlock.1 htop
        isClosed_closure hplus'
    have halpha : alphaSet γ ⊆ morseBlock L f q :=
      alphaSet_subset_Inv_of_eventually hf hγ htopBlock.1 htop isClosed_closure hminus'
    have hpne : (morseBlock L f p).Nonempty :=
      (omegaSet_singleton_nonempty htopBlock.1 hiter).mono homega
    have hqne : (morseBlock L f q).Nonempty :=
      (alphaSet_nonempty htopBlock.1 htop).mono halpha
    refine ⟨⟨p, hp, hpne⟩, ⟨q, hq, hqne⟩, ?_, homega, halpha⟩
    change p < q
    refine lt_of_le_of_ne hpq fun hpq' ↦ ?_
    exact hnot (mem_iUnion.mpr ⟨⟨p, hp, hpne⟩, heq hpq'⟩)

end BlockRepresentation

end LatentEquivariant


namespace LatentEquivariant

open Set Topology

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}
  (hf : Continuous f) (L : Sublattice (Set X)) [Fintype L] [BoundedOrder L]
  (hL : ∀ N ∈ L, IsAttractingBlock f N)

/-- The nonempty Morse labels are exactly the surviving join-irreducibles. -/
noncomputable def morseLabelSurvivorOrderIso :
    letI := omegaImage.lattice hf L hL
    letI := omegaImage.boundedOrder hf L hL
    MorseLabel L f ≃o {p : L // SurvivingJoinIrred (omegaImage.boundedHom hf L hL) p} := by
  letI := omegaImage.lattice hf L hL
  letI := omegaImage.boundedOrder hf L hL
  exact {
    toFun := fun p ↦ ⟨p.val,
      (omegaImage.survives_iff_morse_nonempty hf L hL p.val p.property.1).mpr p.property.2⟩
    invFun := fun p ↦ ⟨p.val, p.property.1,
      (omegaImage.survives_iff_morse_nonempty hf L hL p.val p.property.1).mp p.property⟩
    left_inv := fun _ ↦ rfl
    right_inv := fun _ ↦ rfl
    map_rel_iff' := Iff.rfl }

/-- The Morse representation has the induced order on the attractor-image irreducibles. -/
noncomputable def morseLabelAttractorOrderIso :
    letI := omegaImage.lattice hf L hL
    letI := omegaImage.boundedOrder hf L hL
    MorseLabel L f ≃o {A : omegaImage f L // SupIrred A} := by
  letI := omegaImage.lattice hf L hL
  letI := omegaImage.boundedOrder hf L hL
  exact (morseLabelSurvivorOrderIso hf L hL).trans (omegaImage.survivingOrderIso hf L hL)

end LatentEquivariant
