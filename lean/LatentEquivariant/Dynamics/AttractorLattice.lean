import LatentEquivariant.Dynamics.Basic
import LatentEquivariant.Order.FiniteLatticeQuotient

namespace LatentEquivariant

open Set Topology

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}

/-- The omega-image of a sublattice of attracting blocks. -/
def omegaImage (f : X → X) (L : Sublattice (Set X)) :=
  {A : Set X // ∃ N ∈ L, omegaSet f N = A}

namespace omegaImage

variable (hf : Continuous f) (L : Sublattice (Set X))
  (hL : ∀ N ∈ L, IsAttractingBlock f N)

instance : PartialOrder (omegaImage f L) :=
  inferInstanceAs (PartialOrder {A : Set X // ∃ N ∈ L, omegaSet f N = A})

include hf hL

private theorem compact (A : omegaImage f L) : IsCompact (A.val) := by
  obtain ⟨N, hN, hNA⟩ := A.property
  rw [← hNA]
  exact isCompact_omegaSet hf (hL N hN).1 ((hL N hN).2.trans interior_subset)

private theorem invariant (A : omegaImage f L) : f '' A.val = A.val := by
  obtain ⟨N, hN, hNA⟩ := A.property
  rw [← hNA]
  exact image_omegaSet hf (hL N hN).1 ((hL N hN).2.trans interior_subset)

private def join (A B : omegaImage f L) : omegaImage f L := ⟨A.val ∪ B.val, by
  obtain ⟨N, hN, hNA⟩ := A.property
  rw [← hNA]
  obtain ⟨M, hM, hMB⟩ := B.property
  rw [← hMB]
  exact ⟨N ∪ M, L.sup_mem hN hM, omegaSet_union hf (hL N hN).1 (hL M hM).1
    ((hL N hN).2.trans interior_subset) ((hL M hM).2.trans interior_subset)⟩⟩

private def meet (A B : omegaImage f L) : omegaImage f L :=
  ⟨omegaSet f (A.val ∩ B.val), by
    obtain ⟨N, hN, hNA⟩ := A.property
    obtain ⟨M, hM, hMB⟩ := B.property
    rw [← hNA, ← hMB]
    exact ⟨N ∩ M, L.inf_mem hN hM, (omegaSet_inter_omegaSet hf (hL N hN).1
      (hL M hM).1 ((hL N hN).2.trans interior_subset)
      ((hL M hM).2.trans interior_subset)).symm⟩⟩

/-- Union is join, and omega of intersection is meet, on the omega-image lattice. -/
abbrev lattice : Lattice (omegaImage f L) where
  toPartialOrder := inferInstance
  sup := join hf L hL
  inf := meet hf L hL
  le_sup_left _ _ := subset_union_left
  le_sup_right _ _ := subset_union_right
  sup_le _ _ _ := union_subset
  inf_le_left A B := (omegaSet_subset hf
    ((compact hf L hL A).inter (compact hf L hL B))
    ((image_inter_subset _ _ _).trans (by
      rw [invariant hf L hL A, invariant hf L hL B]))).trans inter_subset_left
  inf_le_right A B := (omegaSet_subset hf
    ((compact hf L hL A).inter (compact hf L hL B))
    ((image_inter_subset _ _ _).trans (by
      rw [invariant hf L hL A, invariant hf L hL B]))).trans inter_subset_right
  le_inf A B C hAB hAC := image_eq_subset_omegaSet hf
    ((compact hf L hL B).inter (compact hf L hL C))
    ((image_inter_subset _ _ _).trans (by
      rw [invariant hf L hL B, invariant hf L hL C]))
    (invariant hf L hL A) (subset_inter hAB hAC)

/-- Omega is a surjective lattice homomorphism onto its image. -/
def hom : letI := lattice hf L hL; LatticeHom L (omegaImage f L) := by
  letI := lattice hf L hL
  exact {
    toFun := fun N ↦ ⟨omegaSet f N.val, N.val, N.property, rfl⟩
    map_sup' := fun N M ↦ Subtype.ext (omegaSet_union hf (hL N N.property).1
      (hL M M.property).1 ((hL N N.property).2.trans interior_subset)
      ((hL M M.property).2.trans interior_subset))
    map_inf' := fun N M ↦ Subtype.ext (omegaSet_inter_omegaSet hf (hL N N.property).1
      (hL M M.property).1 ((hL N N.property).2.trans interior_subset)
      ((hL M M.property).2.trans interior_subset)).symm }

theorem hom_surjective : Function.Surjective (hom hf L hL) := by
  rintro ⟨A, N, hN, rfl⟩
  exact ⟨⟨N, hN⟩, rfl⟩

/-- Distributivity descends along the surjective omega homomorphism. -/
abbrev distribLattice : DistribLattice (omegaImage f L) := by
  letI := lattice hf L hL
  apply DistribLattice.ofInfSupLe
  intro A B C
  obtain ⟨N, rfl⟩ := hom_surjective hf L hL A
  obtain ⟨M, rfl⟩ := hom_surjective hf L hL B
  obtain ⟨P, rfl⟩ := hom_surjective hf L hL C
  simp only [← map_sup, ← map_inf, inf_sup_left]
  exact le_rfl

theorem finite (hfinite : (L : Set (Set X)).Finite) : Finite (omegaImage f L) := by
  haveI : Finite L := hfinite.to_subtype
  exact Finite.of_surjective (hom hf L hL) (hom_surjective hf L hL)

end omegaImage

end LatentEquivariant


namespace LatentEquivariant

open Set Topology

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}

/-- The isolated invariant set of a block pair is the relative dual repeller. -/
theorem IsAttractingBlock.Inv_closure_sdiff_eq_attractorDual (hf : Continuous f)
    {N M : Set X} (hN : IsAttractingBlock f N) (hM : IsAttractingBlock f M) :
    Inv f (closure (N \ M)) = attractorDual f (omegaSet f N) (omegaSet f M) := by
  rw [hN.Inv_closure_sdiff_eq_blockDual hf hM, blockDual_eq_attractorDual hf hN hM]

namespace omegaImage

variable (hf : Continuous f) (L : Sublattice (Set X))
  (hL : ∀ N ∈ L, IsAttractingBlock f N) [BoundedOrder L]

/-- Bounds in the omega image are the images of the existing lattice bounds. -/
abbrev boundedOrder : BoundedOrder (omegaImage f L) where
  bot := hom hf L hL ⊥
  top := hom hf L hL ⊤
  bot_le A := by
    obtain ⟨N, rfl⟩ := hom_surjective hf L hL A
    exact omegaSet_mono (show (⊥ : L).val ⊆ N.val from (bot_le : (⊥ : L) ≤ N))
  le_top A := by
    obtain ⟨N, rfl⟩ := hom_surjective hf L hL A
    exact omegaSet_mono (show N.val ⊆ (⊤ : L).val from (le_top : N ≤ (⊤ : L)))

/-- The omega homomorphism preserves the finite lattice's bounds. -/
def boundedHom : letI := lattice hf L hL; letI := boundedOrder hf L hL;
    BoundedLatticeHom L (omegaImage f L) := by
  letI := lattice hf L hL
  letI := boundedOrder hf L hL
  exact { hom hf L hL with map_bot' := rfl, map_top' := rfl }

theorem boundedHom_surjective : Function.Surjective (boundedHom hf L hL) :=
  hom_surjective hf L hL

variable [Fintype L]

/-- The noncollapsed block join-irreducibles are the join-irreducibles of the attractor image. -/
noncomputable def survivingOrderIso :
    letI := lattice hf L hL
    letI := boundedOrder hf L hL
    {p : L // SurvivingJoinIrred (boundedHom hf L hL) p} ≃o
      {A : omegaImage f L // SupIrred A} := by
  letI := lattice hf L hL
  letI := boundedOrder hf L hL
  exact survivingJoinIrredOrderIso (boundedHom hf L hL) (boundedHom_surjective hf L hL)

/-- Survival is exactly nonemptiness of the invariant block difference. -/
theorem survives_iff_morse_nonempty (p : L) (hp : SupIrred p) :
    letI := lattice hf L hL
    letI := boundedOrder hf L hL
    SurvivingJoinIrred (boundedHom hf L hL) p ↔
      (Inv f (closure (p.val \ (lowerJoin p).val))).Nonempty := by
  letI := lattice hf L hL
  letI := boundedOrder hf L hL
  have hempty := (hL p p.property).Inv_closure_sdiff_empty_iff hf
    (hL ((lowerJoin p : L) : Set X) (lowerJoin p).property)
    (show (lowerJoin p).val ⊆ p.val from lowerJoin_le p)
  have hmono : omegaSet f (lowerJoin p).val ≤ omegaSet f p.val :=
    omegaSet_mono (lowerJoin_le p)
  change (SupIrred p ∧ omegaSet f (lowerJoin p).val < omegaSet f p.val) ↔ _
  rw [and_iff_right hp, lt_iff_le_and_ne, and_iff_right hmono,
    ne_eq, eq_comm, ← hempty]
  exact nonempty_iff_ne_empty.symm

end omegaImage

end LatentEquivariant
