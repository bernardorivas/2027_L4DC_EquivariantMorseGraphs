import LatentEquivariant.Dynamics.Basic

/-!
# Limit sets of trajectories in a compact set

Backward and forward tail limits are nonempty and invariant. Eventual containment in a closed
set places each limit in that set's maximal invariant subset.
-/

namespace LatentEquivariant

open Set Topology

variable {X : Type*} [TopologicalSpace X] [T2Space X]
variable {f : X → X} {γ : ℤ → X} {N K : Set X}

/-- The alpha-limit set of a full trajectory, defined by its closed backward tails. -/
def alphaSet (γ : ℤ → X) : Set X :=
  ⋂ n : ℕ, closure {x | ∃ k : ℤ, k ≤ -(n : ℤ) ∧ γ k = x}

private def backwardTail (γ : ℤ → X) (n : ℕ) : Set X :=
  {x | ∃ k : ℤ, k ≤ -(n : ℤ) ∧ γ k = x}

omit [T2Space X] in
private theorem backwardTail_antitone : Antitone (fun n ↦ closure (backwardTail γ n)) := by
  intro i j hij
  apply closure_mono
  rintro x ⟨k, hk, rfl⟩
  exact ⟨k, by omega, rfl⟩

private theorem backwardTail_compact (hN : IsCompact N) (hγN : ∀ k : ℤ, γ k ∈ N)
    (n : ℕ) : IsCompact (closure (backwardTail γ n)) := by
  apply hN.of_isClosed_subset isClosed_closure
  apply closure_minimal _ hN.isClosed
  rintro x ⟨k, _, rfl⟩
  exact hγN k

private theorem image_closure_of_compact_container (hf : Continuous f) (hN : IsCompact N)
    {A : Set X} (hA : A ⊆ N) : f '' closure A = closure (f '' A) := by
  have hc : IsCompact (closure A) :=
    hN.of_isClosed_subset isClosed_closure (closure_minimal hA hN.isClosed)
  exact Subset.antisymm (image_closure_subset_closure_image hf)
    (closure_minimal (image_mono subset_closure) (hc.image hf).isClosed)

omit [TopologicalSpace X] [T2Space X] in
private theorem image_backwardTail (hγ : IsSolution f γ) (n : ℕ) :
    f '' backwardTail γ (n + 1) = backwardTail γ n := by
  apply Subset.antisymm
  · rintro _ ⟨_, ⟨k, hk, rfl⟩, rfl⟩
    exact ⟨k + 1, by omega, (hγ k).symm⟩
  · rintro _ ⟨k, hk, rfl⟩
    refine ⟨γ (k - 1), ⟨k - 1, by omega, rfl⟩, ?_⟩
    simpa using hγ (k - 1)

/-- A full trajectory contained in a compact set has a nonempty alpha-limit set. -/
theorem alphaSet_nonempty (hN : IsCompact N) (hγN : ∀ k : ℤ, γ k ∈ N) :
    (alphaSet γ).Nonempty := by
  apply IsCompact.nonempty_iInter_of_sequence_nonempty_isCompact_isClosed
    (fun n ↦ closure (backwardTail γ n))
  · intro n
    exact backwardTail_antitone (Nat.le_succ n)
  · intro n
    exact ⟨γ (-(n : ℤ)), subset_closure ⟨-(n : ℤ), le_rfl, rfl⟩⟩
  · exact backwardTail_compact hN hγN 0
  · intro n
    exact isClosed_closure

/-- Continuity and compact containment make the alpha-limit set invariant. -/
theorem image_alphaSet (hf : Continuous f) (hγ : IsSolution f γ) (hN : IsCompact N)
    (hγN : ∀ k : ℤ, γ k ∈ N) : f '' alphaSet γ = alphaSet γ := by
  have hshift : (⋂ n : ℕ, closure (backwardTail γ (n + 1))) = alphaSet γ :=
    backwardTail_antitone.iInter_nat_add 1
  rw [← hshift, image_iInter_of_compact_antitone hf _
    (fun i j hij ↦ backwardTail_antitone (Nat.add_le_add_right hij 1))
    (fun n ↦ backwardTail_compact hN hγN (n + 1))]
  rw [hshift]
  change (⋂ n : ℕ, f '' closure (backwardTail γ (n + 1))) =
    ⋂ n : ℕ, closure (backwardTail γ n)
  congr 1
  funext n
  rw [image_closure_of_compact_container hf hN (by
    rintro x ⟨k, _, rfl⟩
    exact hγN k), image_backwardTail hγ]

private def forwardTail (f : X → X) (x : X) (n : ℕ) : Set X :=
  {y | ∃ k : ℕ, n ≤ k ∧ f^[k] x = y}

omit [T2Space X] in
private theorem forwardTail_antitone {x : X} :
    Antitone (fun n ↦ closure (forwardTail f x n)) := by
  intro i j hij
  apply closure_mono
  rintro y ⟨k, hk, rfl⟩
  exact ⟨k, le_trans hij hk, rfl⟩

omit [T2Space X] in
private theorem omegaSet_singleton_eq_forwardTail (x : X) :
    omegaSet f {x} = ⋂ n : ℕ, closure (forwardTail f x n) := by
  unfold omegaSet
  congr 1
  funext n
  congr 1
  ext y
  simp [forwardTail, eq_comm]

omit [TopologicalSpace X] [T2Space X] in
private theorem image_forwardTail (x : X) (n : ℕ) :
    f '' forwardTail f x n = forwardTail f x (n + 1) := by
  apply Subset.antisymm
  · rintro _ ⟨_, ⟨k, hk, rfl⟩, rfl⟩
    exact ⟨k + 1, by omega, Function.iterate_succ_apply' f k x⟩
  · rintro _ ⟨k, hk, rfl⟩
    obtain ⟨j, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : k ≠ 0)
    refine ⟨f^[j] x, ⟨j, by omega, rfl⟩, ?_⟩
    exact (Function.iterate_succ_apply' f j x).symm

/-- The omega-limit of a forward orbit contained in a compact set is invariant. -/
theorem image_omegaSet_singleton (hf : Continuous f) (hN : IsCompact N) {x : X}
    (hx : ∀ n : ℕ, f^[n] x ∈ N) : f '' omegaSet f {x} = omegaSet f {x} := by
  have htail : ∀ n, forwardTail f x n ⊆ N := by
    rintro n y ⟨k, _, rfl⟩
    exact hx k
  have hc : ∀ n, IsCompact (closure (forwardTail f x n)) := fun n ↦
    hN.of_isClosed_subset isClosed_closure (closure_minimal (htail n) hN.isClosed)
  rw [omegaSet_singleton_eq_forwardTail, image_iInter_of_compact_antitone hf _
    forwardTail_antitone hc]
  simp_rw [image_closure_of_compact_container hf hN (htail _), image_forwardTail]
  exact forwardTail_antitone.iInter_nat_add 1

omit [T2Space X] in
/-- A closed set containing a backward tail contains the alpha-limit set. -/
theorem alphaSet_subset_of_eventually (hK : IsClosed K)
    (hevent : ∃ n : ℕ, ∀ k : ℤ, k ≤ -(n : ℤ) → γ k ∈ K) : alphaSet γ ⊆ K := by
  obtain ⟨n, hn⟩ := hevent
  refine (iInter_subset (fun j : ℕ ↦ closure (backwardTail γ j)) n).trans ?_
  apply closure_minimal _ hK
  rintro x ⟨k, hk, rfl⟩
  exact hn k hk

omit [T2Space X] in
/-- A closed set containing a forward tail contains the omega-limit set. -/
theorem omegaSet_singleton_subset_of_eventually (hγ : IsSolution f γ) (hK : IsClosed K)
    (hevent : ∃ n : ℕ, ∀ k : ℕ, n ≤ k → γ (k : ℤ) ∈ K) :
    omegaSet f {γ 0} ⊆ K := by
  obtain ⟨n, hn⟩ := hevent
  rw [omegaSet_singleton_eq_forwardTail]
  refine (iInter_subset (fun j : ℕ ↦ closure (forwardTail f (γ 0) j)) n).trans ?_
  apply closure_minimal _ hK
  rintro x ⟨k, hk, rfl⟩
  rw [hγ.iterate k]
  exact hn k hk

/-- Backward eventual containment also places the alpha-limit in the maximal invariant set. -/
theorem alphaSet_subset_Inv_of_eventually (hf : Continuous f) (hγ : IsSolution f γ)
    (hN : IsCompact N) (hγN : ∀ k : ℤ, γ k ∈ N) (hK : IsClosed K)
    (hevent : ∃ n : ℕ, ∀ k : ℤ, k ≤ -(n : ℤ) → γ k ∈ K) : alphaSet γ ⊆ Inv f K := by
  rw [← Inv_eq_of_image_eq (image_alphaSet hf hγ hN hγN)]
  exact Inv_mono (alphaSet_subset_of_eventually hK hevent)

/-- Forward eventual containment also places the omega-limit in the maximal invariant set. -/
theorem omegaSet_singleton_subset_Inv_of_eventually (hf : Continuous f)
    (hγ : IsSolution f γ) (hN : IsCompact N) (hγN : ∀ k : ℤ, γ k ∈ N)
    (hK : IsClosed K) (hevent : ∃ n : ℕ, ∀ k : ℕ, n ≤ k → γ (k : ℤ) ∈ K) :
    omegaSet f {γ 0} ⊆ Inv f K := by
  have hforward : ∀ n : ℕ, f^[n] (γ 0) ∈ N := fun n ↦ by
    rw [hγ.iterate n]
    exact hγN n
  rw [← Inv_eq_of_image_eq (image_omegaSet_singleton hf hN hforward)]
  exact Inv_mono (omegaSet_singleton_subset_of_eventually hγ hK hevent)

end LatentEquivariant
