import Mathlib

/-!
# Isolated invariant sets and index pairs for maps

CMGDB computes the Conley index of a continuous map, so everything in this file is stated for a
map `f : X → X`, not for a flow. Isolating neighborhoods and index pairs follow Szymczak, *The
Conley index for discrete semidynamical systems*, Topology Appl. 66 (1995), Definitions 3.2 and
3.4, which the paper uses.

The main results are:

* `exists_solution_lift`: a surjective semiconjugacy `p ∘ f = g ∘ p` with the one-step backward
  lifting property lifts every solution of `g` to a solution of `f`. `exists_solution_lift_of_eq`
  lifts through a prescribed point and needs no surjectivity.
* `IsIndexPair.preimage`: index pairs pull back along a continuous semiconjugacy with compact
  preimages (Szymczak 1995, Lemma 4.4), as in the proof of Proposition `prop:rotational_reduction`.
  `IsIndexPair.Inv_preimage_of_lift` identifies the isolated invariant set of the pulled-back
  pair when `h` lifts one step back through prescribed points.
* `isAttractingBlock_preimage`: Theorem `thm:main_simple` of the latent-dynamics paper. The
  weakened forms `image_preimage_subset_interior`, `isAttractingBlock_preimage_of_isCompact` and
  `isAttractingBlock_preimage_of_isClosed` need no hypothesis on `g`, only a pseudometric on `Z`,
  and compactness of `E⁻¹(N)` only. `tolerance_pos_of_isAttractingBlock` and
  `image_subset_interior_of_tolerance_pos` relate attracting blocks and positive tolerance.
-/

namespace LatentEquivariant

open Set Topology

variable {X Y : Type*}

/-- A solution of `f`: a two-sided sequence `γ` with `f (γ n) = γ (n + 1)` for every `n`. -/
def IsSolution (f : X → X) (γ : ℤ → X) : Prop :=
  ∀ n, f (γ n) = γ (n + 1)

/-- The maximal invariant subset `Inv(N, f)`: the points of `N` that lie on a solution of `f`
contained in `N` (Szymczak 1995, Section 3). -/
def Inv (f : X → X) (N : Set X) : Set X :=
  {x | ∃ γ : ℤ → X, IsSolution f γ ∧ γ 0 = x ∧ ∀ n, γ n ∈ N}

section Inv

variable {f : X → X} {g : Y → Y} {N M : Set X}

theorem IsSolution.iterate {γ : ℤ → X} (hγ : IsSolution f γ) (n : ℕ) : f^[n] (γ 0) = γ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih, hγ n, Nat.cast_succ]

theorem Inv_subset : Inv f N ⊆ N := by
  rintro x ⟨γ, -, rfl, hγN⟩
  exact hγN 0

theorem Inv_mono (h : N ⊆ M) : Inv f N ⊆ Inv f M := by
  rintro x ⟨γ, hγ, rfl, hγN⟩
  exact ⟨γ, hγ, rfl, fun n ↦ h (hγN n)⟩

/-- The image of a solution under a semiconjugacy is a solution. -/
theorem IsSolution.comp {p : X → Y} (hp : ∀ x, p (f x) = g (p x)) {γ : ℤ → X}
    (hγ : IsSolution f γ) : IsSolution g (p ∘ γ) := by
  intro n
  simp only [Function.comp_apply, ← hp, hγ n]

/-- Szymczak (1995), Proposition 3.4: `Inv(p⁻¹(A), f) ⊆ p⁻¹(Inv(A, g))` for a semiconjugacy
`p ∘ f = g ∘ p`. -/
theorem Inv_preimage_subset {p : X → Y} (hp : ∀ x, p (f x) = g (p x)) (A : Set Y) :
    Inv f (p ⁻¹' A) ⊆ p ⁻¹' Inv g A := by
  rintro x ⟨γ, hγ, rfl, hγA⟩
  exact ⟨p ∘ γ, hγ.comp hp, rfl, hγA⟩

theorem image_Inv_subset {p : X → Y} (hp : ∀ x, p (f x) = g (p x)) (K : Set X) :
    p '' Inv f K ⊆ Inv g (p '' K) := by
  rintro _ ⟨x, ⟨γ, hγ, rfl, hγK⟩, rfl⟩
  exact ⟨p ∘ γ, hγ.comp hp, rfl, fun n ↦ mem_image_of_mem p (hγK n)⟩

/-- A solution glued from a forward orbit and a backward sequence. -/
private def glue (f : X → X) (x : X) (β : ℕ → X) : ℤ → X
  | .ofNat n => f^[n] x
  | .negSucc n => β (n + 1)

private theorem glue_isSolution (x : X) (β : ℕ → X) (h0 : β 0 = x)
    (hβ : ∀ n, f (β (n + 1)) = β n) : IsSolution f (glue f x β) := by
  intro n
  rcases n with n | n
  · show f (f^[n] x) = f^[n + 1] x
    rw [Function.iterate_succ_apply']
  · rcases n with _ | m
    · show f (β (0 + 1)) = f^[0] x
      rw [hβ 0, h0]
      rfl
    · have : Int.negSucc (m + 1) + 1 = Int.negSucc m := by
        simp only [Int.negSucc_eq]
        push_cast
        ring
      rw [this]
      exact hβ (m + 1)

/-- Lifting of solutions through a prescribed point, along a semiconjugacy `p ∘ f = g ∘ p` with the
one-step backward lifting property: if `g q = p y`, then `q = p y'` for some `y'` with
`f y' = y`. If `p x = γ 0`, the solution `γ` of `g` lifts to a solution `δ` of `f` with
`δ 0 = x`. -/
theorem exists_solution_lift_of_eq {p : X → Y} (hp : ∀ x, p (f x) = g (p x))
    (hback : ∀ (y : X) (q : Y), g q = p y → ∃ y', f y' = y ∧ p y' = q)
    {γ : ℤ → Y} (hγ : IsSolution g γ) {x : X} (hx : p x = γ 0) :
    ∃ δ : ℤ → X, IsSolution f δ ∧ δ 0 = x ∧ ∀ n, p (δ n) = γ n := by
  -- one step back from a lift of `γ (-n)` to a lift of `γ (-(n + 1))`
  have step : ∀ (n : ℕ) (y : X), p y = γ (-(n : ℤ)) →
      ∃ y', f y' = y ∧ p y' = γ (-((n + 1 : ℕ) : ℤ)) := by
    intro n y hy
    refine hback y _ ?_
    rw [hγ (-((n + 1 : ℕ) : ℤ)), hy]
    congr 1
    push_cast
    ring
  -- the backward sequence, built by recursion
  let β : (n : ℕ) → {y : X // p y = γ (-(n : ℤ))} := fun n ↦
    Nat.rec (motive := fun n ↦ {y : X // p y = γ (-(n : ℤ))}) ⟨x, by simpa using hx⟩
      (fun n y ↦ ⟨Classical.choose (step n y.1 y.2), (Classical.choose_spec (step n y.1 y.2)).2⟩)
      n
  have hβ : ∀ n, f (β (n + 1)).1 = (β n).1 := fun n ↦
    (Classical.choose_spec (step n (β n).1 (β n).2)).1
  refine ⟨glue f x (fun n ↦ (β n).1), glue_isSolution x _ rfl hβ, rfl, ?_⟩
  intro n
  rcases n with n | n
  · show p (f^[n] x) = γ n
    exact ((Function.Semiconj.iterate_right (fun x ↦ hp x) n) x).trans (by rw [hx, hγ.iterate])
  · show p (β (n + 1)).1 = γ (Int.negSucc n)
    rw [(β (n + 1)).2]
    congr 1

/-- Lifting of solutions along a semiconjugacy `p ∘ f = g ∘ p` that is surjective and has the
one-step backward lifting property: if `g q = p y`, then `q = p y'` for some `y'` with
`f y' = y`. Every solution of `g` lifts to a solution of `f`. -/
theorem exists_solution_lift {p : X → Y} (hp : ∀ x, p (f x) = g (p x))
    (hsurj : Function.Surjective p)
    (hback : ∀ (y : X) (q : Y), g q = p y → ∃ y', f y' = y ∧ p y' = q)
    {γ : ℤ → Y} (hγ : IsSolution g γ) :
    ∃ δ : ℤ → X, IsSolution f δ ∧ ∀ n, p (δ n) = γ n := by
  obtain ⟨x, hx⟩ := hsurj (γ 0)
  obtain ⟨δ, hδ, -, hpδ⟩ := exists_solution_lift_of_eq hp hback hγ hx
  exact ⟨δ, hδ, hpδ⟩

/-- Every point of a solution contained in `N` lies in `Inv(N, f)`. -/
theorem IsSolution.mem_Inv {γ : ℤ → X} (hγ : IsSolution f γ) (hN : ∀ n, γ n ∈ N) (m : ℤ) :
    γ m ∈ Inv f N :=
  ⟨fun n ↦ γ (n + m), fun n ↦ by simp only [hγ (n + m), add_right_comm], by simp, fun n ↦ hN _⟩

end Inv

section IndexPair

variable [TopologicalSpace X] [TopologicalSpace Y]

/-- Szymczak (1995), Definition 3.2: a compact set `M` is an isolating neighborhood if
`Inv(M, f) ⊆ int M`. -/
def IsIsolatingNeighborhood (f : X → X) (M : Set X) : Prop :=
  IsCompact M ∧ Inv f M ⊆ interior M

/-- Szymczak (1995), Definition 3.4: a pair `(N, L)` of compact sets is an index pair (for
`S = Inv(cl(N \ L), f)`) if

1. `Inv(cl(N \ L), f) ⊆ int(N \ L)`,
2. `L` is positively invariant in `N`: if `x ∈ L`, then `f x ∉ N \ L`,
3. `L` is an exit set for `N`: if `x ∈ N` and `f x ∉ N`, then `x ∈ L`.

For a map the exit set cannot be a thin piece of `∂N`: by (3) it contains every point of `N`
that leaves `N` in one step. -/
structure IsIndexPair (f : X → X) (N L : Set X) : Prop where
  isCompact_left : IsCompact N
  isCompact_right : IsCompact L
  isolating : Inv f (closure (N \ L)) ⊆ interior (N \ L)
  posInv : ∀ x ∈ L, f x ∉ N \ L
  exit : ∀ x ∈ N, f x ∉ N → x ∈ L

variable {f : X → X} {g : Y → Y} {h : X → Y}

/-- Index pairs pull back along a continuous semiconjugacy `h ∘ f = g ∘ h` such that `h⁻¹(N)` and
`h⁻¹(L)` are compact. -/
theorem IsIndexPair.preimage_of_isCompact (hh : Continuous h) (hconj : ∀ x, h (f x) = g (h x))
    {N L : Set Y} (hNL : IsIndexPair g N L) (hN : IsCompact (h ⁻¹' N))
    (hL : IsCompact (h ⁻¹' L)) : IsIndexPair f (h ⁻¹' N) (h ⁻¹' L) where
  isCompact_left := hN
  isCompact_right := hL
  isolating := by
    rw [← preimage_sdiff]
    intro x hx
    have h1 : Inv f (closure (h ⁻¹' (N \ L))) ⊆ Inv f (h ⁻¹' closure (N \ L)) :=
      Inv_mono (hh.closure_preimage_subset _)
    have h2 := Inv_preimage_subset hconj (closure (N \ L)) (h1 hx)
    exact preimage_interior_subset_interior_preimage hh (hNL.isolating h2)
  posInv := by
    intro x hx hfx
    rw [← preimage_sdiff, mem_preimage, hconj] at hfx
    exact hNL.posInv (h x) hx hfx
  exit := by
    intro x hx hfx
    rw [mem_preimage, hconj] at hfx
    exact hNL.exit (h x) hx hfx

/-- Index pairs pull back along a continuous semiconjugacy `h ∘ f = g ∘ h` whose preimages of
compact sets are compact. This is the index-pair part of Szymczak (1995), Lemma 4.4, which
assumes `h` continuous and proper. Szymczak's
condition that `h⁻¹` of the isolated invariant set downstairs be the one upstairs, and metric
spaces, are not needed for the pair conditions; they enter only when the isolated invariant set
is identified (`IsIndexPair.Inv_preimage`, `IsIndexPair.Inv_preimage_of_lift`). -/
theorem IsIndexPair.preimage (hh : Continuous h)
    (hcpt : ∀ K : Set Y, IsCompact K → IsCompact (h ⁻¹' K)) (hconj : ∀ x, h (f x) = g (h x))
    {N L : Set Y} (hNL : IsIndexPair g N L) : IsIndexPair f (h ⁻¹' N) (h ⁻¹' L) :=
  hNL.preimage_of_isCompact hh hconj (hcpt N hNL.isCompact_left) (hcpt L hNL.isCompact_right)

/-- The isolated invariant set of the pulled-back pair is the preimage of the one downstairs,
provided that preimage is invariant (the hypothesis of Szymczak 1995, Proposition 3.6). -/
theorem IsIndexPair.Inv_preimage (hh : Continuous h) (hconj : ∀ x, h (f x) = g (h x))
    {N L : Set Y} (hNL : IsIndexPair g N L)
    (hS : h ⁻¹' Inv g (closure (N \ L)) ⊆ Inv f (h ⁻¹' Inv g (closure (N \ L)))) :
    Inv f (closure (h ⁻¹' N \ h ⁻¹' L)) = h ⁻¹' Inv g (closure (N \ L)) := by
  apply Subset.antisymm
  · rw [← preimage_sdiff]
    exact (Inv_mono (hh.closure_preimage_subset _)).trans (Inv_preimage_subset hconj _)
  · refine hS.trans (Inv_mono ?_)
    rw [← preimage_sdiff]
    intro x hx
    have hx' : h x ∈ N \ L := interior_subset (hNL.isolating hx)
    exact subset_closure hx'

/-- The isolated invariant set of the pulled-back pair is `h⁻¹` of the one downstairs if `h` lifts
one step back through prescribed points: whenever `g w = h y`, some `y'` with `h y' = w` has
`f y' = y`. This holds when `f(h⁻¹(w)) = h⁻¹(g(w))` for every `w`, as for the rotational
reduction of Proposition `prop:rotational_reduction`. It derives the hypothesis of
`IsIndexPair.Inv_preimage`. -/
theorem IsIndexPair.Inv_preimage_of_lift (hh : Continuous h) (hconj : ∀ x, h (f x) = g (h x))
    (hback : ∀ (y : X) (w : Y), g w = h y → ∃ y', f y' = y ∧ h y' = w)
    {N L : Set Y} (hNL : IsIndexPair g N L) :
    Inv f (closure (h ⁻¹' N \ h ⁻¹' L)) = h ⁻¹' Inv g (closure (N \ L)) := by
  refine hNL.Inv_preimage hh hconj fun x hx ↦ ?_
  obtain ⟨γ, hγ, hγ0, hγK⟩ := hx
  obtain ⟨δ, hδ, hδ0, hpδ⟩ := exists_solution_lift_of_eq hconj hback hγ hγ0.symm
  exact ⟨δ, hδ, hδ0, fun n ↦ by rw [mem_preimage, hpδ n]; exact hγ.mem_Inv hγK n⟩

/-- An attracting block for `f` (the attracting blocks of the latent-dynamics paper):
a compact set `N` with `f(N) ⊆ int N`. -/
def IsAttractingBlock (f : X → X) (N : Set X) : Prop :=
  IsCompact N ∧ f '' N ⊆ interior N

/-- The omega-limit set `ω(N, f) = ⋂ₙ cl(⋃_{k ≥ n} fᵏ(N))` of the latent-dynamics paper. -/
def omegaSet (f : X → X) (N : Set X) : Set X :=
  ⋂ n : ℕ, closure (⋃ k ≥ n, f^[k] '' N)

end IndexPair

section AttractingBlock

open scoped ENNReal

variable {Z : Type*} [PseudoMetricSpace Z]

/-- The tolerance `τ_g(N) = sup {ν > 0 | B_ν(g(N)) ⊆ N}`, where `B_ν` is the open
`ν`-neighborhood (`Metric.thickening`). It takes values in `ℝ≥0∞`, so that it is `∞` when every
`ν` works, for instance when `N` is the whole space. -/
noncomputable def tolerance (g : Z → Z) (N : Set Z) : ℝ≥0∞ :=
  ⨆ (ν : ℝ) (_ : 0 < ν ∧ Metric.thickening ν (g '' N) ⊆ N), ENNReal.ofReal ν

/-- If `r < τ_g(N)`, some admissible radius exceeds `r`. -/
theorem exists_lt_of_lt_tolerance {g : Z → Z} {N : Set Z} {r : ℝ}
    (h : ENNReal.ofReal r < tolerance g N) :
    ∃ ν, r < ν ∧ Metric.thickening ν (g '' N) ⊆ interior N := by
  obtain ⟨ν, hν⟩ := lt_iSup_iff.mp h
  obtain ⟨⟨hν0, hνN⟩, hrν⟩ := lt_iSup_iff.mp hν
  refine ⟨ν, ?_, Metric.isOpen_thickening.subset_interior_iff.mpr hνN⟩
  exact ((ENNReal.ofReal_lt_ofReal_iff hν0).mp hrν)

end AttractingBlock

section Pullback

open scoped ENNReal

variable {Z : Type*} [PseudoMetricSpace Z] [TopologicalSpace X] {f : X → X} {g : Z → Z}
  {E : X → Z}

/-- The inclusion in Theorem `thm:main_simple` of the latent-dynamics paper, with no compactness
and no hypothesis on `N` or `g`: if `d(g(E x), E(f x)) < τ_g(N)` for every `x ∈ E⁻¹(N)`, then
`f(E⁻¹(N)) ⊆ int E⁻¹(N)`. -/
theorem image_preimage_subset_interior (hE : Continuous E) {N : Set Z}
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (dist (g (E x)) (E (f x))) < tolerance g N) :
    f '' (E ⁻¹' N) ⊆ interior (E ⁻¹' N) := by
  rintro _ ⟨x, hx, rfl⟩
  obtain ⟨ν, hν, hνN⟩ := exists_lt_of_lt_tolerance (hres x hx)
  have hfx : E (f x) ∈ interior N := by
    apply hνN
    rw [Metric.mem_thickening_iff]
    exact ⟨g (E x), mem_image_of_mem g hx, by rwa [dist_comm]⟩
  exact preimage_interior_subset_interior_preimage hE hfx

/-- Theorem `thm:main_simple` with compactness of `E⁻¹(N)` in place of compactness of `X`, for
instance `E` proper and `N` compact, and with no hypothesis on `N` or `g`. -/
theorem isAttractingBlock_preimage_of_isCompact (hE : Continuous E) {N : Set Z}
    (hK : IsCompact (E ⁻¹' N))
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (dist (g (E x)) (E (f x))) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) :=
  ⟨hK, image_preimage_subset_interior hE hres⟩

/-- Theorem `thm:main_simple` with `N` only closed: the hypothesis that `N` is an attracting block
for `g` is not needed, `Z` need only be pseudometric, and `f` and `g` need not be continuous. A
residual bound at a single point already makes the tolerance positive, so it forces
`g(N) ⊆ int N` (`image_subset_interior_of_tolerance_pos`). -/
theorem isAttractingBlock_preimage_of_isClosed [CompactSpace X] (hE : Continuous E) {N : Set Z}
    (hN : IsClosed N)
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (dist (g (E x)) (E (f x))) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) :=
  isAttractingBlock_preimage_of_isCompact hE (hN.preimage hE).isCompact hres

omit [TopologicalSpace X] in
/-- A positive tolerance forces `g(N) ⊆ int N`. -/
theorem image_subset_interior_of_tolerance_pos {N : Set Z} (h : 0 < tolerance g N) :
    g '' N ⊆ interior N := by
  obtain ⟨ν, hν, hνN⟩ := exists_lt_of_lt_tolerance (g := g) (N := N) (r := 0) (by simpa using h)
  exact (Metric.self_subset_thickening hν _).trans hνN

omit [TopologicalSpace X] in
/-- Conversely, an attracting block of a map continuous on it has positive tolerance. Without
continuity this fails (`Sharpness.tolerance_gapMap`). -/
theorem tolerance_pos_of_isAttractingBlock {N : Set Z} (hg : ContinuousOn g N)
    (hN : IsAttractingBlock g N) : 0 < tolerance g N := by
  obtain ⟨δ, hδ, hδN⟩ :=
    (hN.1.image_of_continuousOn hg).exists_thickening_subset_open isOpen_interior hN.2
  exact lt_of_lt_of_le (ENNReal.ofReal_pos.mpr hδ)
    (le_iSup₂_of_le (f := fun (ν : ℝ) (_ : 0 < ν ∧ Metric.thickening ν (g '' N) ⊆ N) ↦
      ENNReal.ofReal ν) δ ⟨hδ, hδN.trans interior_subset⟩ le_rfl)

/-- Under the hypotheses of the latent-dynamics paper (`f`, `g` and `E` continuous, `X` compact)
the pointwise residual bound gives the bound on the supremum, which is attained. So the pointwise
form of `thm:main_simple` gains only together with dropping continuity. -/
theorem iSup_residual_lt_of_forall [CompactSpace X] (hf : Continuous f) (hg : Continuous g)
    (hE : Continuous E) {N : Set Z} (hN : IsClosed N) (hne : (E ⁻¹' N).Nonempty)
    {τ : ℝ≥0∞} (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (dist (g (E x)) (E (f x))) < τ) :
    ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (dist (g (E x)) (E (f x))) < τ := by
  have hc : Continuous fun x ↦ dist (g (E x)) (E (f x)) := (hg.comp hE).dist (hE.comp hf)
  obtain ⟨x₀, hx₀, hmax⟩ := (hN.preimage hE).isCompact.exists_isMaxOn hne hc.continuousOn
  exact lt_of_le_of_lt
    (iSup₂_le fun x hx ↦ ENNReal.ofReal_le_ofReal (isMaxOn_iff.mp hmax x hx)) (hres x₀ hx₀)

end Pullback

section MainSimple

open scoped ENNReal

variable {Z : Type*} [MetricSpace Z] [TopologicalSpace X] [CompactSpace X] {f : X → X}
  {g : Z → Z} {E : X → Z}

/-- The latent-dynamics paper, Theorem `thm:main_simple`, with the hypothesis on the supremum
weakened to a pointwise bound: if `d(g(E x), E(f x)) < τ_g(N)` for every `x ∈ E⁻¹(N)`, then
`E⁻¹(N)` is an attracting block for `f`. Continuity of `f` and `g` is not needed. Under the
paper's continuity hypotheses the two bounds are equivalent (`iSup_residual_lt_of_forall`), and
without continuity of `f` the conclusion has no dynamical content
(`Sharpness.main_simple_discontinuous`). See `isAttractingBlock_preimage_of_isClosed` for the
general form. -/
theorem isAttractingBlock_preimage_of_forall (hE : Continuous E) {N : Set Z}
    (hN : IsAttractingBlock g N)
    (hres : ∀ x ∈ E ⁻¹' N, ENNReal.ofReal (dist (g (E x)) (E (f x))) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) :=
  isAttractingBlock_preimage_of_isClosed hE hN.1.isClosed hres

/-- The latent-dynamics paper, Theorem `thm:main_simple`: if
`sup_{x ∈ E⁻¹(N)} d(g(E x), E(f x)) < τ_g(N)`, then `E⁻¹(N)` is an attracting block for `f`. -/
theorem isAttractingBlock_preimage (hE : Continuous E) {N : Set Z} (hN : IsAttractingBlock g N)
    (hres : ⨆ x ∈ E ⁻¹' N, ENNReal.ofReal (dist (g (E x)) (E (f x))) < tolerance g N) :
    IsAttractingBlock f (E ⁻¹' N) :=
  isAttractingBlock_preimage_of_forall hE hN fun x hx ↦
    lt_of_le_of_lt (le_iSup₂ (f := fun x (_ : x ∈ E ⁻¹' N) ↦
      ENNReal.ofReal (dist (g (E x)) (E (f x)))) x hx) hres

end MainSimple

end LatentEquivariant

namespace LatentEquivariant

open Set Topology Pointwise MulAction

section BlockSublattice

variable {X Y : Type*} [TopologicalSpace X] {f : X → X} {N M : Set X}

theorem isAttractingBlock_empty : IsAttractingBlock f ∅ := by
  exact ⟨isCompact_empty, by simp⟩

theorem IsAttractingBlock.union (hN : IsAttractingBlock f N) (hM : IsAttractingBlock f M) :
    IsAttractingBlock f (N ∪ M) := by
  refine ⟨hN.1.union hM.1, ?_⟩
  rintro _ ⟨x, hx | hx, rfl⟩
  · exact interior_mono subset_union_left (hN.2 (mem_image_of_mem f hx))
  · exact interior_mono subset_union_right (hM.2 (mem_image_of_mem f hx))

theorem IsAttractingBlock.inter [T2Space X] (hN : IsAttractingBlock f N)
    (hM : IsAttractingBlock f M) : IsAttractingBlock f (N ∩ M) := by
  refine ⟨hN.1.inter hM.1, ?_⟩
  rw [interior_inter]
  rintro _ ⟨x, hx, rfl⟩
  exact ⟨hN.2 (mem_image_of_mem f hx.1), hM.2 (mem_image_of_mem f hx.2)⟩

/-- The sublattice of attracting blocks, with union and intersection as its operations. -/
def attractingBlockSublattice [T2Space X] (f : X → X) : Sublattice (Set X) where
  carrier := {N | IsAttractingBlock f N}
  supClosed' := fun _ hN _ hM ↦ hN.union hM
  infClosed' := fun _ hN _ hM ↦ hN.inter hM

@[simp] theorem mem_attractingBlockSublattice [T2Space X] :
    N ∈ attractingBlockSublattice f ↔ IsAttractingBlock f N := Iff.rfl

end BlockSublattice

section GeneratedSublattice

variable {α : Type*} [Lattice α]

/-- The sublattice generated by a family of elements. -/
def generatedSublattice (S : Set α) : Sublattice α :=
  sInf {L | S ⊆ (L : Set α)}

theorem subset_generatedSublattice (S : Set α) : S ⊆ generatedSublattice S := by
  intro x hx
  change x ∈ (sInf {L : Sublattice α | S ⊆ (L : Set α)} : Sublattice α)
  simp only [← SetLike.mem_coe, Sublattice.coe_sInf, mem_iInter, mem_setOf_eq]
  exact fun L hL ↦ hL hx

theorem generatedSublattice_le {S : Set α} {L : Sublattice α} (h : S ⊆ L) :
    generatedSublattice S ≤ L :=
  sInf_le h

end GeneratedSublattice

section PreimageSublattice

variable {X Y : Type*}

/-- The image of a sublattice under inverse image. -/
def preimageSublattice (E : X → Y) (L : Sublattice (Set Y)) : Sublattice (Set X) :=
  L.map (CompleteLatticeHom.setPreimage E : LatticeHom (Set Y) (Set X))

@[simp] theorem mem_preimageSublattice (E : X → Y) (L : Sublattice (Set Y)) (M : Set X) :
    M ∈ preimageSublattice E L ↔ ∃ N ∈ L, E ⁻¹' N = M := Iff.rfl

theorem finite_preimageSublattice (E : X → Y) (L : Sublattice (Set Y))
    (hL : (L : Set (Set Y)).Finite) : (preimageSublattice E L : Set (Set X)).Finite :=
  hL.image _

theorem preimage_eq_iff_inter_range (E : X → Y) (N M : Set Y) :
    E ⁻¹' N = E ⁻¹' M ↔ N ∩ range E = M ∩ range E := by
  constructor
  · intro h
    simpa only [image_preimage_eq_inter_range] using congrArg (image E) h
  · intro h
    simpa only [preimage_inter_range] using congrArg (preimage E) h

theorem injOn_preimage_of_inter_range (E : X → Y) (S : Set (Set Y))
    (hS : InjOn (fun N ↦ N ∩ range E) S) : InjOn (preimage E) S := by
  intro N hN M hM h
  exact hS hN hM ((preimage_eq_iff_inter_range E N M).mp h)

/-- Inverse image as a homomorphism onto the preimage sublattice. -/
def preimageSublatticeHom (E : X → Y) (L : Sublattice (Set Y)) :
    LatticeHom L (preimageSublattice E L) where
  toFun N := ⟨E ⁻¹' (N : Set Y), N, N.property, rfl⟩
  map_sup' _ _ := Subtype.ext rfl
  map_inf' _ _ := Subtype.ext rfl

theorem preimageSublatticeHom_surjective (E : X → Y) (L : Sublattice (Set Y)) :
    Function.Surjective (preimageSublatticeHom E L) := by
  rintro ⟨M, N, hN, rfl⟩
  exact ⟨⟨N, hN⟩, rfl⟩

theorem preimageSublatticeHom_injective (E : X → Y) (L : Sublattice (Set Y))
    (hL : InjOn (fun N ↦ N ∩ range E) L) :
    Function.Injective (preimageSublatticeHom E L) := by
  intro N M h
  apply Subtype.ext
  exact injOn_preimage_of_inter_range E L hL N.property M.property (congrArg Subtype.val h)

/-- Distinct intersections with the encoder image yield a lattice isomorphism onto the pullback. -/
noncomputable def preimageSublatticeOrderIso (E : X → Y) (L : Sublattice (Set Y))
    (hL : InjOn (fun N ↦ N ∩ range E) L) : L ≃o preimageSublattice E L where
  toEquiv := Equiv.ofBijective (preimageSublatticeHom E L)
    ⟨preimageSublatticeHom_injective E L hL, preimageSublatticeHom_surjective E L⟩
  map_rel_iff' :=
    (orderEmbeddingOfInjective (preimageSublatticeHom E L)
      (preimageSublatticeHom_injective E L hL)).map_rel_iff

theorem preimage_nonempty_of_injOn (E : X → Y) (S : Set (Set Y))
    (hS : InjOn (preimage E) S) (hempty : ∅ ∈ S) {N : Set Y} (hN : N ∈ S) (hne : N ≠ ∅) :
    (E ⁻¹' N).Nonempty := by
  by_contra h
  apply hne
  apply hS hN hempty
  simpa only [preimage_empty] using not_nonempty_iff_eq_empty.mp h

end PreimageSublattice

section BlockGenerators

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}

theorem generatedSublattice_attractingBlocks {S : Set (Set X)}
    (hS : ∀ N ∈ S, IsAttractingBlock f N) :
    generatedSublattice S ≤ attractingBlockSublattice f :=
  generatedSublattice_le hS

theorem generatedSublattice_insert_empty_attractingBlocks {S : Set (Set X)}
    (hS : ∀ N ∈ S, IsAttractingBlock f N) :
    generatedSublattice (insert ∅ S) ≤ attractingBlockSublattice f := by
  apply generatedSublattice_attractingBlocks
  intro N hN
  rcases hN with rfl | hN
  · exact isAttractingBlock_empty
  · exact hS N hN

end BlockGenerators

end LatentEquivariant

namespace LatentEquivariant

open Set Topology

section InvariantSets

variable {X : Type*} {f : X → X} {N : Set X}

theorem image_Inv_eq (f : X → X) (N : Set X) : f '' Inv f N = Inv f N := by
  apply Subset.antisymm
  · rintro _ ⟨x, ⟨γ, hγ, hγ0, hγN⟩, rfl⟩
    rw [← hγ0, hγ 0]
    exact hγ.mem_Inv hγN 1
  · rintro x ⟨γ, hγ, hγ0, hγN⟩
    refine ⟨γ (-1), hγ.mem_Inv hγN (-1), ?_⟩
    simpa only [hγ (-1), Int.reduceNeg, Int.reduceAdd] using hγ0

/-- A set mapped onto itself consists entirely of points on full solutions. -/
theorem Inv_eq_of_image_eq (hN : f '' N = N) : Inv f N = N := by
  apply Subset.antisymm Inv_subset
  intro x hx
  let fN : N → N := fun y ↦ ⟨f y, hN.le (mem_image_of_mem f y.property)⟩
  have hsurj : Function.Surjective fN := by
    intro y
    have hy : y.val ∈ f '' N := by rw [hN]; exact y.property
    obtain ⟨z, hz, hzy⟩ := hy
    exact ⟨⟨z, hz⟩, Subtype.ext hzy⟩
  obtain ⟨δ, hδ, hδ0, _⟩ := exists_solution_lift_of_eq
    (f := fN) (g := id) (p := fun _ ↦ ()) (fun _ ↦ rfl)
    (fun y _ _ ↦ by obtain ⟨z, hz⟩ := hsurj y; exact ⟨z, hz, rfl⟩)
    (γ := fun _ ↦ ()) (fun _ ↦ rfl) (x := ⟨x, hx⟩) rfl
  exact ⟨fun n ↦ (δ n).val, fun n ↦ congrArg Subtype.val (hδ n),
    congrArg Subtype.val hδ0, fun n ↦ (δ n).property⟩

end InvariantSets

section CompactOmega

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}
  {N M : Set X}

omit [TopologicalSpace X] [T2Space X] in
theorem iterate_image_antitone (hN : f '' N ⊆ N) : Antitone (fun n : ℕ ↦ f^[n] '' N) := by
  rw [show (fun n : ℕ ↦ f^[n] '' N) = (fun n ↦ (image f)^[n] N) by
    funext n; rw [image_iterate_eq]]
  exact antitone_nat_of_succ_le fun n ↦ by
    rw [Function.iterate_succ_apply]
    exact (show Monotone (image f) from fun _ _ h ↦ image_mono h).iterate n hN

theorem omegaSet_eq_iInter_iterate (hf : Continuous f) (hN : IsCompact N)
    (hforward : f '' N ⊆ N) : omegaSet f N = ⋂ n : ℕ, f^[n] '' N := by
  unfold omegaSet
  congr 1
  funext n
  have htail : (⋃ k ≥ n, f^[k] '' N) = f^[n] '' N := by
    apply Subset.antisymm
    · exact iUnion₂_subset fun k hk ↦ iterate_image_antitone hforward hk
    · exact subset_iUnion₂_of_subset n le_rfl subset_rfl
  rw [htail, (hN.image (hf.iterate n)).isClosed.closure_eq]

theorem omegaSet_subset (hf : Continuous f) (hN : IsCompact N)
    (hforward : f '' N ⊆ N) : omegaSet f N ⊆ N := by
  rw [omegaSet_eq_iInter_iterate hf hN hforward]
  simpa only [Function.iterate_zero, id_eq, image_id'] using
    (iInter_subset (fun n : ℕ ↦ f^[n] '' N) 0)

theorem isCompact_omegaSet (hf : Continuous f) (hN : IsCompact N)
    (hforward : f '' N ⊆ N) : IsCompact (omegaSet f N) := by
  exact hN.of_isClosed_subset (isClosed_iInter fun _ ↦ isClosed_closure)
    (omegaSet_subset hf hN hforward)

/-- A continuous map commutes with intersections of a decreasing compact sequence. -/
theorem image_iInter_of_compact_antitone {Y : Type*} [TopologicalSpace Y] [T2Space Y]
    {p : X → Y} (hp : Continuous p) (s : ℕ → Set X) (hs : Antitone s)
    (hc : ∀ n, IsCompact (s n)) : p '' (⋂ n, s n) = ⋂ n, p '' s n := by
  apply Subset.antisymm (image_iInter_subset _ _)
  intro y hy
  let t := fun n ↦ s n ∩ p ⁻¹' {y}
  have htclosed : ∀ n, IsClosed (t n) := fun n ↦
    (hc n).isClosed.inter (isClosed_singleton.preimage hp)
  have htne : ∀ n, (t n).Nonempty := by
    intro n
    obtain ⟨x, hx, hxy⟩ := mem_iInter.mp hy n
    exact ⟨x, hx, hxy⟩
  obtain ⟨x, hx⟩ := IsCompact.nonempty_iInter_of_sequence_nonempty_isCompact_isClosed t
    (fun n ↦ inter_subset_inter_left _ (hs (Nat.le_succ n))) htne
    ((hc 0).inter_right (isClosed_singleton.preimage hp)) htclosed
  exact ⟨x, mem_iInter.mpr fun n ↦ (mem_iInter.mp hx n).1, (mem_iInter.mp hx 0).2⟩

theorem image_omegaSet (hf : Continuous f) (hN : IsCompact N)
    (hforward : f '' N ⊆ N) : f '' omegaSet f N = omegaSet f N := by
  rw [omegaSet_eq_iInter_iterate hf hN hforward,
    image_iInter_of_compact_antitone hf _ (iterate_image_antitone hforward)
      (fun n ↦ hN.image (hf.iterate n))]
  apply Subset.antisymm
  · apply subset_iInter
    intro n
    refine (iInter_subset (fun k : ℕ ↦ f '' (f^[k] '' N)) n).trans ?_
    rw [← image_comp, ← Function.iterate_succ']
    exact iterate_image_antitone hforward (Nat.le_succ n)
  · apply subset_iInter
    intro n
    rw [← image_comp, ← Function.iterate_succ']
    exact iInter_subset (fun k : ℕ ↦ f^[k] '' N) (n + 1)

theorem image_eq_subset_omegaSet (hf : Continuous f) (hN : IsCompact N)
    (hforward : f '' N ⊆ N) {A : Set X} (hA : f '' A = A) (hAN : A ⊆ N) :
    A ⊆ omegaSet f N := by
  rw [omegaSet_eq_iInter_iterate hf hN hforward]
  apply subset_iInter
  intro n
  have hn : f^[n] '' A = A := by
    rw [image_iterate_eq]
    exact Function.IsFixedPt.iterate hA n
  rw [← hn]
  exact image_mono hAN

theorem Inv_eq_omegaSet (hf : Continuous f) (hN : IsCompact N)
    (hforward : f '' N ⊆ N) : Inv f N = omegaSet f N := by
  apply Subset.antisymm
  · exact image_eq_subset_omegaSet hf hN hforward (image_Inv_eq f N) Inv_subset
  · rw [← Inv_eq_of_image_eq (image_omegaSet hf hN hforward)]
    exact Inv_mono (omegaSet_subset hf hN hforward)

omit [T2Space X] in
theorem omegaSet_mono {A B : Set X} (h : A ⊆ B) : omegaSet f A ⊆ omegaSet f B := by
  unfold omegaSet
  exact iInter_mono fun _ ↦ closure_mono (iUnion_mono fun _ ↦ iUnion_mono fun _ ↦ image_mono h)

theorem omegaSet_union (hf : Continuous f) (hN : IsCompact N) (hM : IsCompact M)
    (hforwardN : f '' N ⊆ N) (hforwardM : f '' M ⊆ M) :
    omegaSet f (N ∪ M) = omegaSet f N ∪ omegaSet f M := by
  have hforward : f '' (N ∪ M) ⊆ N ∪ M := by
    rw [image_union]
    exact union_subset_union hforwardN hforwardM
  rw [omegaSet_eq_iInter_iterate hf (hN.union hM) hforward,
    omegaSet_eq_iInter_iterate hf hN hforwardN,
    omegaSet_eq_iInter_iterate hf hM hforwardM]
  simp only [image_union]
  exact iInter_union_of_antitone (iterate_image_antitone hforwardN)
    (iterate_image_antitone hforwardM)

theorem omegaSet_inter_omegaSet (hf : Continuous f) (hN : IsCompact N) (hM : IsCompact M)
    (hforwardN : f '' N ⊆ N) (hforwardM : f '' M ⊆ M) :
    omegaSet f (omegaSet f N ∩ omegaSet f M) = omegaSet f (N ∩ M) := by
  have hforward : f '' (N ∩ M) ⊆ N ∩ M :=
    (image_inter_subset _ _ _).trans (inter_subset_inter hforwardN hforwardM)
  have hleft : omegaSet f (N ∩ M) ⊆ omegaSet f N ∩ omegaSet f M :=
    subset_inter (omegaSet_mono inter_subset_left) (omegaSet_mono inter_subset_right)
  apply Subset.antisymm
  · exact omegaSet_mono (inter_subset_inter (omegaSet_subset hf hN hforwardN)
      (omegaSet_subset hf hM hforwardM))
  · exact image_eq_subset_omegaSet hf
      ((isCompact_omegaSet hf hN hforwardN).inter (isCompact_omegaSet hf hM hforwardM))
      ((image_inter_subset _ _ _).trans (by
        rw [image_omegaSet hf hN hforwardN, image_omegaSet hf hM hforwardM]))
      (image_omegaSet hf (hN.inter hM) hforward) hleft

end CompactOmega

end LatentEquivariant

namespace LatentEquivariant

open Set Topology

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X} {N M : Set X}

/-- Two attracting blocks satisfy the existing index-pair conditions. Applications to a nested
pair impose `M ⊆ N` separately, as required by the geometric pair convention. -/
theorem IsAttractingBlock.isIndexPair (hN : IsAttractingBlock f N)
    (hM : IsAttractingBlock f M) : IsIndexPair f N M where
  isCompact_left := hN.1
  isCompact_right := hM.1
  isolating := by
    rintro x ⟨γ, hγ, hγ0, hγK⟩
    have hKN : closure (N \ M) ⊆ N := closure_minimal sdiff_subset hN.1.isClosed
    have hxN : x ∈ interior N := by
      have h := hN.2 (mem_image_of_mem f (hKN (hγK (-1))))
      simpa only [hγ (-1), Int.reduceNeg, Int.reduceAdd, hγ0] using h
    have hxM : x ∉ M := by
      intro hx
      have hfxM := hM.2 (mem_image_of_mem f hx)
      have hfxK : f x ∈ closure (N \ M) := by simpa only [← hγ0, hγ 0, zero_add] using hγK 1
      have hc : closure (N \ M) ⊆ (interior M)ᶜ := by
        rw [← closure_compl]
        exact closure_mono (sdiff_subset_compl N M)
      exact hc hfxK hfxM
    rw [Set.sdiff_eq, interior_inter, hM.1.isClosed.isOpen_compl.interior_eq]
    exact ⟨hxN, hxM⟩
  posInv := fun x hx hfx ↦ hfx.2 (interior_subset (hM.2 (mem_image_of_mem f hx)))
  exit := fun x hx hfx ↦ (hfx (interior_subset (hN.2 (mem_image_of_mem f hx)))).elim

/-- The boundary added by closing a block difference contributes no invariant points. -/
theorem IsAttractingBlock.Inv_closure_sdiff (hN : IsAttractingBlock f N)
    (hM : IsAttractingBlock f M) : Inv f (closure (N \ M)) = Inv f (N \ M) := by
  apply Subset.antisymm
  · rintro x ⟨γ, hγ, hγ0, hγK⟩
    exact ⟨γ, hγ, hγ0, fun n ↦ interior_subset
      ((hN.isIndexPair hM).isolating (hγ.mem_Inv hγK n))⟩
  · exact Inv_mono subset_closure

end LatentEquivariant

namespace LatentEquivariant

open Set

section FiniteGeneration

variable {α : Type*} [Lattice α]

theorem coe_generatedSublattice (S : Set α) :
    (generatedSublattice S : Set α) = latticeClosure S := by
  apply Subset.antisymm
  · exact generatedSublattice_le (L :=
      ⟨latticeClosure S, isSublattice_latticeClosure.1, isSublattice_latticeClosure.2⟩)
      subset_latticeClosure
  · exact latticeClosure_min (subset_generatedSublattice S)
      ⟨(generatedSublattice S).supClosed, (generatedSublattice S).infClosed⟩

end FiniteGeneration

section FiniteDistributiveGeneration

variable {α : Type*} [DistribLattice α]

theorem finite_generatedSublattice {S : Set α} (hS : S.Finite) :
    (generatedSublattice S : Set α).Finite := by
  rw [coe_generatedSublattice]
  exact hS.latticeClosure

end FiniteDistributiveGeneration

end LatentEquivariant

namespace LatentEquivariant

open Set Topology

section BlockDual

variable {X : Type*} {f : X → X} {S M : Set X}

/-- The part of an invariant set whose forward orbits avoid a trapping block. -/
def blockDual (f : X → X) (S M : Set X) : Set X :=
  {x | x ∈ S ∧ ∀ n : ℕ, f^[n] x ∉ M}

theorem image_blockDual (hS : f '' S = S) (hM : f '' M ⊆ M) :
    f '' blockDual f S M = blockDual f S M := by
  apply Subset.antisymm
  · rintro _ ⟨x, hx, rfl⟩
    refine ⟨hS.le (mem_image_of_mem f hx.1), fun n ↦ ?_⟩
    simpa only [← Function.iterate_succ_apply] using hx.2 (n + 1)
  · intro x hx
    obtain ⟨y, hy, hfy⟩ := hS.ge hx.1
    refine ⟨y, ⟨hy, fun n ↦ ?_⟩, hfy⟩
    cases n with
    | zero =>
      intro hyM
      have hxM := hM (mem_image_of_mem f hyM)
      exact hx.2 0 (by simpa only [Function.iterate_zero, id_eq, hfy] using hxM)
    | succ n => simpa only [Function.iterate_succ_apply, hfy] using hx.2 n

end BlockDual

section MorseSets

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X} {N M : Set X}

theorem IsAttractingBlock.omegaSet_subset_interior (hf : Continuous f)
    (hN : IsAttractingBlock f N) : omegaSet f N ⊆ interior N := by
  rw [← image_omegaSet hf hN.1 (hN.2.trans interior_subset)]
  exact (image_mono (omegaSet_subset hf hN.1 (hN.2.trans interior_subset))).trans hN.2

theorem IsAttractingBlock.Inv_closure_sdiff_eq_blockDual (hf : Continuous f)
    (hN : IsAttractingBlock f N) (hM : IsAttractingBlock f M) :
    Inv f (closure (N \ M)) = blockDual f (omegaSet f N) M := by
  rw [hN.Inv_closure_sdiff hM]
  apply Subset.antisymm
  · rintro x ⟨γ, hγ, hγ0, hγNM⟩
    refine ⟨?_, fun n hn ↦ (hγNM n).2 ?_⟩
    · rw [← Inv_eq_omegaSet hf hN.1 (hN.2.trans interior_subset)]
      exact ⟨γ, hγ, hγ0, fun n ↦ (hγNM n).1⟩
    · simpa only [← hγ0, hγ.iterate] using hn
  · rw [← Inv_eq_of_image_eq (image_blockDual
      (image_omegaSet hf hN.1 (hN.2.trans interior_subset)) (hM.2.trans interior_subset))]
    apply Inv_mono
    intro x hx
    exact ⟨omegaSet_subset hf hN.1 (hN.2.trans interior_subset) hx.1,
      by simpa only [Function.iterate_zero, id_eq] using hx.2 0⟩

theorem IsAttractingBlock.blockDual_empty_iff (hf : Continuous f)
    (hN : IsAttractingBlock f N) (hM : IsAttractingBlock f M) (hMN : M ⊆ N) :
    blockDual f (omegaSet f N) M = ∅ ↔ omegaSet f N = omegaSet f M := by
  have hforwardN := hN.2.trans interior_subset
  have hforwardM := hM.2.trans interior_subset
  have hA := image_omegaSet hf hN.1 hforwardN
  constructor
  · intro hempty
    let U := fun n : ℕ ↦ f^[n] ⁻¹' interior M
    have hU : Monotone U := monotone_nat_of_le_succ fun n x hx ↦ by
      change f^[n + 1] x ∈ interior M
      rw [Function.iterate_succ_apply']
      exact hM.2 (mem_image_of_mem f (interior_subset hx))
    have hcover : omegaSet f N ⊆ ⋃ n, U n := by
      intro x hx
      have havoid : ¬ ∀ n : ℕ, f^[n] x ∉ M := by
        intro h
        have : x ∈ blockDual f (omegaSet f N) M := ⟨hx, h⟩
        rw [hempty] at this
        exact this
      push Not at havoid
      obtain ⟨n, hn⟩ := havoid
      apply mem_iUnion.mpr
      refine ⟨n + 1, ?_⟩
      change f^[n + 1] x ∈ interior M
      rw [Function.iterate_succ_apply']
      exact hM.2 (mem_image_of_mem f hn)
    obtain ⟨k, hk⟩ := (isCompact_omegaSet hf hN.1 hforwardN).elim_directed_cover U
      (fun n ↦ isOpen_interior.preimage (hf.iterate n)) hcover hU.directed_le
    have hAM : omegaSet f N ⊆ M := by
      have hkA : f^[k] '' omegaSet f N = omegaSet f N := by
        rw [image_iterate_eq]
        exact Function.IsFixedPt.iterate hA k
      rw [← hkA]
      rintro _ ⟨x, hx, rfl⟩
      exact interior_subset (hk hx)
    exact Subset.antisymm (image_eq_subset_omegaSet hf hM.1 hforwardM hA hAM)
      (omegaSet_mono hMN)
  · intro h
    apply eq_empty_iff_forall_notMem.mpr
    intro x hx
    exact hx.2 0 (by simpa only [Function.iterate_zero, id_eq] using
      omegaSet_subset hf hM.1 hforwardM (h ▸ hx.1))

theorem IsAttractingBlock.Inv_closure_sdiff_empty_iff (hf : Continuous f)
    (hN : IsAttractingBlock f N) (hM : IsAttractingBlock f M) (hMN : M ⊆ N) :
    Inv f (closure (N \ M)) = ∅ ↔ omegaSet f N = omegaSet f M := by
  rw [hN.Inv_closure_sdiff_eq_blockDual hf hM, hN.blockDual_empty_iff hf hM hMN]

end MorseSets

section PointOmega

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X}

/-- A forward orbit in a compact set has a nonempty omega limit. -/
theorem omegaSet_singleton_nonempty {N : Set X} (hN : IsCompact N) {x : X}
    (hx : ∀ n : ℕ, f^[n] x ∈ N) : (omegaSet f {x}).Nonempty := by
  let t := fun n : ℕ ↦ closure (⋃ k ≥ n, f^[k] '' {x})
  have htN : ∀ n, t n ⊆ N := by
    intro n
    apply closure_minimal _ hN.isClosed
    exact iUnion₂_subset fun k _ ↦ by simpa only [image_singleton, singleton_subset_iff] using hx k
  exact IsCompact.nonempty_iInter_of_sequence_nonempty_isCompact_isClosed t
    (fun n ↦ closure_mono (iUnion₂_subset fun k hk ↦
      subset_iUnion₂_of_subset k (le_trans (Nat.le_succ n) hk) subset_rfl))
    (fun n ↦ ⟨f^[n] x, subset_closure (mem_iUnion.mpr ⟨n,
      mem_iUnion.mpr ⟨le_rfl, mem_image_of_mem _ (mem_singleton x)⟩⟩)⟩)
    (hN.of_isClosed_subset isClosed_closure (htN 0)) (fun _ ↦ isClosed_closure)

/-- Once an orbit enters a compact forward-invariant set, its omega limit is contained in that
set's omega limit. -/
theorem omegaSet_singleton_subset_of_enters (hf : Continuous f) {M : Set X}
    (hM : IsCompact M) (hforward : f '' M ⊆ M) {x : X} {n : ℕ} (hx : f^[n] x ∈ M) :
    omegaSet f {x} ⊆ omegaSet f M := by
  rw [omegaSet_eq_iInter_iterate hf hM hforward]
  apply subset_iInter
  intro m
  refine (iInter_subset (fun j : ℕ ↦ closure (⋃ k ≥ j, f^[k] '' {x})) (n + m)).trans ?_
  apply closure_minimal _ (hM.image (hf.iterate m)).isClosed
  apply iUnion₂_subset
  intro k hk
  rintro _ ⟨y, hy, rfl⟩
  have hyx : y = x := mem_singleton_iff.mp hy
  subst y
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hk
  have hnd : f^[n + d] x ∈ M := by
    rw [Nat.add_comm n d, Function.iterate_add_apply]
    simpa only [Function.iterate_zero, id_eq, image_id'] using
      (iterate_image_antitone hforward (Nat.zero_le d)) (mem_image_of_mem _ hx)
  have heq : n + m + d = m + (n + d) := by omega
  rw [heq, Function.iterate_add_apply]
  exact mem_image_of_mem _ hnd

/-- The dual repeller relative to `S`, using the singleton omega-limit convention. -/
def attractorDual (f : X → X) (S A : Set X) : Set X :=
  {x | x ∈ S ∧ omegaSet f {x} ∩ A = ∅}

theorem blockDual_eq_attractorDual (hf : Continuous f) {N M : Set X}
    (hN : IsAttractingBlock f N) (hM : IsAttractingBlock f M) :
    blockDual f (omegaSet f N) M =
      attractorDual f (omegaSet f N) (omegaSet f M) := by
  ext x
  constructor
  · rintro ⟨hx, havoid⟩
    refine ⟨hx, eq_empty_iff_forall_notMem.mpr fun y hy ↦ ?_⟩
    have hyc := mem_iInter.mp hy.1 0
    obtain ⟨z, hzM, hz⟩ := mem_closure_iff.mp hyc (interior M) isOpen_interior
      (hM.omegaSet_subset_interior hf hy.2)
    obtain ⟨n, hn⟩ := mem_iUnion.mp hz
    obtain ⟨_, hz⟩ := mem_iUnion.mp hn
    obtain ⟨w, hw, hweq⟩ := hz
    have hwx : w = x := mem_singleton_iff.mp hw
    subst w
    exact havoid n (interior_subset (hweq.symm ▸ hzM))
  · rintro ⟨hx, hdisjoint⟩
    refine ⟨hx, fun n hn ↦ ?_⟩
    have hforwardN := hN.2.trans interior_subset
    have hxN := omegaSet_subset hf hN.1 hforwardN hx
    have hiter : ∀ k : ℕ, f^[k] x ∈ N := by
      intro k
      simpa only [Function.iterate_zero, id_eq, image_id'] using
        (iterate_image_antitone hforwardN (Nat.zero_le k)) (mem_image_of_mem _ hxN)
    obtain ⟨y, hy⟩ := omegaSet_singleton_nonempty hN.1 hiter
    have hyM := omegaSet_singleton_subset_of_enters hf hM.1
      (hM.2.trans interior_subset) hn hy
    have : y ∈ omegaSet f {x} ∩ omegaSet f M := ⟨hy, hyM⟩
    rw [hdisjoint] at this
    exact this

end PointOmega

end LatentEquivariant

namespace LatentEquivariant

open Set Topology

variable {X : Type*} [TopologicalSpace X] [T2Space X] {f : X → X} {S M N : Set X}

omit [T2Space X] in
theorem blockDual_eq_avoid_interior (hM : IsAttractingBlock f M) :
    blockDual f S M = S ∩ ⋂ n : ℕ, f^[n] ⁻¹' (interior M)ᶜ := by
  ext x
  constructor
  · rintro ⟨hx, havoid⟩
    exact ⟨hx, mem_iInter.mpr fun n h ↦ havoid n (interior_subset h)⟩
  · rintro ⟨hx, havoid⟩
    refine ⟨hx, fun n hn ↦ ?_⟩
    have h := hM.2 (mem_image_of_mem f hn)
    exact mem_iInter.mp havoid (n + 1) (by
      simpa only [Function.iterate_succ_apply'] using h)

omit [T2Space X] in
/-- The dual repeller defined by a trapping block is a closed subset of its compact ambient set. -/
theorem isCompact_blockDual (hf : Continuous f) (hS : IsCompact S)
    (hM : IsAttractingBlock f M) : IsCompact (blockDual f S M) := by
  rw [blockDual_eq_avoid_interior hM]
  exact hS.inter_right (isClosed_iInter fun n ↦
    isOpen_interior.isClosed_compl.preimage (hf.iterate n))

theorem IsAttractingBlock.isCompact_Inv_closure_sdiff (hf : Continuous f)
    (hN : IsAttractingBlock f N) (hM : IsAttractingBlock f M) :
    IsCompact (Inv f (closure (N \ M))) := by
  rw [hN.Inv_closure_sdiff_eq_blockDual hf hM]
  exact isCompact_blockDual hf (isCompact_omegaSet hf hN.1
    (hN.2.trans interior_subset)) hM

end LatentEquivariant
