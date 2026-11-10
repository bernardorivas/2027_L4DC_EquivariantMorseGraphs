import LatentEquivariant.Combinatorial.MorseOrder

/-!
# The lifted Morse order

Proposition `prop:lifted-order` of the paper. A group `G` acts on the vertices `X` of a
directed graph `R`, and `R` is equivariant. The Morse sets are the recurrent components, ordered
by reverse reachability (`RecurrentComponent`): `C ≤ D` means that `D` reaches `C`. For every
Morse set `P` of the quotient graph, `rep P` is a chosen Morse set `M̂_P` over `P`, with stabilizer
`H_P`. Let `W(Q, P) = {g | M̂_Q ⇝ g M̂_P}` (`liftedWalkSet`).

* `liftedMorseEquiv`: for finite `G`, the Morse sets upstairs are `⊔_P G/H_P`, by
  `⟨P, g H_P⟩ ↦ g M̂_P`.
* `smul_rep_le_smul_rep_iff`: `h M̂_P ≤ k M̂_Q` if and only if `k⁻¹ h ∈ W(Q, P)`, and
  `smul_rep_lt_smul_rep_iff` gives the strict form for `P ≠ Q`.
* `stabilizer_mul_liftedWalkSet_mul_stabilizer`: `H_Q W(Q, P) H_P = W(Q, P)`, so `W(Q, P)` is a
  union of double cosets `H_Q g H_P`.
* `liftedWalkSet_eq_voltages`: `W(Q, P)` is the set of voltages of the positive quotient walks
  from `q` to `p`, for vertices `q` of `Q` and `p` of `P` whose vertex representatives lie in
  `M̂_Q` and `M̂_P`. `smul_rep_le_smul_rep_iff_voltage` combines this with the order criterion.
* `liftedWalkSet_nonempty_iff`: `W(Q, P)` is nonempty if and only if `P ≤ Q`, and
  `liftedWalkSet_self`: `W(P, P) = H_P`.

The paper states the criterion for comparable `[p] < [q]`. Here `W` is defined for every pair,
and the criterion holds for every pair with `≤`. On the diagonal it says that distinct translates
of `M̂_P` are incomparable. Only `liftedMorseEquiv`, `liftedWalkSet_self` and the strict form use
finiteness of `G`. `smul_rep_le_smul_rep_iff_voltage` is the identity behind the reconstruction
of the order from the quotient walks. Its hypotheses hold when `M̂_Q` is taken to be the component
of the representative of a chosen vertex of `Q`.
-/

namespace LatentEquivariant

open Set Pointwise MulAction Relation

section LiftedOrder

variable {G X : Type*} [Group G] [MulAction G X] {R : X → X → Prop}
  [hEquiv : Fact (∀ (g : G) x y, R x y → R (g • x) (g • y))]

local notation "hR" => hEquiv.out

noncomputable local instance : MulAction G (RecurrentComponent R) := recurrentComponentAction hR

/-- Every vertex of a recurrent component lies on a positive cycle. -/
theorem RecurrentComponent.transGen_of_mem (C : RecurrentComponent R) {x : X} (hx : x ∈ C.val) :
    TransGen R x x :=
  TransGen.trans_right (C.reachable_of_mem hx C.point_mem)
    (C.point_recurrent.trans_left (C.reachable_of_mem C.point_mem hx))

theorem recurrentComponent_smul_le_smul_iff (g : G) {C D : RecurrentComponent R} :
    g • C ≤ g • D ↔ C ≤ D := by
  refine ⟨fun h ↦ ?_, recurrentComponent_smul_le_smul g⟩
  simpa only [inv_smul_smul] using recurrentComponent_smul_le_smul g⁻¹ h

/-- For finite `G`, a Morse set below its own translate is fixed by the translation. -/
theorem recurrentComponent_smul_eq_of_smul_le [Finite G] {g : G} {C : RecurrentComponent R}
    (h : g • C ≤ C) : g • C = C := by
  have hgx : g • C.point ∈ (g • C).val := by
    rw [recurrentComponent_smul_val]
    exact smul_mem_smul_set C.point_mem
  have hp := (RecurrentComponent.le_iff_reachable (g • C) C hgx C.point_mem).mp h
  exact le_antisymm h ((RecurrentComponent.le_iff_reachable C (g • C) C.point_mem hgx).mpr
    (reflTransGen_smul_reverse hR hp))

variable (rep : RecurrentComponent (orbitRelation G R) → RecurrentComponent R)

/-- `W(Q, P) = {g | M̂_Q ⇝ g M̂_P}`, in the reverse-reachability order `g M̂_P ≤ M̂_Q`. -/
def liftedWalkSet (Q P : RecurrentComponent (orbitRelation G R)) : Set G :=
  {g | g • rep P ≤ rep Q}

theorem mem_liftedWalkSet {Q P : RecurrentComponent (orbitRelation G R)} {g : G} :
    g ∈ liftedWalkSet rep Q P ↔ g • rep P ≤ rep Q :=
  Iff.rfl

/-- Proposition `prop:lifted-order`, the order: `h M̂_P ≤ k M̂_Q` if and only if
`k⁻¹ h ∈ W(Q, P)`. -/
theorem smul_rep_le_smul_rep_iff (P Q : RecurrentComponent (orbitRelation G R)) (h k : G) :
    h • rep P ≤ k • rep Q ↔ k⁻¹ * h ∈ liftedWalkSet rep Q P := by
  rw [mem_liftedWalkSet, ← recurrentComponent_smul_le_smul_iff k⁻¹ (C := h • rep P),
    inv_smul_smul, smul_smul]

/-- Proposition `prop:lifted-order`, double cosets: `H_Q W(Q, P) H_P = W(Q, P)`. -/
theorem stabilizer_mul_liftedWalkSet_mul_stabilizer (Q P : RecurrentComponent (orbitRelation G R)) :
    (stabilizer G (rep Q) : Set G) * liftedWalkSet rep Q P * (stabilizer G (rep P) : Set G) =
      liftedWalkSet rep Q P := by
  apply Subset.antisymm
  · rintro _ ⟨_, ⟨a, ha, g, hg, rfl⟩, b, hb, rfl⟩
    rw [mem_liftedWalkSet, mul_smul, mul_smul, mem_stabilizer_iff.mp hb]
    calc a • g • rep P ≤ a • rep Q := recurrentComponent_smul_le_smul a hg
      _ = rep Q := mem_stabilizer_iff.mp ha
  · intro g hg
    exact ⟨g, ⟨1, one_mem _, g, hg, one_mul g⟩, 1, one_mem _, mul_one g⟩

/-- Proposition `prop:lifted-order`, computation from quotient walks. Let `vrep` choose a vertex
in every orbit, and let `q` and `p` be vertex orbits whose representatives lie in `M̂_Q` and
`M̂_P`. Then `W(Q, P)` is the set of voltages of the positive voltage paths from `q` to `p`, the
products of the edge labels of the quotient walks. -/
theorem liftedWalkSet_eq_voltages (vrep : orbitRel.Quotient G X → X)
    (hvrep : ∀ q, orbitMap G X (vrep q) = q) {Q P : RecurrentComponent (orbitRelation G R)}
    {q p : orbitRel.Quotient G X} (hq : vrep q ∈ (rep Q).val) (hp : vrep p ∈ (rep P).val) :
    liftedWalkSet rep Q P = {g | ∃ π : VoltagePath (G := G) R vrep q p,
      0 < voltageLength (G := G) R vrep π ∧ pathVoltage (G := G) R vrep π = g} := by
  ext g
  have hgp : g • vrep p ∈ (g • rep P).val := by
    rw [recurrentComponent_smul_val]
    exact smul_mem_smul_set hp
  rw [mem_liftedWalkSet, mem_setOf_eq, exists_voltagePath_iff hR vrep hvrep,
    RecurrentComponent.le_iff_reachable _ _ hgp hq]
  refine ⟨fun h ↦ ?_, TransGen.to_reflTransGen⟩
  rcases reflTransGen_iff_eq_or_transGen.mp h with heq | hpath
  · rw [heq]
    exact (rep Q).transGen_of_mem hq
  · exact hpath

/-- The order criterion with `W(Q, P)` computed from quotient walks: `h M̂_P ≤ k M̂_Q` if and only
if some positive voltage path from `q` to `p` has voltage `k⁻¹ h`. -/
theorem smul_rep_le_smul_rep_iff_voltage (vrep : orbitRel.Quotient G X → X)
    (hvrep : ∀ q, orbitMap G X (vrep q) = q) {Q P : RecurrentComponent (orbitRelation G R)}
    {q p : orbitRel.Quotient G X} (hq : vrep q ∈ (rep Q).val) (hp : vrep p ∈ (rep P).val)
    (h k : G) :
    h • rep P ≤ k • rep Q ↔ ∃ π : VoltagePath (G := G) R vrep q p,
      0 < voltageLength (G := G) R vrep π ∧ pathVoltage (G := G) R vrep π = k⁻¹ * h := by
  rw [smul_rep_le_smul_rep_iff, liftedWalkSet_eq_voltages rep vrep hvrep hq hp, mem_setOf_eq]

variable [Finite G] {rep}
  (hrep : ∀ P, quotientRecurrentComponent (G := G) (R := R) (rep P) = P)

include hrep

theorem quotientRecurrentComponent_smul_rep (g : G) (P : RecurrentComponent (orbitRelation G R)) :
    quotientRecurrentComponent (G := G) (R := R) (g • rep P) = P := by
  rw [quotientRecurrentComponent_smul, hrep]

/-- `W(Q, P)` is nonempty if and only if `P ≤ Q`. -/
theorem liftedWalkSet_nonempty_iff (Q P : RecurrentComponent (orbitRelation G R)) :
    (liftedWalkSet rep Q P).Nonempty ↔ P ≤ Q := by
  conv_rhs => rw [← hrep P, ← hrep Q]
  rw [quotientRecurrentComponent_le_iff]
  constructor
  · rintro ⟨g, hg⟩
    exact ⟨g⁻¹, by simpa only [inv_smul_smul] using recurrentComponent_smul_le_smul g⁻¹ hg⟩
  · rintro ⟨g, hg⟩
    refine ⟨g⁻¹, (mem_liftedWalkSet rep).mpr ?_⟩
    simpa only [inv_smul_smul] using recurrentComponent_smul_le_smul g⁻¹ hg

omit hrep in
/-- `W(P, P) = H_P`: on the diagonal the order criterion says that distinct translates of `M̂_P`
are incomparable. -/
theorem liftedWalkSet_self (P : RecurrentComponent (orbitRelation G R)) :
    liftedWalkSet rep P P = (stabilizer G (rep P) : Set G) := by
  ext g
  rw [mem_liftedWalkSet, SetLike.mem_coe, mem_stabilizer_iff]
  exact ⟨recurrentComponent_smul_eq_of_smul_le, le_of_eq⟩

/-- Proposition `prop:lifted-order`, strict order: for `P ≠ Q`, `h M̂_P < k M̂_Q` if and only if
`k⁻¹ h ∈ W(Q, P)`. -/
theorem smul_rep_lt_smul_rep_iff {P Q : RecurrentComponent (orbitRelation G R)} (hPQ : P ≠ Q)
    (h k : G) : h • rep P < k • rep Q ↔ k⁻¹ * h ∈ liftedWalkSet rep Q P := by
  rw [← smul_rep_le_smul_rep_iff, lt_iff_le_and_ne, and_iff_left_iff_imp]
  intro _ heq
  apply hPQ
  rw [← quotientRecurrentComponent_smul_rep hrep h P, heq, quotientRecurrentComponent_smul_rep hrep]

/-- Proposition `prop:lifted-order`, the Morse sets: `⟨P, g H_P⟩ ↦ g M̂_P` is a bijection from
`⊔_P G/H_P` onto the Morse sets. -/
theorem bijective_liftedMorse :
    Function.Bijective fun c : (Σ P : RecurrentComponent (orbitRelation G R),
      G ⧸ stabilizer G (rep P)) ↦ ofQuotientStabilizer G (rep c.1) c.2 := by
  constructor
  · rintro ⟨P, c⟩ ⟨P', c'⟩ hcc
    obtain ⟨g, rfl⟩ := QuotientGroup.mk_surjective c
    obtain ⟨g', rfl⟩ := QuotientGroup.mk_surjective c'
    simp only [ofQuotientStabilizer_mk] at hcc
    have hP : P = P' := by
      rw [← quotientRecurrentComponent_smul_rep hrep g P, hcc,
        quotientRecurrentComponent_smul_rep hrep]
    subst hP
    have hc : (g : G ⧸ stabilizer G (rep P)) = g' :=
      injective_ofQuotientStabilizer G (rep P) (by simpa only [ofQuotientStabilizer_mk] using hcc)
    rw [hc]
  · intro C
    set P := quotientRecurrentComponent (G := G) (R := R) C
    obtain ⟨g, hg⟩ := (quotientRecurrentComponent_eq_iff (rep P) C).mp (hrep P)
    exact ⟨⟨P, (g : G ⧸ stabilizer G (rep P))⟩, by simp only [ofQuotientStabilizer_mk, hg]⟩

/-- The Morse sets upstairs are `⊔_P G/H_P`. -/
noncomputable def liftedMorseEquiv :
    (Σ P : RecurrentComponent (orbitRelation G R), G ⧸ stabilizer G (rep P)) ≃
      RecurrentComponent R :=
  Equiv.ofBijective _ (bijective_liftedMorse hrep)

@[simp] theorem liftedMorseEquiv_apply (P : RecurrentComponent (orbitRelation G R)) (g : G) :
    liftedMorseEquiv hrep ⟨P, (g : G ⧸ stabilizer G (rep P))⟩ = g • rep P :=
  ofQuotientStabilizer_mk _ _ g

end LiftedOrder

end LatentEquivariant
