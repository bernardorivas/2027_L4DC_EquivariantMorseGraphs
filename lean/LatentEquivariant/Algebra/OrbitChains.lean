import LatentEquivariant.Orbit.Quotient
import LatentEquivariant.Algebra.Chains

/-!
# Orbit chains and the transfer

Chain-level reconstruction from the orbit space: for `G`-invariant subcomplexes `L ⊆ N`,
the component of the trivial representation in the cellular chains `C_*(N, L; F)`, for a field
`F` with `char F ∤ |G|`, is the chain complex of the orbit space `(N/G, L/G)`. This is the
transfer isomorphism of [bredon:72] at the chain level. For a group of order two it is the
identification of the `+1` eigenspace in the proof of Proposition `prop:z2_reconstruction`.

## Model

A group `G` permutes finite sets `I` and `J` of cells in two consecutive dimensions, and the
incidence numbers `inc i j = [i : j]` in a commutative ring `k` are equivariant,
`[g i : g j] = [i : j]`. Chains are functions `I → k` with the boundary `bd inc`. Under the grid
condition of the paper (an element that maps a cell to itself fixes it pointwise), orientations
transported along orbits are well defined, so `G` permutes suitably oriented cells without signs,
which is the model used here. The model is weaker than the grid condition (see `Chains.lean`), and
the identification of `orbitInc` with the cellular complex of `(N/G, L/G)` uses the grid condition
itself; that identification is not formalized. Relative chains of a `G`-invariant pair `(N, L)`
are covered by taking for `I` and `J` the cells of `N` not in `L`, with the restricted incidence
numbers, so absolute statements suffice.

## Main statements

* `push`, `orbitInc`, `push_bd`: the map `π_#` of the orbit map `π : I → I/G` is a chain map to
  the orbit complex, whose incidence numbers are well defined by `orbitInc_orbitMap`. This holds
  over any commutative ring.
* `invariantChains`, `bd_mem_invariantChains`, `bdInvariant`: the invariant chains form a
  subcomplex, over any commutative ring.
* `transferEquiv`, `transferEquiv_bdInvariant`, `transferEquiv_conj_bdInvariant`: if the
  cardinality of every orbit is a unit in `k`, then `π_#` restricts to an isomorphism of chain
  complexes from the invariant chains onto the orbit complex, with inverse `orbitLift`.
  `transferEquivOfIsUnitCard` is the case of `|G|` a unit, the hypothesis `char F ∤ |G|` of the
  paper.
* `bijective_push_domRestrict_iff`: the hypothesis on the orbits is also necessary for `π_#` to
  restrict to an isomorphism.
* `orbitFunEquiv_bdInvariant`, `scaledOrbitEquiv_bdInvariant`: other isomorphisms need other
  hypotheses. If every stabilizer order is a unit in `k`, the invariant chains are isomorphic to
  the orbit complex, and for a free action this holds over any ring. Like everything in this
  file, this is a statement about chains: when `char k` divides `|G|`, the homology of the
  invariant chains need not be the invariant part of the homology.
* `not_injective_push_perm_two`, `not_exists_chainEquiv_reflection`: over `ZMod 2`, `π_#` is not
  injective on the invariant chains of two swapped cells, and for the reflection of `[-1, 1]`
  relative to `{-1, 1}` no isomorphisms intertwine the invariant boundary with the orbit boundary.
  `exists_chainEquiv_circle`: for the free antipodal map of a circle over `ZMod 2` they do.

No finiteness of `G` is needed, except to state the hypothesis on `|G|` in
`transferEquivOfIsUnitCard`: the cells are finite, so the orbits are.
-/

namespace LatentEquivariant

open Finset MulAction

open scoped Classical

variable {G k I J : Type*} [Group G] [CommRing k] [MulAction G I] [MulAction G J]

section Push

variable [Fintype I]

variable (G) in
/-- The chain map `π_#` of the orbit map `π : I → I/G`, `(π_# c)(q) = ∑_{π(i) = q} c(i)`: a cell
is sent to its orbit. -/
noncomputable def push : (I → k) →ₗ[k] (orbitRel.Quotient G I → k) where
  toFun c q := ∑ i ∈ univ.filter (fun i ↦ orbitMap G I i = q), c i
  map_add' c d := by
    ext q
    simp [sum_add_distrib]
  map_smul' a c := by
    ext q
    simp [mul_sum]

@[simp] theorem push_apply (c : I → k) (q : orbitRel.Quotient G I) :
    push G c q = ∑ i ∈ univ.filter (fun i ↦ orbitMap G I i = q), c i :=
  rfl

/-- The fiber of the orbit map through `i` is the orbit of `i`, so it has `|G i|` cells. -/
theorem card_filter_orbitMap_eq_ncard (i : I) :
    (univ.filter (fun i' ↦ orbitMap G I i' = orbitMap G I i)).card = (orbit G i).ncard := by
  rw [← Nat.card_coe_set_eq]
  exact (Nat.subtype_card _ fun i' ↦ by
    simp only [mem_filter, mem_univ, true_and, orbitMap_eq_iff, mem_orbit_iff]).symm

end Push

section OrbitInc

variable [Fintype J]

variable (G) in
/-- The incidence numbers of the orbit complex, `[q : p] = ∑_{π(j) = p} [i : j]` for a chosen
cell `i` with `π(i) = q`. -/
noncomputable def orbitInc (inc : I → J → k) (q : orbitRel.Quotient G I)
    (p : orbitRel.Quotient G J) : k :=
  ∑ j ∈ univ.filter (fun j ↦ orbitMap G J j = p), inc q.out j

/-- The incidence numbers of the orbit complex do not depend on the chosen cell of the orbit,
since the incidence numbers of the cells are equivariant. -/
theorem orbitInc_orbitMap {inc : I → J → k} (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (i : I) (p : orbitRel.Quotient G J) :
    orbitInc G inc (orbitMap G I i) p =
      ∑ j ∈ univ.filter (fun j ↦ orbitMap G J j = p), inc i j := by
  obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp (Quotient.out_eq' (orbitMap G I i))
  rw [orbitInc, ← hg, sum_filter, sum_filter]
  symm
  exact Fintype.sum_equiv (toPerm g) _ _ fun j ↦ by
    rw [toPerm_apply, orbitMap_smul, hinc]

variable [Fintype I]

/-- `π_#` is a chain map, `π_# ∂ = ∂ π_#`, over any commutative ring, when the incidence numbers
are equivariant. -/
theorem push_bd {inc : I → J → k} (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (c : I → k) : push G (bd inc c) = bd (orbitInc G inc) (push G c) := by
  ext p
  simp only [push_apply, bd_apply, sum_mul]
  symm
  calc ∑ q, ∑ i ∈ univ.filter (fun i ↦ orbitMap G I i = q), c i * orbitInc G inc q p
      = ∑ q, ∑ i ∈ univ.filter (fun i ↦ orbitMap G I i = q),
          c i * orbitInc G inc (orbitMap G I i) p :=
        sum_congr rfl fun q _ ↦ sum_congr rfl fun i hi ↦ by rw [(mem_filter.mp hi).2]
    _ = ∑ i, c i * orbitInc G inc (orbitMap G I i) p := sum_fiberwise _ _ _
    _ = ∑ j ∈ univ.filter (fun j ↦ orbitMap G J j = p), ∑ i, c i * inc i j := by
        simp_rw [orbitInc_orbitMap hinc, mul_sum]
        exact sum_comm

end OrbitInc

section Invariant

variable (G k I) in
/-- The `G`-invariant chains, `c(g i) = c(i)` for all `g` and `i`. Under the hypothesis of
`transferEquiv`, for example when `|G|` is a unit in `k`, they are the component of the trivial
representation in the chains on `I`. -/
def invariantChains : Submodule k (I → k) where
  carrier := {c | ∀ (g : G) (i : I), c (g • i) = c i}
  add_mem' hc hd g i := by simp [hc g i, hd g i]
  zero_mem' _ _ := rfl
  smul_mem' a _ hc g i := by simp [hc g i]

theorem mem_invariantChains {c : I → k} :
    c ∈ invariantChains G k I ↔ ∀ (g : G) (i : I), c (g • i) = c i :=
  Iff.rfl

/-- The lift of an orbit chain `d` to the invariant chain `i ↦ |G i|⁻¹ d(π(i))`, with
`Ring.inverse` for the inverse: the transfer `d ↦ d ∘ π` rescaled by the orbit cardinalities. It
inverts `π_#` on invariant chains (`transferEquiv`). -/
noncomputable def orbitLift (d : orbitRel.Quotient G I → k) (i : I) : k :=
  Ring.inverse ((orbit G i).ncard : k) * d (orbitMap G I i)

/-- The lift of an orbit chain is invariant, since `|G (g i)| = |G i|` and `π(g i) = π(i)`. -/
theorem orbitLift_mem (d : orbitRel.Quotient G I → k) : orbitLift d ∈ invariantChains G k I :=
  mem_invariantChains.mpr fun g i ↦ by simp only [orbitLift, orbit_smul, orbitMap_smul]

/-- The cardinality of an orbit divides `|G|` by the orbit-stabilizer theorem, so it is a unit in
`k` when `|G|` is. -/
theorem isUnit_ncard_orbit [Fintype G] (hG : IsUnit (Fintype.card G : k)) (i : I) :
    IsUnit ((orbit G i).ncard : k) := by
  have hdvd : (orbit G i).ncard ∣ Fintype.card G := by
    rw [← index_stabilizer, Fintype.card_eq_nat_card]
    exact (stabilizer G i).index_dvd_card
  exact isUnit_of_dvd_unit (Nat.cast_dvd_cast hdvd) hG

variable [Fintype I] {inc : I → J → k}

/-- The boundary maps invariant chains to invariant chains when the incidence numbers are
equivariant. -/
theorem bd_mem_invariantChains (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    {c : I → k} (hc : c ∈ invariantChains G k I) : bd inc c ∈ invariantChains G k J := by
  rw [mem_invariantChains] at hc ⊢
  intro g j
  simp only [bd_apply]
  symm
  exact Fintype.sum_equiv (toPerm g) _ _ fun i ↦ by rw [toPerm_apply, hc g i, hinc g i j]

/-- The boundary of the invariant subcomplex: `bd inc` restricted to invariant chains. -/
def bdInvariant (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j) :
    invariantChains G k I →ₗ[k] invariantChains G k J :=
  (bd inc).restrict fun _ hc ↦ bd_mem_invariantChains hinc hc

@[simp] theorem coe_bdInvariant_apply (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (c : invariantChains G k I) : (bdInvariant hinc c : J → k) = bd inc c :=
  rfl

/-- On an invariant chain, `π_#` multiplies by the cardinality of the orbit:
`(π_# c)(π(i)) = |G i| c(i)`. For a group of order two this is `e + g e ↦ 2 [e]` in the proof of
Proposition `prop:z2_reconstruction`. -/
theorem push_apply_orbitMap_of_mem {c : I → k} (hc : c ∈ invariantChains G k I) (i : I) :
    push G c (orbitMap G I i) = ((orbit G i).ncard : k) * c i := by
  rw [push_apply, ← card_filter_orbitMap_eq_ncard, ← nsmul_eq_mul, ← sum_const]
  refine sum_congr rfl fun i' hi' ↦ ?_
  obtain ⟨g, rfl⟩ := orbitMap_eq_iff.mp (mem_filter.mp hi').2
  exact mem_invariantChains.mp hc g i

/-- `orbitLift` is a left inverse of `π_#` on invariant chains when orbit cardinalities are
units. -/
theorem orbitLift_push (h : ∀ i : I, IsUnit ((orbit G i).ncard : k)) {c : I → k}
    (hc : c ∈ invariantChains G k I) : orbitLift (push G c) = c :=
  funext fun i ↦ by
    rw [orbitLift, push_apply_orbitMap_of_mem hc, Ring.inverse_mul_cancel_left _ _ (h i)]

/-- `orbitLift` is a right inverse of `π_#` when orbit cardinalities are units. -/
theorem push_orbitLift (h : ∀ i : I, IsUnit ((orbit G i).ncard : k))
    (d : orbitRel.Quotient G I → k) : push G (orbitLift d) = d := by
  funext q
  obtain ⟨i, rfl⟩ := orbitMap_surjective q
  rw [push_apply_orbitMap_of_mem (orbitLift_mem d), orbitLift,
    Ring.mul_inverse_cancel_left _ _ (h i)]

/-- The **transfer isomorphism** at the chain level: if the cardinality of every orbit is a unit
in `k`, then `π_#` restricts to an isomorphism from the invariant chains onto the chains of the
orbit complex. The inverse is `orbitLift`. Section `sec:conley-symmetry` uses its homology form,
citing [bredon:72], for a field `F` with `char F ∤ |G|`. -/
noncomputable def transferEquiv (h : ∀ i : I, IsUnit ((orbit G i).ncard : k)) :
    invariantChains G k I ≃ₗ[k] (orbitRel.Quotient G I → k) where
  toLinearMap := (push G).domRestrict (invariantChains G k I)
  invFun d := ⟨orbitLift d, orbitLift_mem d⟩
  left_inv c := Subtype.ext (orbitLift_push h c.2)
  right_inv d := push_orbitLift h d

@[simp] theorem transferEquiv_apply (h : ∀ i : I, IsUnit ((orbit G i).ncard : k))
    (c : invariantChains G k I) : transferEquiv h c = push G c :=
  rfl

@[simp] theorem coe_transferEquiv_symm_apply (h : ∀ i : I, IsUnit ((orbit G i).ncard : k))
    (d : orbitRel.Quotient G I → k) : ((transferEquiv h).symm d : I → k) = orbitLift d :=
  rfl

/-- `transferEquiv` under the hypothesis of the paper: `|G|` is a unit in `k`, as in a field with
`char F ∤ |G|`. -/
noncomputable def transferEquivOfIsUnitCard [Fintype G] (hG : IsUnit (Fintype.card G : k)) :
    invariantChains G k I ≃ₗ[k] (orbitRel.Quotient G I → k) :=
  transferEquiv (isUnit_ncard_orbit hG)

@[simp] theorem transferEquivOfIsUnitCard_apply [Fintype G] (hG : IsUnit (Fintype.card G : k))
    (c : invariantChains G k I) : transferEquivOfIsUnitCard hG c = push G c :=
  rfl

/-- Conversely, if `π_#` maps the invariant chains onto the orbit chains, then the cardinality of
every orbit is a unit in `k`. So the hypothesis of `transferEquiv` cannot be weakened for the map
`π_#`; other isomorphisms onto the orbit chains need other hypotheses
(`scaledOrbitEquiv_bdInvariant`). -/
theorem isUnit_ncard_orbit_of_surjective
    (h : Function.Surjective ((push G).domRestrict (invariantChains G k I))) (i : I) :
    IsUnit ((orbit G i).ncard : k) := by
  obtain ⟨c, hc⟩ := h (Pi.single (orbitMap G I i) 1)
  have hci := congr_fun hc (orbitMap G I i)
  rw [LinearMap.domRestrict_apply, push_apply_orbitMap_of_mem c.2, Pi.single_eq_same] at hci
  exact IsUnit.of_mul_eq_one _ hci

/-- `π_#` restricts to a bijection from the invariant chains onto the orbit chains if and only if
the cardinality of every orbit is a unit in `k`. -/
theorem bijective_push_domRestrict_iff :
    Function.Bijective ((push G).domRestrict (invariantChains G k I)) ↔
      ∀ i : I, IsUnit ((orbit G i).ncard : k) :=
  ⟨fun hb ↦ isUnit_ncard_orbit_of_surjective hb.2, fun h ↦ (transferEquiv h).bijective⟩

/-- If the cardinality of an orbit vanishes in a nontrivial ring `k`, then `π_#` kills the
indicator chain of that orbit, which is invariant, so `π_#` is not injective on invariant
chains. -/
theorem not_injective_push_of_ncard_orbit_eq_zero [Nontrivial k] {i : I}
    (h : ((orbit G i).ncard : k) = 0) :
    ¬ Function.Injective ((push G).domRestrict (invariantChains G k I)) := by
  intro hinj
  let c : I → k := fun i' ↦ if orbitMap G I i' = orbitMap G I i then 1 else 0
  have hc : c ∈ invariantChains G k I := mem_invariantChains.mpr fun g i' ↦ by simp [c]
  have h0 : push G c = 0 := by
    funext q
    obtain ⟨i', rfl⟩ := orbitMap_surjective q
    rw [push_apply_orbitMap_of_mem hc, Pi.zero_apply]
    by_cases hi' : orbitMap G I i' = orbitMap G I i
    · obtain ⟨g, rfl⟩ := orbitMap_eq_iff.mp hi'
      rw [orbit_smul, h, zero_mul]
    · simp [c, hi']
  have h1 : (push G).domRestrict (invariantChains G k I) ⟨c, hc⟩ =
      (push G).domRestrict (invariantChains G k I) 0 := by
    rw [map_zero]
    exact h0
  have hci := congr_arg (fun c : invariantChains G k I ↦ (c : I → k) i) (hinj h1)
  simp [c] at hci

variable [Fintype J]

/-- `transferEquiv` is an isomorphism of chain complexes: it intertwines the boundary of the
invariant subcomplex with the boundary of the orbit complex. -/
theorem transferEquiv_bdInvariant (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (hI : ∀ i : I, IsUnit ((orbit G i).ncard : k)) (hJ : ∀ j : J, IsUnit ((orbit G j).ncard : k))
    (c : invariantChains G k I) :
    transferEquiv hJ (bdInvariant hinc c) = bd (orbitInc G inc) (transferEquiv hI c) := by
  simp only [transferEquiv_apply, coe_bdInvariant_apply]
  exact push_bd hinc c

/-- Conjugating the boundary of the invariant subcomplex by `transferEquiv` gives the boundary of
the orbit complex. -/
theorem transferEquiv_conj_bdInvariant (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (hI : ∀ i : I, IsUnit ((orbit G i).ncard : k)) (hJ : ∀ j : J, IsUnit ((orbit G j).ncard : k)) :
    (transferEquiv hJ).toLinearMap ∘ₗ bdInvariant hinc ∘ₗ (transferEquiv hI).symm.toLinearMap =
      bd (orbitInc G inc) :=
  LinearMap.ext fun d ↦ by
    rw [LinearMap.comp_apply, LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.coe_coe,
      transferEquiv_bdInvariant hinc hI hJ, LinearEquiv.apply_symm_apply]

/-- The chain-level reconstruction from the orbit space when `|G|` is a unit in `k`:
`transferEquivOfIsUnitCard` intertwines the boundary of the invariant subcomplex with the boundary
of the orbit complex. -/
theorem transferEquivOfIsUnitCard_bdInvariant [Fintype G]
    (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j) (hG : IsUnit (Fintype.card G : k))
    (c : invariantChains G k I) :
    transferEquivOfIsUnitCard hG (bdInvariant hinc c) =
      bd (orbitInc G inc) (transferEquivOfIsUnitCard hG c) :=
  transferEquiv_bdInvariant hinc _ _ c

end Invariant

section Stabilizers

/-! ### Other isomorphisms onto the orbit complex

The orbit condition of `transferEquiv` is exact for `π_#`, but not for the identification of the
invariant chains with the orbit complex. If every stabilizer order is a unit, evaluation at orbit
representatives, rescaled by the stabilizer orders, is another isomorphism. Over a field of
characteristic `p` the orbit condition asks that every stabilizer contain a Sylow `p`-subgroup,
and the stabilizer condition that no stabilizer have order divisible by `p`; `char k ∤ |G|`
implies both. These are statements about chains only.
-/

/-- Evaluation at orbit representatives, `c ↦ (q ↦ c(q.out))`: an isomorphism from the invariant
chains onto the chains on the orbits, over any ring. -/
noncomputable def orbitFunEquiv : invariantChains G k I ≃ₗ[k] (orbitRel.Quotient G I → k) where
  toFun c q := (c : I → k) q.out
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun d := ⟨d ∘ orbitMap G I, mem_invariantChains.mpr fun g i ↦ by simp⟩
  left_inv c := by
    apply Subtype.ext
    funext i
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp (Quotient.out_eq' (orbitMap G I i))
    simp only [Function.comp_apply]
    rw [← hg]
    exact mem_invariantChains.mp c.2 g i
  right_inv d := by
    funext q
    simp only [Function.comp_apply]
    exact congrArg d (Quotient.out_eq' q)

theorem orbitFunEquiv_apply (c : invariantChains G k I) (q : orbitRel.Quotient G I) :
    orbitFunEquiv c q = (c : I → k) q.out :=
  rfl

variable [Fintype I] [Fintype J] {inc : I → J → k}

/-- For a free action, the incidence sum over the orbit of a cell equals the incidence sum over
the orbit of a face. -/
theorem sum_fiber_inc_eq_of_free (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (hI : ∀ (g : G) (i : I), g • i = i → g = 1) (hJ : ∀ (g : G) (j : J), g • j = j → g = 1)
    (i₀ : I) (j₀ : J) :
    ∑ i ∈ univ.filter (fun i ↦ orbitMap G I i = orbitMap G I i₀), inc i j₀ =
      ∑ j ∈ univ.filter (fun j ↦ orbitMap G J j = orbitMap G J j₀), inc i₀ j := by
  have hinjI : Function.Injective (fun g : G ↦ g • i₀) := fun g h hgh ↦ by
    have h1 := hI (h⁻¹ * g) i₀ (by
      simp only at hgh
      rw [mul_smul, hgh, inv_smul_smul])
    exact (inv_mul_eq_one.mp h1).symm
  have hinjJ : Function.Injective (fun g : G ↦ g • j₀) := fun g h hgh ↦ by
    have h1 := hJ (h⁻¹ * g) j₀ (by
      simp only at hgh
      rw [mul_smul, hgh, inv_smul_smul])
    exact (inv_mul_eq_one.mp h1).symm
  have : Finite G := Finite.of_injective _ hinjI
  letI : Fintype G := Fintype.ofFinite G
  have hfI : univ.filter (fun i ↦ orbitMap G I i = orbitMap G I i₀) =
      univ.image (fun g : G ↦ g • i₀) := by
    ext i
    simp [orbitMap_eq_iff]
  have hfJ : univ.filter (fun j ↦ orbitMap G J j = orbitMap G J j₀) =
      univ.image (fun g : G ↦ g • j₀) := by
    ext j
    simp [orbitMap_eq_iff]
  rw [hfI, hfJ, sum_image fun x _ y _ hxy ↦ hinjI hxy, sum_image fun x _ y _ hxy ↦ hinjJ hxy]
  exact Fintype.sum_equiv (Equiv.inv G) _ _ fun g ↦ by
    simpa using (hinc g⁻¹ (g • i₀) j₀).symm

/-- For a free action, `orbitFunEquiv` is an isomorphism of chain complexes from the invariant
chains onto the orbit complex, over any commutative ring. No hypothesis on `|G|` is needed. -/
theorem orbitFunEquiv_bdInvariant (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (hI : ∀ (g : G) (i : I), g • i = i → g = 1) (hJ : ∀ (g : G) (j : J), g • j = j → g = 1)
    (c : invariantChains G k I) :
    orbitFunEquiv (bdInvariant hinc c) = bd (orbitInc G inc) (orbitFunEquiv c) := by
  funext p
  change (bd inc (c : I → k)) p.out = ∑ q, (c : I → k) q.out * orbitInc G inc q p
  rw [bd_apply, ← sum_fiberwise univ (orbitMap G I) (fun i ↦ (c : I → k) i * inc i p.out)]
  refine sum_congr rfl fun q _ ↦ ?_
  obtain ⟨i₀, rfl⟩ := orbitMap_surjective q
  have hc : ∀ i, orbitMap G I i = orbitMap G I i₀ → (c : I → k) i = (c : I → k) i₀ := by
    intro i hi
    obtain ⟨g, rfl⟩ := orbitMap_eq_iff.mp hi
    exact mem_invariantChains.mp c.2 g i₀
  rw [sum_congr rfl fun i hi ↦ by rw [hc i (mem_filter.mp hi).2], ← mul_sum,
    hc _ (Quotient.out_eq' _), orbitInc_orbitMap hinc]
  congr 1
  have := sum_fiber_inc_eq_of_free hinc hI hJ i₀ p.out
  have hp : orbitMap G J p.out = p := Quotient.out_eq' p
  rw [hp] at this
  exact this

variable [Fintype G]

omit [MulAction G I] [MulAction G J] [Fintype I] [Fintype J] in
/-- The elements of `G` sending `a` to `h a` form a coset of the stabilizer. -/
theorem card_filter_smul_eq_natCard_stabilizer (X : Type*) [MulAction G X] (a : X) (h : G) :
    #(univ.filter (fun g : G ↦ g • a = h • a)) = Nat.card (stabilizer G a) := by
  have hst : Nat.card (stabilizer G a) = #(univ.filter (fun g : G ↦ g • a = a)) :=
    Nat.subtype_card _ fun g ↦ by simp [mem_stabilizer_iff]
  rw [hst]
  refine card_nbij' (fun g ↦ h⁻¹ * g) (fun g ↦ h * g) ?_ ?_ ?_ ?_
  · intro g hg
    simp only [coe_filter, mem_univ, true_and, Set.mem_setOf_eq] at hg ⊢
    rw [mul_smul, hg, inv_smul_smul]
  · intro g hg
    simp only [coe_filter, mem_univ, true_and, Set.mem_setOf_eq] at hg ⊢
    rw [mul_smul, hg]
  · intro g _
    simp
  · intro g _
    simp

omit [MulAction G I] [MulAction G J] [Fintype I] [Fintype J] in
/-- A sum over `G` of a function of `g a` is `|G_a|` times the sum over the orbit of `a`. -/
theorem sum_smul_eq_natCard_stabilizer_nsmul {X M : Type*} [MulAction G X] [Fintype X]
    [AddCommMonoid M] (φ : X → M) (a : X) :
    ∑ g : G, φ (g • a) =
      Nat.card (stabilizer G a) •
        ∑ x ∈ univ.filter (fun x ↦ orbitMap G X x = orbitMap G X a), φ x := by
  rw [Finset.sum_comp φ (fun g : G ↦ g • a), Finset.smul_sum]
  have himg : univ.image (fun g : G ↦ g • a) =
      univ.filter (fun x ↦ orbitMap G X x = orbitMap G X a) := by
    ext x
    simp [orbitMap_eq_iff]
  rw [himg]
  refine sum_congr rfl fun x hx ↦ ?_
  obtain ⟨h, rfl⟩ := orbitMap_eq_iff.mp (mem_filter.mp hx).2
  rw [card_filter_smul_eq_natCard_stabilizer X a h]

/-- `|G_i| ∑_{i' ∈ G i} [i' : j] = |G_j| ∑_{j' ∈ G j} [i : j']` for equivariant incidences. -/
theorem natCard_stabilizer_mul_sum_eq (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (i₀ : I) (j₀ : J) :
    (Nat.card (stabilizer G i₀) : k) *
        ∑ i ∈ univ.filter (fun i ↦ orbitMap G I i = orbitMap G I i₀), inc i j₀ =
      (Nat.card (stabilizer G j₀) : k) *
        ∑ j ∈ univ.filter (fun j ↦ orbitMap G J j = orbitMap G J j₀), inc i₀ j := by
  rw [← nsmul_eq_mul, ← nsmul_eq_mul, ← sum_smul_eq_natCard_stabilizer_nsmul (fun i ↦ inc i j₀) i₀,
    ← sum_smul_eq_natCard_stabilizer_nsmul (fun j ↦ inc i₀ j) j₀]
  exact Fintype.sum_equiv (Equiv.inv G) _ _ fun g ↦ by
    simpa using (hinc g⁻¹ (g • i₀) j₀).symm

omit [Fintype G] [Fintype I] [Fintype J] in
/-- `orbitFunEquiv` rescaled by the inverse stabilizer orders. -/
noncomputable def scaledOrbitEquiv {X : Type*} [MulAction G X]
    (h : ∀ x : X, IsUnit (Nat.card (stabilizer G x) : k)) :
    invariantChains G k X ≃ₗ[k] (orbitRel.Quotient G X → k) :=
  orbitFunEquiv.trans (LinearEquiv.piCongrRight fun q ↦ LinearEquiv.smulOfUnit (h q.out).unit⁻¹)

omit [Fintype G] [Fintype I] [Fintype J] in
theorem scaledOrbitEquiv_apply {X : Type*} [MulAction G X]
    (h : ∀ x : X, IsUnit (Nat.card (stabilizer G x) : k)) (c : invariantChains G k X)
    (q : orbitRel.Quotient G X) :
    scaledOrbitEquiv h c q = ((h q.out).unit⁻¹ : kˣ) * (c : X → k) q.out := by
  simp only [scaledOrbitEquiv, LinearEquiv.trans_apply, LinearEquiv.piCongrRight_apply]
  rfl

/-- If every stabilizer order is a unit in `k`, `scaledOrbitEquiv` is an isomorphism of chain
complexes from the invariant chains onto the orbit complex. Over a field of characteristic `p`
this asks that no stabilizer have order divisible by `p`, for example that no cell of `N \ L`
lie on the mirror of a reflection when `p = 2`. A weakening of the hypothesis `char F ∤ |G|`,
complementary to the orbit condition of `transferEquiv`. -/
theorem scaledOrbitEquiv_bdInvariant (hinc : ∀ (g : G) i j, inc (g • i) (g • j) = inc i j)
    (hI : ∀ i : I, IsUnit (Nat.card (stabilizer G i) : k))
    (hJ : ∀ j : J, IsUnit (Nat.card (stabilizer G j) : k)) (c : invariantChains G k I) :
    scaledOrbitEquiv hJ (bdInvariant hinc c) = bd (orbitInc G inc) (scaledOrbitEquiv hI c) := by
  funext p
  rw [scaledOrbitEquiv_apply, coe_bdInvariant_apply, bd_apply, bd_apply,
    ← sum_fiberwise univ (orbitMap G I) (fun i ↦ (c : I → k) i * inc i p.out), mul_sum]
  refine sum_congr rfl fun q _ ↦ ?_
  have hc : ∀ i, orbitMap G I i = q → (c : I → k) i = (c : I → k) q.out := by
    intro i hi
    have hq : orbitMap G I i = orbitMap G I q.out := hi.trans (Quotient.out_eq' q).symm
    obtain ⟨g, hg⟩ := orbitMap_eq_iff.mp hq
    rw [← hg]
    exact mem_invariantChains.mp c.2 g _
  rw [sum_congr rfl fun i hi ↦ by rw [hc i (mem_filter.mp hi).2], ← mul_sum,
    scaledOrbitEquiv_apply]
  have hq : orbitMap G I q.out = q := Quotient.out_eq' q
  have hp : orbitMap G J p.out = p := Quotient.out_eq' p
  have key := natCard_stabilizer_mul_sum_eq hinc q.out p.out
  rw [hq, hp] at key
  have hA : (((hJ p.out).unit⁻¹ : kˣ) : k) *
      ∑ i ∈ univ.filter (fun i ↦ orbitMap G I i = q), inc i p.out =
      (((hI q.out).unit⁻¹ : kˣ) : k) * orbitInc G inc q p := by
    rw [Units.inv_mul_eq_iff_eq_mul, ← mul_assoc, mul_comm _ (((hI q.out).unit⁻¹ : kˣ) : k),
      mul_assoc, Units.eq_inv_mul_iff_mul_eq]
    simpa [orbitInc] using key
  linear_combination ((c : I → k) q.out) * hA

end Stabilizers

section Counterexample

-- The permutations of two cells.
local notation "S₂" => Equiv.Perm (Fin 2)

/-- The permutations of two cells act transitively, so every orbit has two cells. -/
theorem ncard_orbit_perm_two (i : Fin 2) : (orbit S₂ i).ncard = 2 := by
  rw [orbit_eq_univ, Set.ncard_univ, Nat.card_eq_fintype_card, Fintype.card_fin]

/-- The transfer fails without invertibility: for the swap of two cells over `ZMod 2` the orbit
has `2 = 0` cells, and `π_#` kills the invariant chain `e₀ + e₁`, so it is not injective on
invariant chains. -/
theorem not_injective_push_perm_two :
    ¬ Function.Injective ((push S₂).domRestrict (invariantChains S₂ (ZMod 2) (Fin 2))) :=
  not_injective_push_of_ncard_orbit_eq_zero (i := 0) (by
    rw [ncard_orbit_perm_two]
    exact ZMod.natCast_self 2)

/-- Incidence numbers over `ZMod 2` of `[-1, 1]` relative to `{-1, 1}` with the reflection
`x ↦ -x`. The cells `Fin 2` are the edges `[0, 1]` and `[-1, 0]`, oriented away from `0`, which
the reflection swaps; their only face outside `{-1, 1}` is the vertex `0`, the cell `()`, with
incidence number `-1`. A counterexample to the chain-level reconstruction in characteristic
`2`. -/
def reflectionInc : Fin 2 → Unit → ZMod 2 := -1

/-- The incidence numbers of the reflected interval are equivariant. -/
theorem reflectionInc_equivariant (g : S₂) (i : Fin 2) (j : Unit) :
    reflectionInc (g • i) (g • j) = reflectionInc i j :=
  rfl

/-- The orbit complex of the reflected interval is `[0, 1]` relative to `{1}`, with incidence
number `-1` between the edge and the vertex `0`. -/
theorem orbitInc_reflectionInc (q : orbitRel.Quotient S₂ (Fin 2))
    (p : orbitRel.Quotient S₂ Unit) : orbitInc S₂ reflectionInc q p = -1 := by
  obtain ⟨⟨⟩, rfl⟩ := orbitMap_surjective p
  simp [orbitInc, reflectionInc]

/-- Without invertibility the reconstruction can fail for every choice of isomorphisms, not only
for `π_#`: for the reflection of `[-1, 1]` relative to `{-1, 1}` over `ZMod 2`, which satisfies
the grid condition, the boundary of the invariant chains vanishes, while the boundary of the orbit
complex, `[0, 1]` relative to `{1}`, does not. So no isomorphisms from the invariant chains onto
the orbit chains intertwine the boundaries. It need not fail: see `exists_chainEquiv_circle`. -/
theorem not_exists_chainEquiv_reflection :
    ¬ ∃ (e₁ : invariantChains S₂ (ZMod 2) (Fin 2) ≃ₗ[ZMod 2]
          (orbitRel.Quotient S₂ (Fin 2) → ZMod 2))
        (e₀ : invariantChains S₂ (ZMod 2) Unit ≃ₗ[ZMod 2] (orbitRel.Quotient S₂ Unit → ZMod 2)),
        ∀ c, e₀ (bdInvariant reflectionInc_equivariant c) =
          bd (orbitInc S₂ reflectionInc) (e₁ c) := by
  rintro ⟨e₁, e₀, he⟩
  have hzero : ∀ c, bdInvariant reflectionInc_equivariant c = 0 := by
    intro c
    have h10 : (c : Fin 2 → ZMod 2) 1 = (c : Fin 2 → ZMod 2) 0 := by
      simpa [Equiv.Perm.smul_def] using mem_invariantChains.mp c.2 (Equiv.swap 0 1) 0
    apply Subtype.ext
    funext j
    simp [reflectionInc, Fin.sum_univ_two, h10, CharTwo.neg_eq, CharTwo.add_self_eq_zero]
  have hd : bd (orbitInc S₂ reflectionInc) (Pi.single (orbitMap S₂ (Fin 2) 0) 1) ≠ 0 := by
    intro h0
    have h0' := congr_fun h0 (orbitMap S₂ Unit ())
    simp [orbitInc_reflectionInc] at h0'
  have he' := he (e₁.symm (Pi.single (orbitMap S₂ (Fin 2) 0) 1))
  rw [hzero, map_zero, LinearEquiv.apply_symm_apply] at he'
  exact hd he'.symm

/-- The antipodal map of a circle with two vertices and two edges, `∂eᵢ = v₁₋ᵢ - vᵢ`, over
`ZMod 2`: the permutations of two cells act freely on edges and on vertices. -/
def circleInc : Fin 2 → Fin 2 → ZMod 2 := fun i j ↦ if i = j then -1 else 1

theorem circleInc_equivariant (g : S₂) (i j : Fin 2) :
    circleInc (g • i) (g • j) = circleInc i j := by
  simp [circleInc, Equiv.Perm.smul_def]

/-- The permutations of two cells act freely. -/
theorem perm_two_eq_one_of_smul_eq (g : S₂) (i : Fin 2) (h : g • i = i) : g = 1 := by
  revert g i
  decide

/-- The reconstruction can hold without the orbit condition: the orbits of the antipodal circle
have `2 = 0` cells in `ZMod 2`, so `π_#` is not injective on invariant chains, but the action is
free and `orbitFunEquiv` intertwines the boundaries. Compare `not_exists_chainEquiv_reflection`. -/
theorem exists_chainEquiv_circle :
    ∃ (e₁ : invariantChains S₂ (ZMod 2) (Fin 2) ≃ₗ[ZMod 2]
          (orbitRel.Quotient S₂ (Fin 2) → ZMod 2))
        (e₀ : invariantChains S₂ (ZMod 2) (Fin 2) ≃ₗ[ZMod 2]
          (orbitRel.Quotient S₂ (Fin 2) → ZMod 2)),
        ∀ c, e₀ (bdInvariant circleInc_equivariant c) =
          bd (orbitInc S₂ circleInc) (e₁ c) :=
  ⟨orbitFunEquiv, orbitFunEquiv, orbitFunEquiv_bdInvariant circleInc_equivariant
    perm_two_eq_one_of_smul_eq perm_two_eq_one_of_smul_eq⟩

end Counterexample

end LatentEquivariant
