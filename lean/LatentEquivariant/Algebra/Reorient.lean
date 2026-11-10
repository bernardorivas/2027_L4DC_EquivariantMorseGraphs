import LatentEquivariant.Algebra.Chains

/-!
# Reorienting cells: from signed to unsigned cellular actions

The files `OrbitChains` and `SignChains`, and `ChainMap.ofInvolutive` with its eigenspace lemmas in
`Homology/Basic`, assume that `G` permutes cells without signs and preserves the incidence numbers,
`[g i : g j] = [i : j]`. (The splitting `homologyEquivEigen` of `Homology/Basic` holds for any
involutive chain map, so it covers signed actions as well.) CMGDB's cubical complexes carry the
standard orientations, on which a group acts with signs: the reflection `x ↦ -x` maps the edge
`[0, 1]`, oriented from `0` to `1`, onto `[-1, 0]` traversed from `0` to `-1`, so on chains it
sends `[0, 1]` to `-[-1, 0]`. This file closes the gap between the two models in two consecutive
dimensions, with the cells `I` and `J` and the incidence numbers `inc i j = [i : j]` of
`Chains.lean`. It concerns the grid condition of Section `sec:equivariant-outer` and the chains
`e ± ge` in the proof of Proposition `prop:z2_reconstruction`.

## Model

`G` acts on `I` and `J`, and the orientation signs `ε : G → I → kˣ`, `ε' : G → J → kˣ` say that
`g` maps the oriented cell `i` to `ε(g, i)` times the oriented cell `g i`. The signed action on
chains (`signedAct`) is an action if and only if the signs satisfy the cocycle identity
`ε(g h, i) = ε(g, h i) ε(h, i)` (`IsSignCocycle`, `isSignCocycle_iff_signedAct_mul`), and it
commutes with the boundary if and only if `ε(g, i) [g i : g j] = [i : j] ε'(g, j)`
(`bd_comp_signedAct_iff`).

## Main statements

* `exists_reorient`: suppose that stabilizers preserve orientations, `g i = i → ε(g, i) = 1`
  and likewise on `J`. This is the orientation part of the grid condition. Then the orientations
  `θ`, `θ'` chosen on orbit representatives and transported by the signs
  (`transportOrientation`) satisfy `θ(g i) = ε(g, i) θ(i)`, the reoriented incidences
  `θ(i) [i : j] θ'(j)⁻¹` (`reorientInc`) are `G`-invariant, and the diagonal change of basis
  `reorient` is a chain isomorphism from `bd inc` to `bd (reorientInc θ θ' inc)` that
  intertwines the signed action with the permutation action `c ↦ c ∘ (g⁻¹ • ·)` (`permAct`).
  So the unsigned model of `OrbitChains` and `SignChains` applies to every signed cellular action
  whose stabilizers preserve orientations. Under `reorient`, the invariant chains and the `-1`
  eigenspaces of the signed action correspond to those of the permutation action
  (`signedAct_eq_self_iff`, `signedAct_eq_smul_iff`). The reoriented incidences again satisfy
  `∂ ∂ = 0` (`bd_comp_bd_reorientInc`).
* `exists_orientation_iff`, `exists_reorient_iff`, `not_exists_reorient`: conversely, a
  reorientation that makes the signed action unsigned exists only if the signs are a cocycle and
  stabilizers preserve orientations. If `g i = i` and `ε(g, i) ≠ 1`, there is none.
* `GridReflection`: the reflection of the grid of `[-1, 1]` with vertices `-1`, `0`, `1` and
  edges oriented from left to right, where `ε = -1` on the edges and `1` on the vertices. Only the
  vertex `0` is fixed, the incidences are not invariant (`GridReflection.not_gridInc_smul`), and
  orienting both edges away from `0` gives the incidences `awayInc`
  (`GridReflection.reorientInc_gridInc`), which are invariant (`GridReflection.awayInc_smul`).
  On the cells relative to `{-1, 1}`, the reoriented incidences are all `-1`
  (`GridReflection.reorientInc_gridInc_zero`, `GridReflection.reorientInc_gridInc_relative`), as
  for `reflectionInc` in `OrbitChains`.
* `det_coordCycle`, `image_coordCycle_Icc`, `exists_coordCycle_ne` support the claim that the
  stabilizer hypothesis is weaker than the grid condition. The cyclic permutation of the
  coordinates of `ℝ³` maps `[0, 1]³` onto itself with determinant `1`, so it maps the oriented
  3-cell to itself with sign `+1`, but it does not fix the cube pointwise. These are geometric
  facts about the cube; the signs of the model are not computed from them (see below).
* `exists_reorient_transportOrientation`: the witnesses of `exists_reorient` are
  `transportOrientation ε` and `transportOrientation ε'`, and each depends only on the signs of
  its own dimension, so the orientation chosen on `J` for the pair `(I, J)` is the one chosen on
  `J` for the next pair, and a whole complex can be reoriented degree by degree.

The group `G` need not be finite, `k` is any commutative ring, and only `I` needs to be finite.

## Not formalized

The signs of an actual grid, the sign of the determinant of `g` on the tangent space of a cell,
are not computed from the geometry: the model takes `ε` as data. The implication from the grid
condition to the stabilizer hypothesis (a cell fixed pointwise keeps its orientation) is not
formalized, nor is the claim that the two fixed vertices and the cube are the only cells of
`[0, 1]³` mapped to themselves by the cyclic permutation and its square. As in `OrbitChains`, the
identification of the orbit complex with the cellular complex of `N/G` needs the grid condition
itself and is not formalized.
-/

namespace LatentEquivariant

open Finset MulAction

variable {G k I J : Type*} [Group G] [CommRing k] [MulAction G I] [MulAction G J]

section Signs

/-- The cocycle identity `ε(g h, i) = ε(g, h i) ε(h, i)` for orientation signs
`ε : G → I → kˣ`: if `g` maps the oriented cell `i` to `ε(g, i)` times the oriented cell `g i`,
this says that the signs compose (`isSignCocycle_iff_signedAct_mul`). The model of signed
cellular actions. -/
def IsSignCocycle (ε : G → I → kˣ) : Prop :=
  ∀ (g h : G) (i : I), ε (g * h) i = ε g (h • i) * ε h i

/-- The identity element acts with sign `1`. -/
theorem IsSignCocycle.apply_one {ε : G → I → kˣ} (hε : IsSignCocycle ε) (i : I) :
    ε 1 i = 1 := by
  have h := hε 1 1 i
  rw [one_mul, one_smul] at h
  exact mul_eq_left.mp h.symm

/-- The signed action of `g` on chains: the basis vector of `i` goes to `ε(g, i)` times the basis
vector of `g i` (`signedAct_single`), so `(g c)(i) = ε(g, g⁻¹ i) c(g⁻¹ i)`. This is how `G` acts
on the cellular chains of a grid with its standard orientations. -/
def signedAct (ε : G → I → kˣ) (g : G) : (I → k) →ₗ[k] (I → k) where
  toFun c i := ε g (g⁻¹ • i) * c (g⁻¹ • i)
  map_add' c d := by
    funext i
    simp [mul_add]
  map_smul' a c := by
    funext i
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    ring

/-- The values of the signed action. -/
@[simp] theorem signedAct_apply (ε : G → I → kˣ) (g : G) (c : I → k) (i : I) :
    signedAct ε g c i = ε g (g⁻¹ • i) * c (g⁻¹ • i) :=
  rfl

/-- The signed action sends the chain `a i` to `ε(g, i) a (g i)`. -/
theorem signedAct_single [DecidableEq I] (ε : G → I → kˣ) (g : G) (i : I) (a : k) :
    signedAct ε g (Pi.single i a) = Pi.single (g • i) (ε g i * a) := by
  funext i'
  rw [signedAct_apply]
  by_cases h : i' = g • i
  · subst h
    simp
  · have h' : g⁻¹ • i' ≠ i := fun h'' ↦ h (by rw [← h'', smul_inv_smul])
    simp [h, h']

/-- For a cocycle the signed action is multiplicative, `(g h) c = g (h c)`. -/
theorem signedAct_mul {ε : G → I → kˣ} (hε : IsSignCocycle ε) (g h : G) :
    signedAct ε (g * h) = signedAct ε g ∘ₗ signedAct ε h := by
  refine LinearMap.ext fun c ↦ funext fun i ↦ ?_
  simp only [LinearMap.comp_apply, signedAct_apply, mul_inv_rev, mul_smul]
  rw [hε g h, smul_inv_smul, Units.val_mul]
  ring

/-- The signed action is multiplicative if and only if the signs satisfy the cocycle identity.
-/
theorem isSignCocycle_iff_signedAct_mul {ε : G → I → kˣ} :
    IsSignCocycle ε ↔ ∀ g h : G, signedAct ε (g * h) = signedAct ε g ∘ₗ signedAct ε h := by
  classical
  refine ⟨fun hε g h ↦ signedAct_mul hε g h, fun H g h i ↦ Units.ext ?_⟩
  have := congr_fun (LinearMap.congr_fun (H g h) (Pi.single i 1)) ((g * h) • i)
  simp only [LinearMap.comp_apply, signedAct_single, mul_smul, Pi.single_eq_same, mul_one] at this
  rw [this, Units.val_mul]

/-- For a cocycle the identity element acts trivially on chains. -/
theorem signedAct_one {ε : G → I → kˣ} (hε : IsSignCocycle ε) : signedAct ε 1 = LinearMap.id :=
  LinearMap.ext fun c ↦ funext fun i ↦ by simp [hε.apply_one]

/-- The unsigned permutation action `c ↦ c ∘ (g⁻¹ • ·)` on chains, the action of `OrbitChains`
and `SignChains`: the basis vector of `i` goes to the basis vector of `g i`. Its invariant
chains are the `invariantChains` of `OrbitChains`. -/
def permAct (g : G) : (I → k) →ₗ[k] (I → k) :=
  LinearMap.funLeft k k (g⁻¹ • · : I → I)

/-- The values of the permutation action. -/
@[simp] theorem permAct_apply (g : G) (c : I → k) (i : I) : permAct g c i = c (g⁻¹ • i) :=
  rfl

/-- With all signs `1` the signed action is the permutation action. -/
theorem signedAct_one_eq_permAct (g : G) :
    signedAct (fun _ _ ↦ (1 : kˣ)) g = (permAct g : (I → k) →ₗ[k] (I → k)) :=
  LinearMap.ext fun c ↦ funext fun i ↦ by simp

end Signs

section Boundary

variable [Fintype I]

/-- The boundary of the chain `a i` is `a ∂i`. -/
theorem bd_single [DecidableEq I] (inc : I → J → k) (i : I) (a : k) (j : J) :
    bd inc (Pi.single i a) j = a * inc i j := by
  simp [bd_apply, Pi.single_apply]

/-- The signed actions commute with the boundary, `∂ (g c) = g (∂ c)`, if and only if
`ε(g, i) [g i : g j] = [i : j] ε'(g, j)` for all `g`, `i` and `j`: the signed form of the
equivariance of the incidences assumed in `OrbitChains`. -/
theorem bd_comp_signedAct_iff {ε : G → I → kˣ} {ε' : G → J → kˣ} {inc : I → J → k} :
    (∀ g, bd inc ∘ₗ signedAct ε g = signedAct ε' g ∘ₗ bd inc) ↔
      ∀ g i j, ε g i * inc (g • i) (g • j) = inc i j * ε' g j := by
  classical
  constructor
  · intro H g i j
    have h := congr_fun (LinearMap.congr_fun (H g) (Pi.single i 1)) (g • j)
    simp only [LinearMap.comp_apply, signedAct_single, bd_single, signedAct_apply,
      inv_smul_smul, mul_one, one_mul] at h
    rw [h, mul_comm]
  · intro H g
    refine LinearMap.ext fun c ↦ funext fun j ↦ ?_
    simp only [LinearMap.comp_apply, bd_apply, signedAct_apply]
    rw [← Equiv.sum_comp (MulAction.toPerm g)]
    simp only [MulAction.toPerm_apply, inv_smul_smul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    have h := H g i (g⁻¹ • j)
    rw [smul_inv_smul] at h
    linear_combination c i * h

end Boundary

section Reorient

/-- The reoriented incidences `[i : j]' = θ(i) [i : j] θ'(j)⁻¹`: the incidence numbers in the
bases `θ(i) eᵢ` and `θ'(j) eⱼ`, since `∂ (θ(i) eᵢ) = ∑ⱼ θ(i) [i : j] θ'(j)⁻¹ (θ'(j) eⱼ)`. For
signs `±1` this is a change of the orientations of the cells. -/
def reorientInc (θ : I → kˣ) (θ' : J → kˣ) (inc : I → J → k) (i : I) (j : J) : k :=
  θ i * inc i j * ↑(θ' j)⁻¹

/-- The diagonal change of basis to the basis `θ(i) eᵢ`, in coordinates `c ↦ (i ↦ θ(i)⁻¹ c(i))`.
It is a chain isomorphism onto the reoriented complex (`reorient_bd`). -/
def reorient (θ : I → kˣ) : (I → k) ≃ₗ[k] (I → k) where
  toFun c i := ↑(θ i)⁻¹ * c i
  map_add' c d := by
    funext i
    simp [mul_add]
  map_smul' a c := by
    funext i
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    ring
  invFun c i := θ i * c i
  left_inv c := by
    funext i
    simp
  right_inv c := by
    funext i
    simp

/-- The coordinates in the reoriented basis. -/
@[simp] theorem reorient_apply (θ : I → kˣ) (c : I → k) (i : I) :
    reorient θ c i = ↑(θ i)⁻¹ * c i :=
  rfl

/-- The inverse change of coordinates. -/
@[simp] theorem reorient_symm_apply (θ : I → kˣ) (c : I → k) (i : I) :
    (reorient θ).symm c i = θ i * c i :=
  rfl

/-- `reorient` is a chain map: `reorient θ' (∂ c) = ∂' (reorient θ c)` for the reoriented
incidences, for all `θ` and `θ'`. -/
theorem reorient_bd [Fintype I] (θ : I → kˣ) (θ' : J → kˣ) (inc : I → J → k) (c : I → k) :
    reorient θ' (bd inc c) = bd (reorientInc θ θ' inc) (reorient θ c) := by
  funext j
  simp only [reorient_apply, bd_apply, reorientInc, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  have h : (↑(θ i)⁻¹ : k) * θ i = 1 := Units.inv_mul (θ i)
  linear_combination (-(c i * inc i j * ↑(θ' j)⁻¹)) * h

/-- `reorient_bd` as a commutative square of linear maps. -/
theorem reorient_comp_bd [Fintype I] (θ : I → kˣ) (θ' : J → kˣ) (inc : I → J → k) :
    (reorient θ').toLinearMap ∘ₗ bd inc = bd (reorientInc θ θ' inc) ∘ₗ (reorient θ).toLinearMap :=
  LinearMap.ext fun c ↦ reorient_bd θ θ' inc c

/-- Reorienting the cells of three consecutive dimensions preserves `∂ ∂ = 0`, so the reoriented
incidences of a cell complex form a cell complex. -/
theorem bd_comp_bd_reorientInc {K : Type*} [Fintype K] [Fintype I] {inc₁ : K → I → k}
    {inc₂ : I → J → k} (h : bd inc₂ ∘ₗ bd inc₁ = 0) (θ₂ : K → kˣ) (θ₁ : I → kˣ)
    (θ₀ : J → kˣ) :
    bd (reorientInc θ₁ θ₀ inc₂) ∘ₗ bd (reorientInc θ₂ θ₁ inc₁) = 0 := by
  refine LinearMap.ext fun c ↦ ?_
  have h1 := reorient_bd θ₂ θ₁ inc₁ ((reorient θ₂).symm c)
  rw [LinearEquiv.apply_symm_apply] at h1
  have h2 : bd inc₂ (bd inc₁ ((reorient θ₂).symm c)) = 0 := by
    rw [← LinearMap.comp_apply, h, LinearMap.zero_apply]
  rw [LinearMap.comp_apply, ← h1, ← reorient_bd, h2, map_zero, LinearMap.zero_apply]

/-- If `θ(g i) = ε(g, i) θ(i)`, `θ'(g j) = ε'(g, j) θ'(j)` and the signed action commutes with the
boundary, then the reoriented incidences are `G`-invariant, `[g i : g j]' = [i : j]'`: the
hypothesis `hinc` of `OrbitChains`. -/
theorem reorientInc_smul {ε : G → I → kˣ} {ε' : G → J → kˣ} {θ : I → kˣ} {θ' : J → kˣ}
    {inc : I → J → k} (hθ : ∀ g i, θ (g • i) = ε g i * θ i)
    (hθ' : ∀ g j, θ' (g • j) = ε' g j * θ' j)
    (hinc : ∀ g i j, ε g i * inc (g • i) (g • j) = inc i j * ε' g j) (g : G) (i : I) (j : J) :
    reorientInc θ θ' inc (g • i) (g • j) = reorientInc θ θ' inc i j := by
  simp only [reorientInc, hθ, hθ', mul_inv_rev, Units.val_mul]
  have h1 := hinc g i j
  have h2 : (ε' g j : k) * ↑(ε' g j)⁻¹ = 1 := Units.mul_inv _
  linear_combination (↑(θ i) * ↑(θ' j)⁻¹ * ↑(ε' g j)⁻¹) * h1 +
    (↑(θ i) * inc i j * ↑(θ' j)⁻¹) * h2

/-- In the coordinates `reorient θ`, the signed action of `g` is the permutation action if and
only if `θ(g i) = ε(g, i) θ(i)` for every cell `i`, that is, `g` maps the basis vector `θ(i) eᵢ`
to the basis vector `θ(g i) e_{g i}`. -/
theorem reorient_signedAct_iff {ε : G → I → kˣ} {θ : I → kˣ} (g : G) :
    (∀ c, reorient θ (signedAct ε g c) = permAct g (reorient θ c)) ↔
      ∀ i, θ (g • i) = ε g i * θ i := by
  classical
  constructor
  · intro H i
    have h := congr_fun (H (Pi.single i 1)) (g • i)
    simp only [reorient_apply, signedAct_single, permAct_apply, inv_smul_smul,
      Pi.single_eq_same, mul_one] at h
    have h' : (θ (g • i))⁻¹ * ε g i = (θ i)⁻¹ := Units.ext (by rw [Units.val_mul]; exact h)
    rw [inv_mul_eq_iff_eq_mul] at h'
    rw [h', inv_mul_cancel_right]
  · intro H c
    funext i
    simp only [reorient_apply, signedAct_apply, permAct_apply]
    have h := H (g⁻¹ • i)
    rw [smul_inv_smul] at h
    have h2 := Units.inv_mul (ε g (g⁻¹ • i))
    rw [h, mul_inv_rev, Units.val_mul]
    linear_combination (↑(θ (g⁻¹ • i))⁻¹ * c (g⁻¹ • i)) * h2

/-- Eigenchains correspond: if `θ(g i) = ε(g, i) θ(i)`, then `g c = a c` for the signed action
if and only if `d = reorient θ c` satisfies `d(g⁻¹ i) = a d(i)`. For an involution `g` and
`a = -1` this is membership of `d` in `antiChains (g • ·)` of `SignChains`, the `-1` eigenspace of
Proposition `prop:z2_reconstruction`. -/
theorem signedAct_eq_smul_iff {ε : G → I → kˣ} {θ : I → kˣ} {g : G}
    (hθ : ∀ i, θ (g • i) = ε g i * θ i) (a : k) (c : I → k) :
    signedAct ε g c = a • c ↔ ∀ i, reorient θ c (g⁻¹ • i) = a * reorient θ c i := by
  rw [← (reorient θ).injective.eq_iff, (reorient_signedAct_iff g).mpr hθ c, map_smul, funext_iff]
  simp only [permAct_apply, Pi.smul_apply, smul_eq_mul]

end Reorient

section Transport

variable (G) in
/-- The chosen representative of the orbit of `i`, `Quotient.out` of its class in `I/G`. -/
noncomputable def orbitRep (i : I) : I :=
  (Quotient.mk'' i : orbitRel.Quotient G I).out

/-- The representative depends only on the orbit. -/
theorem orbitRep_smul (g : G) (i : I) : orbitRep G (g • i) = orbitRep G i :=
  congrArg Quotient.out (Quotient.sound' (orbitRel_apply.mpr (mem_orbit i g)))

/-- Every cell is the image of its representative. -/
theorem exists_smul_orbitRep (i : I) : ∃ g : G, g • orbitRep G i = i := by
  have h : orbitRep G i ∈ orbit G i :=
    orbitRel_apply.mp (Quotient.mk_out' (s₁ := orbitRel G I) i)
  obtain ⟨g, hg⟩ := mem_orbit_iff.mp h
  exact ⟨g⁻¹, by rw [← hg, inv_smul_smul]⟩

/-- A representative is its own representative. -/
theorem orbitRep_orbitRep (i : I) : orbitRep G (orbitRep G i) = orbitRep G i := by
  obtain ⟨g, hg⟩ := exists_smul_orbitRep (G := G) i
  calc orbitRep G (orbitRep G i) = orbitRep G (g • orbitRep G i) := (orbitRep_smul g _).symm
    _ = orbitRep G i := by rw [hg]

/-- The orientation transported from the orbit representatives: `θ(i) = ε(g, r)`, where `r` is
the representative of the orbit of `i` and `g r = i`. The representatives keep their
orientations (`transportOrientation_orbitRep`), and the others get the orientation of the image of
their representative. Well defined under the hypotheses of `transportOrientation_eq`. These are
the orientations along orbits that the grid condition provides, as assumed
in the model of `OrbitChains`. -/
noncomputable def transportOrientation (ε : G → I → kˣ) (i : I) : kˣ :=
  ε (exists_smul_orbitRep (G := G) i).choose (orbitRep G i)

variable {ε : G → I → kˣ}

/-- The transported orientation does not depend on the element `h` with `h r = i`, if the signs
are a cocycle and stabilizers preserve orientations. -/
theorem transportOrientation_eq (hε : IsSignCocycle ε) (hfix : ∀ g i, g • i = i → ε g i = 1)
    {i : I} {h : G} (hh : h • orbitRep G i = i) :
    transportOrientation ε i = ε h (orbitRep G i) := by
  have hg := (exists_smul_orbitRep (G := G) i).choose_spec
  set g := (exists_smul_orbitRep (G := G) i).choose
  have hfixr : (h⁻¹ * g) • orbitRep G i = orbitRep G i := by
    rw [mul_smul, hg]
    exact inv_smul_eq_iff.mpr hh.symm
  have h1 := hfix _ _ hfixr
  rw [hε h⁻¹ g, hg] at h1
  have h2 : ε (h⁻¹ * h) (orbitRep G i) = 1 := by
    rw [inv_mul_cancel]
    exact hε.apply_one _
  rw [hε h⁻¹ h, hh] at h2
  exact mul_left_cancel (h1.trans h2.symm)

/-- The transported orientation is equivariant, `θ(g i) = ε(g, i) θ(i)`, if the signs are a
cocycle and stabilizers preserve orientations. -/
theorem transportOrientation_smul (hε : IsSignCocycle ε)
    (hfix : ∀ g i, g • i = i → ε g i = 1) (g : G) (i : I) :
    transportOrientation ε (g • i) = ε g i * transportOrientation ε i := by
  have hh := (exists_smul_orbitRep (G := G) i).choose_spec
  set h := (exists_smul_orbitRep (G := G) i).choose
  have key : (g * h) • orbitRep G (g • i) = g • i := by rw [orbitRep_smul, mul_smul, hh]
  rw [transportOrientation_eq hε hfix key, orbitRep_smul, hε g h, hh]
  rfl

/-- The representatives keep their orientations: `θ(r) = 1`. -/
theorem transportOrientation_orbitRep (hε : IsSignCocycle ε)
    (hfix : ∀ g i, g • i = i → ε g i = 1) (i : I) :
    transportOrientation ε (orbitRep G i) = 1 := by
  have key : (1 : G) • orbitRep G (orbitRep G i) = orbitRep G i := by
    rw [one_smul, orbitRep_orbitRep]
  rw [transportOrientation_eq hε hfix key, orbitRep_orbitRep, hε.apply_one]

/-- For signs `±1`, as for the standard orientations of a grid, the transported orientations
are `±1`, so the reorientation reverses the orientations of some cells and keeps the others. -/
theorem transportOrientation_eq_one_or (hpm : ∀ g i, ε g i = 1 ∨ ε g i = -1) (i : I) :
    transportOrientation ε i = 1 ∨ transportOrientation ε i = -1 :=
  hpm _ _

/-- Orientations `θ` with `θ(g i) = ε(g, i) θ(i)` exist if and only if the signs are a cocycle
and stabilizers preserve orientations, `g i = i → ε(g, i) = 1`: the orientation part of the grid
condition is exactly what is needed. -/
theorem exists_orientation_iff :
    (∃ θ : I → kˣ, ∀ g i, θ (g • i) = ε g i * θ i) ↔
      IsSignCocycle ε ∧ ∀ g i, g • i = i → ε g i = 1 := by
  constructor
  · rintro ⟨θ, hθ⟩
    refine ⟨fun g h i ↦ ?_, fun g i hgi ↦ ?_⟩
    · have h1 := hθ (g * h) i
      rw [mul_smul, hθ g (h • i), hθ h i, ← mul_assoc] at h1
      exact (mul_right_cancel h1).symm
    · have h1 := hθ g i
      rw [hgi] at h1
      exact mul_eq_right.mp h1.symm
  · rintro ⟨hε, hfix⟩
    exact ⟨transportOrientation ε, transportOrientation_smul hε hfix⟩

/-- A diagonal change of basis makes the signed action unsigned, that is, intertwines it with the
permutation action, if and only if the signs are a cocycle and stabilizers preserve
orientations. -/
theorem exists_reorient_iff :
    (∃ θ : I → kˣ, ∀ (g : G) (c : I → k),
        reorient θ (signedAct ε g c) = permAct g (reorient θ c)) ↔
      IsSignCocycle ε ∧ ∀ g i, g • i = i → ε g i = 1 := by
  rw [← exists_orientation_iff]
  exact exists_congr fun θ ↦ forall_congr' fun g ↦ reorient_signedAct_iff g

/-- The converse of `exists_reorient`: if `g i = i` and `ε(g, i) ≠ 1`, then no reorientation
makes the signed action of `g` unsigned, so the unsigned model of `OrbitChains` and `SignChains`
does not apply in any orientation. -/
theorem not_exists_reorient {g : G} {i : I} (hgi : g • i = i) (hne : ε g i ≠ 1) :
    ¬ ∃ θ : I → kˣ, ∀ c : I → k, reorient θ (signedAct ε g c) = permAct g (reorient θ c) := by
  rintro ⟨θ, hθ⟩
  have h := (reorient_signedAct_iff g).mp hθ i
  rw [hgi] at h
  exact hne (mul_eq_right.mp h.symm)

/-- Invariant chains correspond: if `θ(g i) = ε(g, i) θ(i)`, a chain is invariant under the signed
action if and only if `reorient θ c` is invariant under the permutation action, that is, lies in
`invariantChains` of `OrbitChains`. -/
theorem signedAct_eq_self_iff {θ : I → kˣ} (hθ : ∀ g i, θ (g • i) = ε g i * θ i) (c : I → k) :
    (∀ g, signedAct ε g c = c) ↔ ∀ (g : G) (i : I), reorient θ c (g • i) = reorient θ c i := by
  constructor
  · intro H g i
    have h := congr_fun ((reorient_signedAct_iff g⁻¹).mpr (fun i ↦ hθ g⁻¹ i) c) i
    rw [H] at h
    simp only [permAct_apply, inv_inv] at h
    exact h.symm
  · intro H g
    apply (reorient θ).injective
    rw [(reorient_signedAct_iff g).mpr (fun i ↦ hθ g i) c]
    funext i
    rw [permAct_apply]
    exact H g⁻¹ i

end Transport

section Main

/-- **Reorientation.** Let `G` act on the cells `I` and `J` of two consecutive dimensions with
orientation signs `ε`, `ε'` that satisfy the cocycle identity, such that the signed action
commutes with the boundary, `ε(g, i) [g i : g j] = [i : j] ε'(g, j)` (`bd_comp_signedAct_iff`),
and such that stabilizers preserve orientations, `g i = i → ε(g, i) = 1` and likewise on `J`:
the orientation part of the grid condition. Then there are orientations
`θ`, `θ'` with `θ(g i) = ε(g, i) θ(i)` and `θ'(g j) = ε'(g, j) θ'(j)` (the transported
orientations), the reoriented incidences `θ(i) [i : j] θ'(j)⁻¹` are `G`-invariant, and the
diagonal changes of basis `reorient θ`, `reorient θ'` form a chain isomorphism from `bd inc` to
the reoriented boundary that intertwines the signed actions with the permutation actions
`c ↦ c ∘ (g⁻¹ • ·)`. So the unsigned model of `OrbitChains` and `SignChains`, used for the
reconstruction from the orbit space and Proposition `prop:z2_reconstruction`, applies. The
converse is `not_exists_reorient`. -/
theorem exists_reorient [Fintype I] {ε : G → I → kˣ} {ε' : G → J → kˣ} {inc : I → J → k}
    (hε : IsSignCocycle ε) (hε' : IsSignCocycle ε')
    (hfix : ∀ g i, g • i = i → ε g i = 1) (hfix' : ∀ g j, g • j = j → ε' g j = 1)
    (hinc : ∀ g i j, ε g i * inc (g • i) (g • j) = inc i j * ε' g j) :
    ∃ (θ : I → kˣ) (θ' : J → kˣ),
      (∀ (g : G) i, θ (g • i) = ε g i * θ i) ∧ (∀ (g : G) j, θ' (g • j) = ε' g j * θ' j) ∧
      (∀ (g : G) i j, reorientInc θ θ' inc (g • i) (g • j) = reorientInc θ θ' inc i j) ∧
      (∀ c, reorient θ' (bd inc c) = bd (reorientInc θ θ' inc) (reorient θ c)) ∧
      (∀ (g : G) c, reorient θ (signedAct ε g c) = permAct g (reorient θ c)) ∧
      (∀ (g : G) c, reorient θ' (signedAct ε' g c) = permAct g (reorient θ' c)) :=
  ⟨transportOrientation ε, transportOrientation ε', transportOrientation_smul hε hfix,
    transportOrientation_smul hε' hfix',
    reorientInc_smul (transportOrientation_smul hε hfix) (transportOrientation_smul hε' hfix')
      hinc,
    reorient_bd _ _ inc,
    fun g ↦ (reorient_signedAct_iff g).mpr fun i ↦ transportOrientation_smul hε hfix g i,
    fun g ↦ (reorient_signedAct_iff g).mpr fun j ↦ transportOrientation_smul hε' hfix' g j⟩

/-- `exists_reorient` with the witnesses named: the orientations are `transportOrientation ε` on
`I` and `transportOrientation ε'` on `J`, and each depends only on the signs of its own
dimension. So for consecutive pairs of dimensions `(I, J)` and `(J, K)` the orientation of `J` is
the same, and the reoriented incidences of all degrees fit into one complex (with
`bd_comp_bd_reorientInc` for `∂ ∂ = 0`). -/
theorem exists_reorient_transportOrientation [Fintype I] {ε : G → I → kˣ} {ε' : G → J → kˣ}
    {inc : I → J → k} (hε : IsSignCocycle ε) (hε' : IsSignCocycle ε')
    (hfix : ∀ g i, g • i = i → ε g i = 1) (hfix' : ∀ g j, g • j = j → ε' g j = 1)
    (hinc : ∀ g i j, ε g i * inc (g • i) (g • j) = inc i j * ε' g j) :
    (∀ (g : G) i j, reorientInc (transportOrientation ε) (transportOrientation ε') inc (g • i)
        (g • j) = reorientInc (transportOrientation ε) (transportOrientation ε') inc i j) ∧
      (∀ c, reorient (transportOrientation ε') (bd inc c) =
        bd (reorientInc (transportOrientation ε) (transportOrientation ε') inc)
          (reorient (transportOrientation ε) c)) ∧
      (∀ (g : G) c, reorient (transportOrientation ε) (signedAct ε g c) =
        permAct g (reorient (transportOrientation ε) c)) :=
  ⟨reorientInc_smul (transportOrientation_smul hε hfix) (transportOrientation_smul hε' hfix') hinc,
    reorient_bd _ _ inc,
    fun g ↦ (reorient_signedAct_iff g).mpr fun i ↦ transportOrientation_smul hε hfix g i⟩

end Main

namespace GridReflection

/-! ### The reflection of the grid of `[-1, 1]`

The grid of `[-1, 1]` has the vertices `-1`, `0`, `1` and the edges `[-1, 0]`, `[0, 1]`, oriented
from left to right. The group `ℤˣ = {1, -1}` acts by `x ↦ u x`. The reflection `-1` swaps the
edges and reverses their orientations, `ε = -1` on the edges, and fixes only the vertex `0`, where
`ε = 1`. -/

/-- The vertices `-1`, `0` and `1` of the grid of `[-1, 1]`. -/
inductive Vertex
  | neg
  | zero
  | pos
  deriving DecidableEq, Fintype

/-- The edges `[-1, 0]` (`left`) and `[0, 1]` (`right`) of the grid of `[-1, 1]`. -/
inductive Edge
  | left
  | right
  deriving DecidableEq, Fintype

/-- The reflection `x ↦ -x` on the vertices: it swaps `-1` and `1` and fixes `0`. -/
def Vertex.reflect : Vertex → Vertex
  | .neg => .pos
  | .zero => .zero
  | .pos => .neg

/-- The reflection `x ↦ -x` on the edges: it swaps `[-1, 0]` and `[0, 1]`. -/
def Edge.reflect : Edge → Edge
  | .left => .right
  | .right => .left

/-- `ℤˣ = {1, -1}` acts on the vertices, `-1` by the reflection `x ↦ -x`. -/
instance : MulAction ℤˣ Vertex where
  smul u v := if (u : ℤ) = 1 then v else v.reflect
  one_smul _ := if_pos rfl
  mul_smul u w v := by
    rcases Int.units_eq_one_or u with rfl | rfl <;> rcases Int.units_eq_one_or w with rfl | rfl <;>
      cases v <;> rfl

/-- `ℤˣ = {1, -1}` acts on the edges, `-1` by the reflection `x ↦ -x`. -/
instance : MulAction ℤˣ Edge where
  smul u e := if (u : ℤ) = 1 then e else e.reflect
  one_smul _ := if_pos rfl
  mul_smul u w e := by
    rcases Int.units_eq_one_or u with rfl | rfl <;> rcases Int.units_eq_one_or w with rfl | rfl <;>
      cases e <;> rfl

/-- `-1` acts on the vertices by the reflection. -/
theorem neg_one_smul_vertex (v : Vertex) : (-1 : ℤˣ) • v = v.reflect :=
  if_neg (by decide)

/-- `-1` acts on the edges by the reflection. -/
theorem neg_one_smul_edge (e : Edge) : (-1 : ℤˣ) • e = e.reflect :=
  if_neg (by decide)

/-- The reflection fixes only the vertex `0`. -/
theorem neg_one_smul_vertex_eq_iff (v : Vertex) : (-1 : ℤˣ) • v = v ↔ v = .zero := by
  rw [neg_one_smul_vertex]
  cases v <;> decide

/-- The reflection fixes no edge. -/
theorem neg_one_smul_edge_ne (e : Edge) : (-1 : ℤˣ) • e ≠ e := by
  rw [neg_one_smul_edge]
  cases e <;> decide

variable (k) in
/-- The signs on the edges oriented from left to right: `u ∈ {1, -1}` maps the oriented edge `e`
to `u` times the oriented edge `u e`, since `x ↦ -x` reverses orientations. -/
def edgeSign (u : ℤˣ) (_ : Edge) : kˣ :=
  Units.map (Int.castRingHom k : ℤ →* k) u

variable (k) in
/-- The signs on the vertices: vertices keep their orientation. -/
def vertexSign (_ : ℤˣ) (_ : Vertex) : kˣ :=
  1

/-- The edge sign of `u ∈ {1, -1}` is `u`. -/
@[simp] theorem coe_edgeSign (u : ℤˣ) (e : Edge) : (edgeSign k u e : k) = ((u : ℤ) : k) := by
  simp [edgeSign]

/-- The edge signs satisfy the cocycle identity. -/
theorem isSignCocycle_edgeSign : IsSignCocycle (edgeSign k) := fun u w _ ↦ by
  simp [edgeSign]

/-- The vertex signs satisfy the cocycle identity. -/
theorem isSignCocycle_vertexSign : IsSignCocycle (vertexSign k) := fun _ _ _ ↦ by
  simp [vertexSign]

/-- The stabilizer hypothesis on the edges: an element that maps an edge to itself is `1`, which
acts with sign `1`. On the vertices it holds since all vertex signs are `1`. -/
theorem edgeSign_eq_one_of_smul_eq {u : ℤˣ} {e : Edge} (h : u • e = e) :
    edgeSign k u e = 1 := by
  rcases Int.units_eq_one_or u with rfl | rfl
  · simp [edgeSign]
  · exact absurd h (neg_one_smul_edge_ne e)

/-- The incidences of the grid with the edges oriented from left to right:
`∂[-1, 0] = [0] - [-1]` and `∂[0, 1] = [1] - [0]`. -/
def gridInc : Edge → Vertex → k
  | .left, .neg => -1
  | .left, .zero => 1
  | .right, .zero => -1
  | .right, .pos => 1
  | _, _ => 0

/-- The signed reflection commutes with the boundary: `ε(u, e) [u e : u v] = [e : v] ε'(u, v)`,
the hypothesis of `exists_reorient` (see `bd_comp_signedAct_iff`). -/
theorem gridInc_compat (u : ℤˣ) (e : Edge) (v : Vertex) :
    (edgeSign k u e : k) * gridInc (u • e) (u • v) = gridInc e v * (vertexSign k u v : k) := by
  rcases Int.units_eq_one_or u with rfl | rfl
  · simp [edgeSign, vertexSign]
  · rw [neg_one_smul_edge, neg_one_smul_vertex]
    cases e <;> cases v <;> simp [vertexSign, gridInc, Edge.reflect, Vertex.reflect]

/-- Without reorienting, the incidences are not invariant when `2 ≠ 0` in `k`:
`[-[-1, 0] : -(-1)] = [[0, 1] : 1] = 1`, while `[[-1, 0] : -1] = -1`. So the unsigned model of
`OrbitChains` does not apply to the standard orientations. -/
theorem not_gridInc_smul (h2 : (2 : k) ≠ 0) :
    gridInc ((-1 : ℤˣ) • Edge.left) ((-1 : ℤˣ) • Vertex.neg) ≠ (gridInc .left .neg : k) := by
  rw [neg_one_smul_edge, neg_one_smul_vertex]
  intro h
  apply h2
  simp only [Edge.reflect, Vertex.reflect, gridInc] at h
  linear_combination h

/-- The reflection sends the oriented edge `[0, 1]` to `-[-1, 0]`: it maps `[0, 1]` onto
`[-1, 0]`, reversing the orientation. -/
theorem signedAct_edgeSign_single_right :
    signedAct (edgeSign k) (-1) (Pi.single Edge.right 1) = Pi.single Edge.left (-1) := by
  rw [signedAct_single, neg_one_smul_edge]
  simp [Edge.reflect]

variable (k) in
/-- The orientations of the edges away from `0`: `[-1, 0]` is reversed, `[0, 1]` is kept. They
are the orientations transported from the representative `[0, 1]`. -/
def edgeOrientation : Edge → kˣ
  | .left => -1
  | .right => 1

variable (k) in
/-- The vertices keep their orientations. -/
def vertexOrientation : Vertex → kˣ := fun _ ↦ 1

/-- The edge orientations are transported by the signs, `θ(u e) = ε(u, e) θ(e)`. -/
theorem edgeOrientation_smul (u : ℤˣ) (e : Edge) :
    edgeOrientation k (u • e) = edgeSign k u e * edgeOrientation k e := by
  rcases Int.units_eq_one_or u with rfl | rfl
  · simp [edgeSign]
  · rw [neg_one_smul_edge]
    ext
    cases e <;> simp [edgeOrientation, Edge.reflect]

/-- The vertex orientations are transported by the signs. -/
theorem vertexOrientation_smul (u : ℤˣ) (v : Vertex) :
    vertexOrientation k (u • v) = vertexSign k u v * vertexOrientation k v := by
  simp [vertexSign, vertexOrientation]

/-- The incidences with both edges oriented away from `0`: `∂[-1, 0]' = [-1] - [0]` and
`∂[0, 1]' = [1] - [0]`. -/
def awayInc : Edge → Vertex → k
  | .left, .neg => 1
  | .left, .zero => -1
  | .right, .zero => -1
  | .right, .pos => 1
  | _, _ => 0

/-- The reoriented incidences of the grid are `awayInc`: reorienting the edge `[-1, 0]` orients
both edges away from `0`. -/
theorem reorientInc_gridInc :
    reorientInc (edgeOrientation k) (vertexOrientation k) gridInc =
      (awayInc : Edge → Vertex → k) := by
  funext e v
  cases e <;> cases v <;> simp [reorientInc, edgeOrientation, vertexOrientation, gridInc, awayInc]

/-- On the cells relative to `{-1, 1}`, the two edges and the vertex `0`, the reoriented
incidences are all `-1`: these are the incidences `reflectionInc` of `OrbitChains`, which are
stated over `ZMod 2`. -/
theorem reorientInc_gridInc_zero (e : Edge) :
    reorientInc (edgeOrientation k) (vertexOrientation k) gridInc e .zero = -1 := by
  rw [reorientInc_gridInc]
  cases e <;> rfl

/-- The reoriented incidences between the relative cells, with the vertex `0` as the single cell
of `Unit`, are the constant `-1`: the definition `reflectionInc := -1` of `OrbitChains`, there
with `Fin 2` for the edges and over `ZMod 2`. -/
theorem reorientInc_gridInc_relative :
    (fun (e : Edge) (_ : Unit) ↦
      reorientInc (edgeOrientation k) (vertexOrientation k) gridInc e .zero) = -1 := by
  funext e _
  rw [reorientInc_gridInc_zero]
  rfl

/-- The reoriented incidences are invariant, `[u e : u v]' = [e : v]'`, by `reorientInc_smul`. -/
theorem awayInc_smul (u : ℤˣ) (e : Edge) (v : Vertex) :
    (awayInc (u • e) (u • v) : k) = awayInc e v := by
  rw [← reorientInc_gridInc]
  exact reorientInc_smul edgeOrientation_smul vertexOrientation_smul gridInc_compat u e v

/-- In the reoriented basis the signed action of the reflection on the edge chains is the
permutation action. -/
theorem reorient_signedAct_grid (u : ℤˣ) (c : Edge → k) :
    reorient (edgeOrientation k) (signedAct (edgeSign k) u c) =
      permAct u (reorient (edgeOrientation k) c) :=
  (reorient_signedAct_iff u).mpr (edgeOrientation_smul u) c

/-- The change of basis is a chain isomorphism from the grid with edges oriented from left to
right onto the grid with edges oriented away from `0`. -/
theorem reorient_bd_grid (c : Edge → k) :
    reorient (vertexOrientation k) (bd gridInc c) =
      bd awayInc (reorient (edgeOrientation k) c) := by
  rw [reorient_bd, reorientInc_gridInc]

end GridReflection

section CoordCycle

/-! ### The stabilizer hypothesis is weaker than the grid condition -/

/-- The cyclic permutation of coordinates `(x, y, z) ↦ (y, z, x)` of `ℝ³`, the permutation matrix
of `finRotate 3`. -/
noncomputable def coordCycle : (Fin 3 → ℝ) →ₗ[ℝ] (Fin 3 → ℝ) :=
  Matrix.toLin' ((finRotate 3).permMatrix ℝ)

/-- `coordCycle x = x ∘ finRotate 3`, that is, `(x₀, x₁, x₂) ↦ (x₁, x₂, x₀)`. -/
theorem coordCycle_apply (x : Fin 3 → ℝ) : coordCycle x = x ∘ finRotate 3 := by
  rw [coordCycle, Matrix.toLin'_apply, Matrix.permMatrix_mulVec]

/-- The cyclic permutation of coordinates preserves orientation: its determinant is the sign of
a 3-cycle, `1`. So it maps the oriented cell `[0, 1]³` to itself with sign `+1`, and the
stabilizer hypothesis of `exists_reorient` holds for this cell. -/
theorem det_coordCycle : LinearMap.det coordCycle = 1 := by
  rw [coordCycle, LinearMap.det_toLin', Matrix.det_permutation, sign_finRotate]
  norm_num

/-- The cyclic permutation of coordinates maps the cell `[0, 1]³` onto itself. -/
theorem image_coordCycle_Icc : coordCycle '' Set.Icc 0 1 = Set.Icc 0 1 := by
  apply Set.Subset.antisymm
  · rintro _ ⟨x, hx, rfl⟩
    rw [coordCycle_apply]
    exact ⟨fun i ↦ hx.1 (finRotate 3 i), fun i ↦ hx.2 (finRotate 3 i)⟩
  · intro y hy
    refine ⟨y ∘ (finRotate 3).symm,
      ⟨fun i ↦ hy.1 ((finRotate 3).symm i), fun i ↦ hy.2 ((finRotate 3).symm i)⟩, ?_⟩
    rw [coordCycle_apply]
    funext i
    simp

/-- The cyclic permutation of coordinates does not fix `[0, 1]³` pointwise: it moves the vertex
`(1, 0, 0)` to `(0, 0, 1)`. So the grid condition fails for the cube,
which the cycle maps to itself with sign `+1` (`det_coordCycle`, `image_coordCycle_Icc`), and the
stabilizer hypothesis is weaker than the grid condition. -/
theorem exists_coordCycle_ne : ∃ x ∈ Set.Icc (0 : Fin 3 → ℝ) 1, coordCycle x ≠ x := by
  refine ⟨Pi.single 0 1, ⟨fun i ↦ ?_, fun i ↦ ?_⟩, fun h ↦ ?_⟩
  · by_cases hi : i = 0 <;> simp [hi]
  · by_cases hi : i = 0 <;> simp [hi]
  · have h2 := congr_fun h 2
    rw [coordCycle_apply] at h2
    simp at h2

end CoordCycle

end LatentEquivariant
