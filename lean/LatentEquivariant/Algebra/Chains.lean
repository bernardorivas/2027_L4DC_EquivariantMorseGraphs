import Mathlib

/-!
# Cellular chains with prescribed incidence numbers

A finite set of cells `I` in one dimension and `J` in the next lower one, with incidence numbers
`inc i j = [i : j]` in a commutative ring `k`. Chains are functions on cells, and the boundary is
`(∂c)(j) = ∑ᵢ c(i) [i : j]`. The files `OrbitChains`, `SignChains` and `Homology/Basic` use this
model for the cellular chains of a grid on which a group acts.

In those files the group permutes cells without signs and preserves the incidence numbers. Under
the grid condition of the paper (an element that maps a cell to itself fixes it pointwise), cells
can be oriented along orbits so that this holds. The converse fails: the unsigned model only says
that an element mapping a cell to itself preserves its orientation. For example, the 3-cycle of
coordinates maps `[0,1]³` to itself, preserving orientation, and fixes only the diagonal. The
other consequence of the grid condition, that the faces of a fixed cell are fixed, is assumed
separately where it is used. Identifying the fixed cells with the cells of `N^G`, or the orbit
chains with the chains of `N/G`, needs the grid condition itself, which is not formalized.
-/

namespace LatentEquivariant

variable {k I J : Type*} [CommRing k] [Fintype I]

/-- The boundary `(∂c)(j) = ∑ᵢ c(i) [i : j]` for the incidence numbers `inc i j = [i : j]`. -/
def bd (inc : I → J → k) : (I → k) →ₗ[k] (J → k) where
  toFun c j := ∑ i, c i * inc i j
  map_add' c d := by
    ext j
    simp [add_mul, Finset.sum_add_distrib]
  map_smul' a c := by
    ext j
    simp [Finset.mul_sum, mul_assoc]

@[simp] theorem bd_apply (inc : I → J → k) (c : I → k) (j : J) :
    bd inc c j = ∑ i, c i * inc i j :=
  rfl

end LatentEquivariant
