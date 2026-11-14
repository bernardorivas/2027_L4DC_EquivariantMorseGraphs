"""Binary64 image enclosures for the explicit sphere and O(3) quotient maps.

The sphere map is the time-ten map of
    z' = z (4/25 - z**2) (1 - z**2).
The O(3) map is the time-log(3) map in coordinates (a, r):
    (a, r) -> (3a/sqrt(D), 3r/sqrt(D(D + 8r**2))), D = 1 + 8a**2.

Input boxes determine the output row order. Sphere boxes have columns (z_lower, z_upper). O(3) boxes have columns (a_lower, r_lower, a_upper, r_upper).
"""

from fractions import Fraction
from functools import lru_cache
from math import inf, nextafter, sqrt

import numpy as np


@lru_cache(maxsize=1)
def exact_exp_interval():
    """Rational lower and upper bounds for exp(336/5)."""
    x = Fraction(336, 5)
    term = total = Fraction(1)
    for k in range(1, 201):
        term *= x / k
        total += term
    remainder = term * (x / 201) / (1 - x / 202)
    return total, total + remainder


def q_of_height(z):
    """The separated-flow expression Q(z) on either side of z = 2/5."""
    w = z * z
    return w**21 * (1 - w)**4 / abs(w - Fraction(4, 25))**25


@lru_cache(maxsize=None)
def exact_time_ten(height):
    """Enclose h_10(height), treating a binary64 height as an exact rational."""
    height = float(height)
    if not -1 <= height <= 1:
        raise ValueError("Sphere heights must be in [-1, 1].")
    if height < 0:
        low, high = exact_time_ten(-height)
        return -high, -low
    z = Fraction(height)
    tropic = Fraction(2, 5)
    if z in (0, tropic, 1):
        return height, height
    q = q_of_height(z)
    exp_low, exp_high = exact_exp_interval()
    lower, upper = q * exp_low, q * exp_high
    lo, hi = sorted((z, tropic))
    # The flow satisfies Q(h_10(z)) = exp(336/5) Q(z).
    for _ in range(44):
        mid = (lo + hi) / 2
        q_mid = q_of_height(mid)
        if lower <= q_mid <= upper:
            break
        if (q_mid < lower) == (z < tropic):
            lo = mid
        else:
            hi = mid
    return max(-1., float(np.nextafter(float(lo), -np.inf))), min(1., float(np.nextafter(float(hi), np.inf)))


def _boxes(values, dimension):
    boxes = np.asarray(values, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 2 * dimension:
        raise ValueError(f"Expected a two-dimensional array with {2 * dimension} columns.")
    if not np.all(np.isfinite(boxes)):
        raise ValueError("Box endpoints must be finite.")
    if np.any(boxes[:, :dimension] > boxes[:, dimension:]):
        raise ValueError("A lower endpoint exceeds its upper endpoint.")
    return boxes


def sphere_box_images(boxes):
    """Enclose h_10 on each interval, using monotonicity and endpoint bounds."""
    boxes = _boxes(boxes, 1)
    if np.any(boxes < -1) or np.any(boxes > 1):
        raise ValueError("Sphere boxes must be contained in [-1, 1].")
    images = [(exact_time_ten(lo)[0], exact_time_ten(hi)[1]) for lo, hi in boxes]
    return np.asarray(images, dtype=np.float64).reshape(-1, 2)


def square_float_exact(x):
    return Fraction.from_float(float(x)) ** 2


@lru_cache(maxsize=None)
def sqrt_enclosure(q):
    """Find neighboring binary64 bounds by exact comparisons of squares."""
    if q == 0:
        return 0.0, 0.0
    if q < 0:
        raise ValueError("The radicand must be nonnegative.")
    trial = sqrt(float(q))
    trial_square = square_float_exact(trial)
    if trial_square == q:
        return trial, trial
    if trial_square < q:
        lower = trial
        upper = nextafter(lower, inf)
        while square_float_exact(upper) < q:
            lower = upper
            upper = nextafter(upper, inf)
        if square_float_exact(upper) == q:
            return upper, upper
    else:
        upper = trial
        lower = nextafter(upper, 0.0)
        while square_float_exact(lower) > q:
            upper = lower
            lower = nextafter(lower, 0.0)
        if square_float_exact(lower) == q:
            return lower, lower
    assert square_float_exact(lower) < q < square_float_exact(upper)
    assert nextafter(lower, inf) == upper
    return lower, upper


def scalar_enclosure(a):
    magnitude = sqrt_enclosure(9 * a**2 / (1 + 8 * a**2))
    if a < 0:
        return -magnitude[1], -magnitude[0]
    return magnitude


def radial_enclosure(abs_a, r):
    d = 1 + 8 * abs_a**2
    return sqrt_enclosure(9 * r**2 / (d * (d + 8 * r**2)))


@lru_cache(maxsize=None)
def o3_box_image(rect):
    """Enclose the exact coordinate range on one O(3) quotient rectangle."""
    if len(rect) != 4:
        raise ValueError("An O(3) rectangle has four endpoints.")
    a_lo, r_lo, a_hi, r_hi = map(Fraction.from_float, map(float, rect))
    if not (-Fraction(3, 2) <= a_lo <= a_hi <= Fraction(3, 2) and 0 <= r_lo <= r_hi <= Fraction(3, 2)):
        raise ValueError("O(3) boxes must be contained in [-3/2, 3/2] x [0, 3/2].")
    abs_max = max(abs(a_lo), abs(a_hi))
    abs_min = 0 if a_lo <= 0 <= a_hi else min(abs(a_lo), abs(a_hi))
    return (
        scalar_enclosure(a_lo)[0],
        radial_enclosure(abs_max, r_lo)[0],
        scalar_enclosure(a_hi)[1],
        radial_enclosure(abs_min, r_hi)[1],
    )


def o3_box_images(boxes):
    """Enclose the O(3) quotient map on boxes in their supplied row order."""
    boxes = _boxes(boxes, 2)
    return np.asarray([o3_box_image(tuple(box)) for box in boxes], dtype=np.float64).reshape(-1, 4)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=("sphere", "o3"))
    parser.add_argument("input", type=Path, help="NPZ file containing the boxes array")
    parser.add_argument("output", type=Path, help="New NPZ file for boxes and image_boxes")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("The output filename must end in .npz.")
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as data:
        boxes = data["boxes"]
    images = sphere_box_images(boxes) if args.system == "sphere" else o3_box_images(boxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, boxes=boxes, image_boxes=images)
    print(f"Computed {len(boxes)} {args.system} image enclosures.")


if __name__ == "__main__":
    main()
