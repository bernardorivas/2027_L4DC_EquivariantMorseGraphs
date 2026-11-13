"""Closed-form O(3) quotient flow and named fixed orbits."""

import numpy as np

MATH = {"A-": r"A_-", "A+": r"A_+", "C": "C", "O": "O"}
ANCHORS = {"A-": (-1.0, 0.0), "A+": (1.0, 0.0), "C": (0.0, 1.0), "O": (0.0, 0.0)}


def flow(a0, r0, t):
    """Exact closed form, evaluated numerically for illustration."""
    exponential = np.exp(2*t)
    d = (1-a0*a0) + a0*a0*exponential
    total = a0*a0 + r0*r0
    total_denominator = (1-total) + total*exponential
    return a0*np.sqrt(exponential/d), r0*np.sqrt(exponential/(d*total_denominator))
