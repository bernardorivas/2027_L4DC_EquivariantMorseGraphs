"""Orthographic sphere preimages of interval Morse cells and basin labels."""

import numpy as np
from matplotlib.collections import PolyCollection
from . import figure_style as style


def sphere_axis(ax, membership, basins, knot_x, knot_y):
    """Project visible latitude cells as polygons, with the saved native colors."""
    count = len(membership)
    boundaries = np.interp(np.linspace(-1, 1, count+1), knot_y, knot_x)
    height = np.unique(np.r_[np.linspace(-1, 1, 181), boundaries])
    phi = np.linspace(0, 2*np.pi, 181)
    elevation, azimuth = np.deg2rad(8), -np.pi/3
    direction = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    vertical = np.cross(direction, horizontal)
    polygons, colors = [], []
    for lo, hi in zip(height[:-1], height[1:], strict=True):
        mid = (lo+hi)/2
        cell = np.clip(np.searchsorted(boundaries, mid, side="right")-1, 0, count-1)
        node = membership[cell]
        color = style.MORSE[node % len(style.MORSE)] if node >= 0 else (
            style.TINT[basins[cell] % len(style.TINT)] if basins[cell] >= 0 else "#f4f4f4")
        z = np.array([lo, hi])
        radius = np.sqrt(np.maximum(0., 1-z*z))
        ring = np.stack((radius[:, None]*np.cos(phi), radius[:, None]*np.sin(phi), np.broadcast_to(z[:, None], (2, len(phi)))), axis=-1)
        for j in range(len(phi)-1):
            quad = ring[[0, 0, 1, 1], [j, j+1, j+1, j]]
            if quad.mean(axis=0) @ direction <= 0:
                continue
            polygons.append(np.column_stack((quad @ horizontal, quad @ vertical)))
            colors.append(color)
    ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="none", antialiased=False))
    theta = np.linspace(0, 2*np.pi, 401)
    ax.plot(np.cos(theta), np.sin(theta), color=style.GRAY, lw=.6)
    for z in (-.75, 0, .75):
        points = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), np.full_like(phi, z)))
        visible = points @ direction > 0
        ax.plot(np.where(visible, points @ horizontal, np.nan), np.where(visible, points @ vertical, np.nan), color=style.GRAY, lw=.35)
    for z, name in ((1., "$N$"), (-1., "$S$")):
        ax.text(0., z*1.09, name, ha="center", va="center", fontsize=11)
    ax.set(aspect="equal", xlim=(-1.09, 1.09), ylim=(-1.18, 1.18))
    ax.axis("off")
