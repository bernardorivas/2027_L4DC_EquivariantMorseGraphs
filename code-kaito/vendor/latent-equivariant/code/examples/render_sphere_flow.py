"""Render the rotating sphere flow and the time-three height map."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
_MPL_TMP = tempfile.TemporaryDirectory(prefix="sphere-flow-mpl-")
os.environ["MPLCONFIGDIR"] = _MPL_TMP.name

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sphere_geometry as geom
import figure_style as style


FOLDER = "sphere/flow"

SPHERE_AXES_IN = (6.2 * .43, 2.6 * .90)
HEIGHT_AXES_IN = (6.2 * .37, 2.6 * .74)
TAU = 3.0
ELEV = np.deg2rad(8.0)
AZIM = geom.AZIM


NODE = {"T-": 0, "S": 1, "T+": 2, "E": 3, "N": 4}
RESULTS = HERE / "data/sphere"


def check_nodes():
    """Check the saved CMGDB node ids."""
    for tag in ("exact-K128", "seed0-K128"):
        nodes = json.loads((RESULTS / tag / "native-result.json").read_text())["nodes"]
        names = {row["geometric_name"]: row["node"] for row in nodes}
        assert names == NODE, (tag, names)

RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
      "font.size": 9, "axes.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False,
      "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42}


def view_basis():
    d = np.array([np.cos(ELEV) * np.cos(AZIM), np.cos(ELEV) * np.sin(AZIM), np.sin(ELEV)])
    u = np.array([-np.sin(AZIM), np.cos(AZIM), 0.0])
    return d, u, np.cross(d, u)


D, U, W = view_basis()


def project(P):
    """Screen coordinates and visibility of points P (n x 3) on the unit sphere."""
    P = np.atleast_2d(P)
    return P @ U, P @ W, P @ D > 0


def draw_curve(ax, P, front, back=None, closed=False):
    """Visible part of a curve on the sphere with style front, hidden part with style back."""
    x, y, vis = project(P)
    for mask, kw in ((vis, front), (~vis, back)):
        if kw is None:
            continue
        runs = geom.split_runs(mask)
        if closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == len(mask):
            a, b = runs.pop()
            s, e = runs.pop(0)
            idx = np.r_[np.arange(a, b), np.arange(s, e)]
            ax.plot(x[idx], y[idx], **kw)
        for a, b in runs:
            ax.plot(x[a:b], y[a:b], **kw)


def draw_arrowhead(ax, P, frac):
    """Arrowhead on the longest visible run of P, at the fraction frac of it."""
    x, y, vis = project(P)
    runs = [r for r in geom.split_runs(vis) if r[1] - r[0] > 40]
    if not runs:
        return
    a, b = max(runs, key=lambda r: r[1] - r[0])
    i = a + int(frac * (b - a - 6))
    ax.annotate("", xy=(x[i + 6], y[i + 6]), xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="-|>,head_length=0.44,head_width=0.23", lw=.66,
                                color=style.GRAPHITE, shrinkA=0, shrinkB=0))


def panel_sphere(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=style.GRAY, lw=.65))
    for z, name, dash in ((0, "E", "--"), (geom.Z_T, "T+", "-"), (-geom.Z_T, "T-", "-")):
        color = style.MORSE[NODE[name]]
        draw_curve(ax, geom.latitude(z), dict(color=color, lw=1.15, ls=dash, zorder=3),
                   dict(color=color, lw=.7, ls=dash, alpha=.3), closed=True)
    for z0, phi in ((.85, -.7), (.10, -2.1), (-.10, .1), (-.85, 2.1)):
        path = geom.trajectory(z0, phi, t_max=10)
        draw_curve(ax, path, dict(color=style.GRAY, lw=.65, alpha=.65))
        draw_arrowhead(ax, path, .5)
    for z, name in ((1, "N"), (-1, "S")):
        x, y, _ = project([0, 0, z])
        ax.plot(x, y, "o", ms=4, mfc="white", mec=style.MORSE[NODE[name]], zorder=4)
        ax.text(0, z * 1.08, f"${name}$", ha="center", va="center", color=style.INK)
    for z, name in ((.4, "T_+"), (0, "E"), (-.4, "T_-")):
        ax.text(1.06, z * 1.03, f"${name}$", va="center", color=style.INK)
    ax.set_xlim(-1.08, 1.28)
    ax.set_ylim(-1.15, 1.15)


def panel_height(ax):
    diagonal = np.unique(np.r_[np.linspace(-1, 1, 901), 1 - 10. ** -np.linspace(1, 14, 250),
                               -1 + 10. ** -np.linspace(1, 14, 250)])
    z = np.unique(np.r_[np.linspace(-1, 1, 1601), -.4, .4])
    ax.plot(diagonal, diagonal, color=style.LIGHT, lw=1)
    ax.plot(z, geom.hmap(z, tau=TAU), color=style.INK, lw=1)
    for fixed, name in ((-1, "S"), (-.4, "T-"), (0, "E"), (.4, "T+"), (1, "N")):
        color = style.MORSE[NODE[name]]
        attractor = abs(fixed) == .4
        ax.plot(fixed, fixed, "o", ms=4, mfc=color if attractor else "white", mec=color)
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), xlabel="$z$", ylabel=r"$h_3(z)$")
    ax.set_xticks([-1, -.4, 0, .4, 1], ["$-1$", "$-z_t$", "$0$", "$z_t$", "$1$"])
    ax.set_yticks([-1, 0, 1])


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="figures directory; writes OUT/" + FOLDER)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    args = parser.parse_args()
    RESULTS = args.data / "sphere"

    check_nodes()
    plt.rcParams.update(RC)
    plt.rcParams.update(style.mpl_rc())
    out = args.out / FOLDER
    out.mkdir(parents=True, exist_ok=True)
    for name, size, draw in (("sphere", SPHERE_AXES_IN, panel_sphere), ("height-map", HEIGHT_AXES_IN, panel_height)):
        fig = plt.figure(figsize=size)
        draw(fig.add_axes([0, 0, 1, 1]))
        style.save_panel(fig, out / f"{name}.pdf")
        plt.close(fig)
        print(out / f"{name}.pdf")


if __name__ == "__main__":
    main()
