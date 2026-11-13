"""Render native CMGDB results and physical sphere cells from a new EMLP fit."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
import graphviz
import jax.numpy as jnp
import CMGDB

from common import figure_style as style
from .training import evaluator, load, meridian


def read(path):
    return json.loads(path.read_text())


def save(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)


def native_panels(folder, out, prefix, coordinate):
    record = read(folder/"native-result.json")
    positions = sorted((float(np.mean(np.asarray(row["boxes"]))), row["node"])
                       for row in record["nodes"])
    levels, labels = [], []
    for x, node in positions:
        level = next((i for i, last in enumerate(levels) if x-last >= .10), len(levels))
        if level == len(levels):
            levels.append(x)
        else:
            levels[level] = x
        labels.append((x, node, level))
    top_level = max((level for _, _, level in labels), default=0)
    fig, ax = CMGDB.PlotMorseSets1D(str(folder/"morse-sets.csv"), clist=style.MORSE,
                  fig_w=5, fig_h=1.2+.18*top_level, xlim=[-1.04, 1.04], xlabel=coordinate,
                  axis_arrow=False, height=.20, label_sets=False, show=False)
    for x, node, level in labels:
        ax.annotate(str(node), xy=(x, .11), xytext=(x, .25+.23*level), ha="center", va="bottom",
                    fontsize=10, arrowprops=dict(arrowstyle="-", color=style.GRAY, lw=.4))
    ax.set_ylim(-.43, .55+.25*top_level)
    save(fig, out/f"{prefix}-cells")
    plt.close(fig)
    lines = []
    for line in (folder/"morse-graph.dot").read_text().splitlines():
        match = re.match(r'\s*(\d+)\s+\[label=', line)
        if match:
            node = int(match.group(1))
            color = style.MORSE[node % len(style.MORSE)]
            tint = style.TINT[node % len(style.TINT)]
            line = re.sub(r'fillcolor="[^"]*"', f'fillcolor="{tint}"', line)
            line = line.replace('];', f', color="{color}", penwidth="1.0", fontcolor="{style.INK}"];')
        lines.append(line)
    graph = graphviz.Source("\n".join(lines))
    for extension in ("pdf", "png"):
        (out/f"{prefix}-graph.{extension}").write_bytes(graph.pipe(format=extension))
    return record


def sphere_panel(record, physical_edges, out):
    """Draw full preimages of the computed intervals on the visible hemisphere."""
    depth = record["uniform_depth"]
    cell_nodes = np.full(2**depth, -1, int)
    for row in record["nodes"]:
        cell_nodes[row["cell_indices"]] = row["node"]
    base_z = np.unique(np.r_[np.linspace(-1, 1, 181), physical_edges])
    phi = np.linspace(0, 2*np.pi, 241)
    azimuth, elevation = -np.pi/3, np.deg2rad(8)
    direction = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth),
                          np.sin(elevation)])
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    vertical = np.cross(direction, horizontal)
    polygons, colors = [], []
    for lo, hi in zip(base_z[:-1], base_z[1:], strict=True):
        height = (lo+hi)/2
        cell = np.clip(np.searchsorted(physical_edges, height, side="right")-1, 0, len(cell_nodes)-1)
        node = cell_nodes[cell]
        color = style.TINT[node % len(style.TINT)] if node >= 0 else "#f1f1f1"
        z = np.array([lo, hi])
        radius = np.sqrt(np.maximum(0., (1-z)*(1+z)))
        ring = np.stack((radius[:, None]*np.cos(phi), radius[:, None]*np.sin(phi),
                         np.broadcast_to(z[:, None], (2, len(phi)))), axis=-1)
        for j in range(len(phi)-1):
            quad = ring[[0, 0, 1, 1], [j, j+1, j+1, j]]
            if quad.mean(axis=0)@direction <= 0:
                continue
            polygons.append(np.column_stack((quad@horizontal, quad@vertical)))
            colors.append(color)
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="none", rasterized=True))
    theta = np.linspace(0, 2*np.pi, 401)
    ax.plot(np.cos(theta), np.sin(theta), color=style.GRAY, lw=.6)
    for z in (-.75, 0, .75):
        points = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi),
                                  np.full_like(phi, z)))
        visible = points@direction > 0
        x, y = points@horizontal, points@vertical
        ax.plot(np.where(visible, x, np.nan), np.where(visible, y, np.nan), color=style.GRAY, lw=.35)
    for height, text in ((1., "$N$"), (-1., "$S$")):
        point = np.array([0., 0., height])
        ax.plot(point@horizontal, point@vertical, "o", color=style.INK, ms=2)
        ax.text(0., height*1.10, text, ha="center", va="center")
    ax.set(aspect="equal", xlim=(-1.08, 1.08), ylim=(-1.18, 1.18))
    ax.axis("off")
    save(fig, out)
    plt.close(fig)


def height_reference(z):
    z = np.asarray(z)
    result = z.copy()
    active = np.abs(z) < 1
    sol = solve_ivp(lambda _, w: np.tanh(w)*(.16-np.tanh(w)**2), (0., 10.),
                    np.arctanh(z[active]), method="DOP853", rtol=2e-13, atol=2e-14)
    result[active] = np.tanh(sol.y[:, -1])
    return result


def diagnostics_panels(folder, out, prefix):
    model = load(folder)
    encode = evaluator(model, lambda z: model.encode(meridian(z)))
    decode = evaluator(model, model.decode)
    def ten(u):
        for _ in range(10):
            u = model.advance(u)
        return u
    advance = evaluator(model, ten)
    z = np.linspace(-1, 1, 2001)
    eta = np.asarray(encode(jnp.asarray(z)))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(z, z, color=style.GRAY, lw=.8, ls="--", label="height")
    ax.plot(z, eta, color=style.MORSE[0], label="learned encoder")
    ax.plot(z, np.asarray(decode(jnp.asarray(eta))), color=style.MORSE[1], lw=.8,
            label="decoded orbit height")
    ax.set(xlabel="Physical height $z$", ylabel="Interval coordinate", xlim=(-1.02, 1.02))
    ax.legend(frameon=False, fontsize=8)
    save(fig, out/f"{prefix}-encoder")
    plt.close(fig)

    u = np.linspace(-1, 1, 2001)
    physical_start = np.interp(u, eta, z)
    reference = np.asarray(encode(jnp.asarray(height_reference(physical_start))))
    learned = np.asarray(advance(jnp.asarray(u)))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(u, u, color=style.LIGHT, lw=.8)
    ax.plot(u, reference, color=style.INK, lw=1.1, label="reference in learned coordinates")
    ax.plot(u, learned, color=style.MORSE[1], lw=.9, ls="--", label="learned tenth iterate")
    ax.set(xlabel="Latent coordinate $u$", ylabel="Time-ten image", xlim=(-1.02, 1.02), ylim=(-1.02, 1.02))
    ax.legend(frameon=False, fontsize=8)
    save(fig, out/f"{prefix}-time-map")
    plt.close(fig)

    history = read(folder/"history.json")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.semilogy([r["step"] for r in history], [r["validation"] for r in history], color=style.MORSE[0])
    ax.set(xlabel="Joint optimization step", ylabel="Validation loss")
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    save(fig, out/f"{prefix}-loss")
    plt.close(fig)


def image_axis(ax, path, title):
    ax.imshow(np.asarray(Image.open(path)))
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--depth", type=int, default=7)
    args = parser.parse_args()
    args.seeds.sort(key=lambda seed: read(args.run/f"seed{seed}"/"config.json")["best_validation"])
    selection = dict(criterion="lowest validation objective", selected_seed=args.seeds[0],
                     ranking=[dict(seed=seed, validation=read(args.run/f"seed{seed}"/"config.json")["best_validation"])
                              for seed in args.seeds])
    (args.run/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    out = args.run/"figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(style.mpl_rc())
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "pdf.fonttype": 42})
    reference = native_panels(args.run/"reference"/f"K{2**args.depth}", out, "explicit", "$z$")
    boundaries = np.linspace(-1, 1, 2**args.depth+1)
    sphere_panel(reference, boundaries, out/"explicit-lift")
    with PdfPages(args.run/"sphere-emlp-comparison.pdf") as pdf:
        for seed in args.seeds:
            prefix = f"seed{seed}"
            folder = args.run/prefix
            record = native_panels(folder/f"K{2**args.depth}", out, prefix, "$u$")
            physical = np.load(folder/f"K{2**args.depth}"/"lifted-grid.npz")["physical_boundaries"]
            sphere_panel(record, physical, out/f"{prefix}-lift")
            diagnostics_panels(folder, out, prefix)
            metrics = read(folder/"metrics.json")
            config = read(folder/"config.json")
            fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5), gridspec_kw={"height_ratios": [2., .8, 1.5]})
            for col, name, title in ((0, "explicit", "Explicit reference"), (1, prefix, f"Joint EMLP fit, seed {seed}")):
                for row, suffix in enumerate(("lift", "cells", "graph")):
                    image_axis(axes[row, col], out/f"{name}-{suffix}.png", title if row == 0 else "")
            fig.suptitle(f"Sphere to interval: physical preimages, CMGDB cells, and native indices ({2**args.depth} cells)", fontsize=12)
            fig.text(.5, .025, "Learned graphs use sampled image hulls. Comparisons are numerical; no transfer certificate is claimed.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .045, 1, .965))
            pdf.savefig(fig)
            fig.savefig(args.run/f"comparison-{prefix}.png", dpi=180)
            fig.savefig(args.run/f"comparison-{prefix}.pdf")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
            for ax, suffix, title in zip(axes.flat[:3], ("encoder", "time-map", "loss"),
                                         ("Encoder and orbit reconstruction", "Latent time-ten map", "Joint training"), strict=True):
                image_axis(ax, out/f"{prefix}-{suffix}.png", title)
            ax = axes[1, 1]
            ax.axis("off")
            lines = [f"Seed {seed}, checkpoint at step {config['best_step']}",
                     f"Physical inputs: 3; latent dimension: 1", "",
                     f"Reconstruction orbit RMS: {metrics['forecasts']['0']['orbit_chordal_rms']:.3g}",
                     f"One-step orbit RMS: {metrics['forecasts']['1']['orbit_chordal_rms']:.3g}",
                     f"Ten-step orbit RMS: {metrics['forecasts']['10']['orbit_chordal_rms']:.3g}",
                     f"Sampled residual maximum: {metrics['sampled_residual_maximum']:.3g}",
                     f"Sampled encoder invariance defect: {metrics['sampled_encoder_invariance_error']:.3g}",
                     f"Minimum sampled encoder derivative: {metrics['minimum_sampled_encoder_derivative']:.3g}",
                     f"Minimum sampled time-map derivative: {metrics['minimum_sampled_time_map_derivative']:.3g}", "",
                     f"Native reference Morse sets: {len(reference['nodes'])}",
                     f"Native learned Morse sets: {len(record['nodes'])}", "",
                     "Encoder, orbit decoder, and time map trained jointly.",
                     "Checkpoint selected by validation loss before CMGDB."]
            ax.text(0., .98, "\n".join(lines), ha="left", va="top", fontsize=10, linespacing=1.5)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(args.run/"sphere-emlp-comparison.pdf")


if __name__ == "__main__":
    main()
