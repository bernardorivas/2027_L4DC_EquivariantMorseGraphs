"""Native CMGDB panels and diagnostics for the two learned attitude reductions."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import graphviz
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
import CMGDB

from common import figure_style as style
from common.workflow import read, write


def save(fig, folder, name):
    """Keep fixed page coordinates so vector panels can be placed precisely."""
    path = folder / f"{name}.pdf"
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    return path


def render(path):
    subprocess.run(["pdftoppm", "-scale-to", "1800", "-png", "-singlefile", str(path), str(path.with_suffix(""))],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def footer(fig, text):
    fig.text(.045, .025, text, fontsize=8, color=style.GRAPHITE)


def phase_by_bin(folder, graph_folder="cmgdb"):
    with np.load(folder / graph_folder / "phase.npz") as phase:
        count = len(phase["boxes"])
        order = np.clip(np.floor((phase["boxes"].mean(axis=1)+1)*count/2).astype(int), 0, count-1)
        assert np.array_equal(np.sort(order), np.arange(count))
        membership, basins = np.empty(count, int), np.empty(count, int)
        membership[order], basins[order] = phase["membership"], phase["basin_node"]
    return membership, basins


def native_panels(folder, out, name, graph_folder="cmgdb"):
    """Use the native labels, graph layout, and one-dimensional cell renderer."""
    record = read(folder / graph_folder / "results.json")
    source = (folder / graph_folder / "morse-graph.dot").read_text()
    lines, seen = [], set()
    records = {row["node"]: row for row in record["nodes"]}
    for line in source.splitlines():
        match = re.match(r'\s*(\d+)\s+\[label="([^"]*)"', line)
        if match:
            node, label = int(match.group(1)), match.group(2)
            annotations = records[node]["annotations"]
            expected = f"{node} : ("+", ".join(annotations)+")" if annotations else str(node)
            assert label == expected
            seen.add(node)
            line = re.sub(r'fillcolor="[^"]*"', f'fillcolor="{style.TINT[node % len(style.TINT)]}"', line)
            line = line.replace('];', f', color="{style.MORSE[node % len(style.MORSE)]}", penwidth="0.8", fontcolor="{style.INK}"];')
        lines.append(line)
        if line.strip() == "digraph {":
            lines.append(f'graph [{style.GRAPH_FRAME}];')
            lines.append(f'edge [color="{style.GRAPHITE}", penwidth="0.9"];')
    assert seen == set(records)
    display = "\n".join(lines)+"\n"
    (out / f"{name}-morse-graph.dot").write_text(display)
    graph = out / f"{name}-morse-graph.pdf"
    graph.write_bytes(graphviz.Source(display).pipe(format="pdf"))
    fig, ax = CMGDB.PlotMorseSets1D(str(folder / graph_folder / "morse-sets.csv"), clist=style.MORSE,
                  fig_w=5.8, fig_h=1.4, xlim=[-1.04, 1.04], xlabel="$u$", fontsize=11,
                  axis_arrow=False, height=.16, label_sets=False, show=False)
    ax.set_yticks([])
    cells = save(fig, out, f"{name}-morse-cells")
    return graph, cells


def place(page, path, rectangle, enlarge=False):
    """Merge a PDF panel without converting its text, edges, or nodes to pixels."""
    panel = PdfReader(path).pages[0]
    x, y, width, height = rectangle
    size = (float(panel.mediabox.width), float(panel.mediabox.height))
    scale = min(width/size[0], height/size[1])
    if not enlarge:
        scale = min(1., scale)
    x += (width-scale*size[0])/2
    y += (height-scale*size[1])/2
    page.merge_transformed_page(panel, Transformation().scale(scale).translate(x, y))
    return scale


def summary_page(run, selection):
    rows = read(run / "summary.json")
    count = rows[0]["graph"]["cells"]
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.05, .94, f"Example 3: regular interval CMGDB, {count} cells", fontsize=17)
    fig.text(.05, .895, "Native interval annotations and every computed component retained.", fontsize=11)
    values = []
    for row in rows:
        policy, seed = row["policy"], row["seed"]
        folder = run / f"{policy}/seed{seed}"
        physical = read(folder / "sphere-metrics.json")
        old_path = Path(__file__).parent / f"reference/{policy}/seed{seed}/cmgdb/results.json"
        old_count = len(read(old_path)["nodes"]) if old_path.exists() else "n/a"
        nodes = row["graph"]["nodes"]
        nonzero = sum(any(value != "0" for value in node["annotations"]) for node in nodes)
        selected = " *" if seed == selection[policy]["selected_seed"] else ""
        values.append([policy, str(seed)+selected, old_count, len(nodes), nonzero,
                       len(row["graph"]["sink_nodes"]), f"{physical['physical_one_step_sphere_consistency_rms']:.4f}",
                       f"{row['interval']['consistency_rms']:.4f}", f"{physical['physical_source_supported_cells']}/{count}"])
    ax = fig.add_axes([.045, .52, .91, .29])
    ax.axis("off")
    labels = ["Controller", "Seed", "Stored 64", f"Current {count}", "Nonzero", "Sinks", "Sphere RMS", "Interval RMS", "Support"]
    table = ax.table(cellText=values, colLabels=labels, loc="center", cellLoc="center",
                     colWidths=[.12, .06, .10, .10, .09, .07, .13, .13, .10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(style.LIGHT)
        if row == 0:
            cell.set_facecolor("#f1f1f1")
    fig.text(.06, .44, "* Selected by full-plant validation prediction before graph analysis.\n"
             "Stored/current columns count all native components; nonzero counts nonzero native annotations.\n"
             "Sphere RMS tests the physical-to-sphere square. Interval RMS tests the sphere-to-interval square.",
             fontsize=10, linespacing=1.7)
    fig.text(.06, .29, "The native tuples below belong to the interval, not the sphere.\n"
             "Circle fibers and collapsed pole fibers require separate index reconstruction.\n"
             "An attracting sphere cycle has index tuple (x-1, x-1, 0), not (x-1, 0, 0).",
             fontsize=11, linespacing=1.7)
    endpoint_errors = [row["interval"]["endpoint_compatibility_maximum_error"] for row in rows
                       if "endpoint_compatibility_maximum_error" in row["interval"]]
    if endpoint_errors:
        fig.text(.06, .22, f"Largest mismatch with sphere-induced pole images: {max(endpoint_errors):.5f}.", fontsize=10)
    references = []
    for policy in selection:
        path = run / f"thrust-axis-reference/{policy}/results.json"
        if path.exists():
            reference = read(path)
            references.append(f"{policy}: {len(reference['nodes'])} components, {len(reference['nonzero_nodes'])} nonzero")
    if references:
        fig.text(.06, .18, "Matched time-0.1 references: "+"; ".join(references)+".", fontsize=10)
    fig.text(.06, .09, "Current controller reference orders: nominal S -> N; coning S -> C <- D -> N.\n"
             "The learned fits do not reproduce these orders. Small-cell self-loops and model error are separate issues.\n"
             "No physical transfer certificate is claimed.", fontsize=10, linespacing=1.7)
    footer(fig, "CMGDB on sampled function hulls; physical support is measured from encoded training and validation data.")
    return save(fig, run / "figures", "summary")


def learning_page(run, policy, seed):
    folder = run / f"{policy}/seed{seed}"
    with np.load(folder / "interval/diagnostics.npz") as saved:
        heights, coordinates = saved["heights"], saved["coordinates"]
    from . import interval
    import jax.numpy as jnp
    reduced = interval.load(folder / "interval")
    _, _, advance, _ = interval.functions(reduced)
    grid = np.linspace(-1, 1, 4097, dtype=np.float32)
    values = np.asarray(advance(jnp.asarray(grid)))
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
    axes[0, 0].plot(heights, coordinates, color=style.MORSE[0], lw=1.3)
    axes[0, 0].plot([-1, 1], [-1, 1], color=style.GRAY, lw=.7, ls=":")
    axes[0, 0].set(xlabel="Latent sphere height $z_3$", ylabel=r"$\pi(z)$", title="Learned invariant encoder")
    axes[0, 1].plot(grid, values, color=style.MORSE[0], lw=1.3)
    axes[0, 1].plot([-1, 1], [-1, 1], color=style.GRAY, lw=.7, ls=":")
    axes[0, 1].set(xlabel="$u$", ylabel=r"$\overline{f}(u)$", title="Learned time map, 0.1 s")
    history = read(folder / "interval/history.json")
    axes[1, 0].semilogy([row["step"] for row in history], [row["validation"] for row in history], color=style.MORSE[0], lw=1.)
    axes[1, 0].set(xlabel="Interval-stage update", ylabel="Validation objective", title="Joint interval learning")
    axes[1, 1].axhline(0, color=style.GRAY, lw=.7)
    axes[1, 1].plot(grid, values-grid, color=style.MORSE[0], lw=1.3)
    axes[1, 1].set(xlabel="$u$", ylabel=r"$\overline{f}(u)-u$", title="Drift and fixed-point crossings")
    fig.suptitle(f"{policy.capitalize()}, seed {seed}: learned maps", fontsize=15)
    fig.tight_layout(rect=(.025, .075, .98, .94), h_pad=3., w_pad=3.)
    footer(fig, "The sphere coordinate is learned; it is not prescribed to equal physical thrust height.")
    return save(fig, run / "figures", f"{policy}-learning")


def sphere_axis(ax, membership, basins, knot_x, knot_y):
    """Project visible latitude cells directly; keep the geometry as vector paths."""
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


def geometry_page(run, policy, seed, graph_path, cells_path):
    folder = run / f"{policy}/seed{seed}"
    graph = read(folder / "cmgdb/results.json")
    membership, basins = phase_by_bin(folder)
    with np.load(folder / "interval/diagnostics.npz") as data:
        knot_x, knot_y = data["knot_heights"], data["knot_coordinates"]
    panel = PdfReader(graph_path).pages[0]
    graph_width, graph_height = float(panel.mediabox.width), float(panel.mediabox.height)
    graph_scale = min(1., 9.8*72/graph_width)
    graph_inches = graph_height*graph_scale/72
    height = max(8.3, graph_inches+4.8)
    fig = plt.figure(figsize=(11.7, height))
    fig.text(.05, 1-.38/height, f"{policy.capitalize()}, seed {seed}: native CMGDB", fontsize=16)
    fig.text(.05, 1-.72/height, f"{graph['cells']} interval cells; {len(graph['nodes'])} components; interval index annotations", fontsize=11)
    ax = fig.add_axes([.055, 1-3.48/height, .30, 2.50/height])
    sphere_axis(ax, membership, basins, knot_x, knot_y)
    fig.text(.075, 1-3.65/height, "Lift to the learned sphere", fontsize=10)
    fig.text(.47, 1-1.20/height, "Native interval Morse cells", fontsize=11)
    fig.text(.47, 1-3.20/height, "Sphere: dark Morse cells, pale sink basins.\nLight gray: unresolved. Interval: Morse cells.\nColors follow graph node IDs.", fontsize=10, linespacing=1.45)
    fig.text(.05, 1-3.95/height, "Native Morse graph", fontsize=12)
    footer(fig, "Sphere regions are latent. These interval tuples have not been reconstructed on the sphere.")
    path = save(fig, run / "figures", f"{policy}-geometry")
    reader = PdfReader(path)
    page = reader.pages[0]
    width_pt, height_pt = float(page.mediabox.width), float(page.mediabox.height)
    place(page, cells_path, (.42*width_pt, height_pt-2.70*72, .54*width_pt, 1.40*72), enlarge=True)
    place(page, graph_path, (.05*width_pt, .55*72, .90*width_pt, height_pt-4.75*72))
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run, selection = args.run, read(args.run / "selection.json")
    out = run / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({**style.mpl_rc(), "font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    panels, all_graphs = {}, PdfWriter()
    for row in read(run / "summary.json"):
        policy, seed = row["policy"], row["seed"]
        name = f"{policy}-seed{seed}"
        panels[(policy, seed)] = native_panels(run / f"{policy}/seed{seed}", out, name)
        all_graphs.append(panels[(policy, seed)][0], outline_item=f"{policy}, seed {seed}")
    all_graphs.write(run / "native-graphs.pdf")
    reference_graphs = PdfWriter()
    for policy in selection:
        path = run / f"thrust-axis-reference/{policy}/morse-graph.pdf"
        if path.exists():
            reference_graphs.append(path, outline_item=f"{policy}, time 0.1 reference")
    if len(reference_graphs.pages):
        reference_graphs.write(run / "reference-graphs.pdf")
    pages = [summary_page(run, selection)]
    for policy, choice in selection.items():
        seed = choice["selected_seed"]
        pages.append(geometry_page(run, policy, seed, *panels[(policy, seed)]))
        pages.append(learning_page(run, policy, seed))
    writer = PdfWriter()
    for path in pages:
        writer.append(path)
        render(path)
    writer.add_metadata({"/Title": "Example 3: native CMGDB and learned-map diagnostics"})
    writer.write(run / "results.pdf")
    write(run / "figure-summary.json", dict(pages=len(pages), selected_seeds={p: v["selected_seed"] for p, v in selection.items()},
          native_graphs=len(panels), vector_graphs=True, scope="native interval graphs and cells, latent sphere lifts, learned-map diagnostics"))


if __name__ == "__main__":
    main()
