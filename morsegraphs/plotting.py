"""Palette and panels shared by the three examples.

Every panel is saved bare, as a PDF and an SVG cropped to its ink: no panel letters,
titles or legends, which the manuscript sets in LaTeX. A Morse set is colored by its role
in the example (an attractor, a repeller, a saddle orbit), through the map `ROLE`, so the
same invariant set keeps its color in every figure whatever its number in a computed graph.
"""
from pathlib import Path
import subprocess
import tempfile

import numpy as np

INK, GRAPHITE, GRAY, LIGHT, WHITE = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#d9d9d9", "#ffffff"
BLUE, VERMILION, GREEN, PURPLE, ORANGE, SKY, OCHRE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#A6761D")
# Attractors take cool colors, repellers warm ones, and the saddle orbits purple.
ROLE = {"A-": BLUE, "A+": GREEN, "C": PURPLE, "S": VERMILION, "N": ORANGE, "O": ORANGE,
        "e3": GREEN, "-e3": VERMILION, "A": SKY}
UNLABELED = OCHRE
NODE_STROKE_PT, EDGE_STROKE_PT, PAD_BP = 0.8, 0.9, 1.0


def mix(color, fraction=0.4, base=WHITE):
    """The color at `fraction` of the way from `base` to `color`."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(base[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(fraction * x + (1 - fraction) * y) for x, y in zip(a, b))


def color(role):
    return ROLE.get(role, UNLABELED)


def tint(role):
    return mix(color(role))


def style():
    """Matplotlib settings of the panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "mathtext.fontset": "cm", "pdf.fonttype": 42,
        "svg.fonttype": "none", "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAPHITE,
        "xtick.color": GRAPHITE, "ytick.color": GRAPHITE, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": WHITE,
        "axes.facecolor": WHITE, "savefig.facecolor": WHITE})
    return plt


def save_panel(fig, path, dpi=600):
    """Save `fig` as `<path>.pdf` and `<path>.svg`, cropped to the ink plus one point.

    The ink box is measured on a rasterized PDF (Poppler's `pdftoppm`), so whitespace that
    Matplotlib counts, such as empty axes, is dropped. Returns the PDF path.
    """
    from matplotlib.transforms import Bbox
    from PIL import Image
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    probe = Bbox.from_extents(tight.x0 - .1, tight.y0 - .1, tight.x1 + .1, tight.y1 + .1)
    with tempfile.TemporaryDirectory(prefix="panel-") as tmp:
        pdf, stem = Path(tmp) / "probe.pdf", Path(tmp) / "probe"
        fig.savefig(pdf, bbox_inches=probe)
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-singlefile", str(pdf), str(stem)], check=True)
        image = np.asarray(Image.open(stem.with_suffix(".pgm")))
    rows, cols = np.nonzero(image < 255)
    per_in = image.shape[1] / probe.width
    pad = PAD_BP / 72
    box = Bbox.from_extents(probe.x0 + cols.min() / per_in - pad, probe.y1 - (rows.max() + 1) / per_in - pad,
                            probe.x0 + (cols.max() + 1) / per_in + pad, probe.y1 - rows.min() / per_in + pad)
    fig.savefig(path, bbox_inches=box, metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".svg"), bbox_inches=box, format="svg", metadata={"Creator": None, "Date": None})
    return path


def graph(nodes, edges, path, scale=0.55):
    """Draw a Morse graph with Graphviz as `<path>.pdf` and `<path>.svg`.

    `nodes` maps a node to `(label, role)`, for example `{0: ("A−: (x-1, 0)", "A-")}`;
    `edges` lists `(a, b)` pairs. `scale` is the printed size over the natural size, so
    strokes print at fixed widths.
    """
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['digraph {', '  graph [margin="0", pad="0.04", ranksep="0.3", nodesep="0.25"];',
             f'  node [shape=ellipse, style=filled, fontname="Times-Roman", fontsize=14, fontcolor="{INK}", '
             f'penwidth="{NODE_STROKE_PT / scale:.3f}"];',
             f'  edge [color="{GRAPHITE}", penwidth="{EDGE_STROKE_PT / scale:.3f}", arrowsize=0.7];']
    for node, (text, role) in nodes.items():
        lines.append(f'  {node} [label="{text}", color="{color(role)}", fillcolor="{tint(role)}"];')
    lines += [f"  {a} -> {b};" for a, b in edges] + ["}"]
    source = "\n".join(lines)
    for extension in ("pdf", "svg"):
        subprocess.run(["dot", f"-T{extension}", "-o", str(path.with_suffix("." + extension))],
                       input=source, text=True, check=True)
    path.with_suffix(".dot").write_text(source + "\n")
    return path


def label(role, annotations):
    """A node label such as `A−: (x-1, 0)`, with a true minus sign."""
    name = role.replace("-", "−") if role else "?"
    return f"{name}: ({', '.join(annotations)})"


def interval_strip(ax, boxes, roles, xlabel, membership, height=0.2):
    """Morse sets on an interval: each cell of a Morse set as a bar in the color of its role.

    `boxes` are the cells (rows lower, upper), `membership` the Morse set of each cell (-1
    for none) and `roles` maps a Morse set to its role.
    """
    ax.axhline(0, color=GRAPHITE, lw=.6, zorder=0)
    for cell, node in enumerate(membership):
        if node >= 0:
            lo, hi = boxes[cell]
            ax.add_patch(_rectangle((lo, -height / 2), hi - lo, height, color(roles.get(node))))
    lo, hi = float(np.min(boxes)), float(np.max(boxes))
    ax.set(xlim=(lo - .03 * (hi - lo), hi + .03 * (hi - lo)), ylim=(-.5, .5), xlabel=xlabel, yticks=[])
    ax.spines[["left", "top", "right"]].set_visible(False)


def rectangle_cells(ax, boxes, colors, xlabel, ylabel):
    """Cells of a planar grid, each filled with its color (None leaves it blank)."""
    from matplotlib.collections import PatchCollection
    patches, faces = [], []
    for box, value in zip(boxes, colors):
        if value is not None:
            patches.append(_rectangle(box[:2], box[2] - box[0], box[3] - box[1], value))
            faces.append(value)
    ax.add_collection(PatchCollection(patches, facecolors=faces, edgecolors="none"))
    ax.set(xlim=(boxes[:, 0].min(), boxes[:, 2].max()), ylim=(boxes[:, 1].min(), boxes[:, 3].max()),
           xlabel=xlabel, ylabel=ylabel, aspect="equal")


def sphere_bands(ax, bands, elevation=8., azimuth=-60., resolution=241):
    """Bands of latitude on the visible hemisphere of the unit sphere.

    `bands` lists `(z_lower, z_upper, color)`. The view looks from the given elevation and
    azimuth in degrees; the outline and the circles of latitude at -0.75, 0 and 0.75 are
    drawn in gray, and the poles are marked N and S.
    """
    from matplotlib.collections import PolyCollection
    elevation, azimuth = np.deg2rad(elevation), np.deg2rad(azimuth)
    direction = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    vertical = np.cross(direction, horizontal)
    phi = np.linspace(0, 2 * np.pi, resolution)
    polygons, faces = [], []
    for lo, hi, value in bands:
        for a, b in zip(np.linspace(lo, hi, 4)[:-1], np.linspace(lo, hi, 4)[1:]):
            z = np.array([a, b])
            radius = np.sqrt(np.maximum(0., (1 - z) * (1 + z)))
            ring = np.stack((radius[:, None] * np.cos(phi), radius[:, None] * np.sin(phi),
                             np.broadcast_to(z[:, None], (2, resolution))), axis=-1)
            for j in range(resolution - 1):
                quad = ring[[0, 0, 1, 1], [j, j + 1, j + 1, j]]
                if quad.mean(axis=0) @ direction > 0:
                    polygons.append(np.column_stack((quad @ horizontal, quad @ vertical)))
                    faces.append(value)
    ax.add_collection(PolyCollection(polygons, facecolors=faces, edgecolors=faces, linewidths=.1))
    theta = np.linspace(0, 2 * np.pi, 401)
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=.6)
    for z in (-.75, 0., .75):
        points = np.column_stack((np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), np.full_like(phi, z)))
        visible = points @ direction > 0
        ax.plot(np.where(visible, points @ horizontal, np.nan), np.where(visible, points @ vertical, np.nan),
                color=GRAY, lw=.35)
    for height, text in ((1., "$N$"), (-1., "$S$")):
        point = np.array([0., 0., height])
        ax.plot(point @ horizontal, point @ vertical, "o", color=INK, ms=2)
        ax.text(0., height * 1.12, text, ha="center", va="center")
    ax.set(aspect="equal", xlim=(-1.08, 1.08), ylim=(-1.2, 1.2))
    ax.axis("off")


def show(paths, titles=None, columns=None, width=10.):
    """Display saved panels side by side in a notebook, as one image."""
    import io
    from IPython.display import Image, display
    from PIL import Image as PILImage
    plt = style()
    paths = [Path(p).with_suffix(".pdf") for p in paths]
    columns = columns or len(paths)
    rows = int(np.ceil(len(paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(width, 3.2 * rows), squeeze=False)
    with tempfile.TemporaryDirectory(prefix="show-") as tmp:
        for ax, path, title in zip(axes.flat, paths, titles or [""] * len(paths)):
            stem = Path(tmp) / path.stem
            subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", str(path), str(stem)], check=True)
            ax.imshow(np.asarray(PILImage.open(stem.with_suffix(".png"))))
            ax.set_title(title, fontsize=11)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    display(Image(data=buffer.getvalue()))


def _rectangle(corner, width, height, value):
    from matplotlib.patches import Rectangle
    return Rectangle(corner, width, height, facecolor=value, edgecolor="none")
