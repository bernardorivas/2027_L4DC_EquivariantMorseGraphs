"""Colors, panel cropping, and graph strokes for the example figures."""

import re

MORSE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#A6761D",
]

TINT_FRACTION = 0.40

INK = "#1a1a1a"
GRAPHITE = "#4d4d4d"
GRAY = "#8c8c8c"
LIGHT = "#d9d9d9"
WHITE = "#ffffff"


NODE_STROKE_PT = 0.8
EDGE_STROKE_PT = 0.9


def _mix(hex_color, t, base=WHITE):
    a = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(base[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(t * x + (1 - t) * y) for x, y in zip(a, b))


TINT = [_mix(c, TINT_FRACTION) for c in MORSE]


def morse(k):
    """Ink of Morse set k: cells, curves, markers, graph node stroke."""
    return MORSE[k]


def tint(k):
    """Tint of Morse set k: graph node fill and regions attached to set k."""
    return TINT[k]


def mpl_rc():
    """Color-related matplotlib rcParams; fonts and sizes stay with each figure."""
    return {
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRAPHITE,
        "xtick.color": GRAPHITE,
        "ytick.color": GRAPHITE,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }


def graphviz_node(k, scale=1.0):
    """Graphviz attributes for the node of Morse set k.

    scale is the factor from the graph PDF to the printed page, so that the
    stroke is NODE_STROKE_PT at the final size.
    """
    return {
        "style": "filled",
        "fillcolor": TINT[k],
        "color": MORSE[k],
        "fontcolor": INK,
        "penwidth": "%.3f" % (NODE_STROKE_PT / scale),
    }


def graphviz_edge(scale=1.0):
    return {"color": GRAPHITE, "penwidth": "%.3f" % (EDGE_STROKE_PT / scale)}




TEXTWIDTH_BP = 432.0
PANEL_PAD_BP = 1.0
GRAPH_FRAME = 'margin="0", pad="0.04"'


def save_panel(fig, path, pad_bp=PANEL_PAD_BP, dpi=600, svg=False):
    """Save fig as a PDF cropped to its ink plus pad_bp on each side.

    The ink box is measured on the PDF itself: fig is saved with a generous box around its tight bounding
    box, rasterized by poppler (pdftoppm), and saved again with the ink box. Whitespace that matplotlib
    counts (empty axes area, invisible spines) is dropped.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np
    from matplotlib.transforms import Bbox
    from PIL import Image

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    probe = Bbox.from_extents(tight.x0 - .1, tight.y0 - .1, tight.x1 + .1, tight.y1 + .1)
    with tempfile.TemporaryDirectory(prefix="panel-") as tmp:
        pdf, stem = Path(tmp) / "probe.pdf", Path(tmp) / "probe"
        fig.savefig(pdf, bbox_inches=probe)
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-singlefile", str(pdf), str(stem)], check=True)
        image = np.asarray(Image.open(stem.with_suffix(".pgm")))
    rows, cols = np.nonzero(image < 255)
    assert rows.size and rows.min() > 0 and cols.min() > 0, path
    assert rows.max() < image.shape[0] - 1 and cols.max() < image.shape[1] - 1, path
    per_in = image.shape[1] / probe.width
    pad = pad_bp / 72
    box = Bbox.from_extents(probe.x0 + cols.min() / per_in - pad, probe.y1 - (rows.max() + 1) / per_in - pad,
                            probe.x0 + (cols.max() + 1) / per_in + pad, probe.y1 - rows.min() / per_in + pad)
    fig.savefig(path, bbox_inches=box, metadata={"CreationDate": None})
    if svg:
        changed = []
        for ax in fig.axes:
            for collection in ax.collections:
                if collection.get_rasterized():
                    changed.append((collection, collection.get_edgecolor(), collection.get_linewidth()))
                    collection.set_rasterized(False)
                    collection.set_edgecolor("face")
                    collection.set_linewidth(.02)
        svg_path = Path(path).with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches=box, format="svg",
                    metadata={"Creator": None, "Date": None})
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines())+"\n")
        for collection, edge, width in changed:
            collection.set_rasterized(True)
            collection.set_edgecolor(edge)
            collection.set_linewidth(width)


def printed_width_bp(layout, panel):
    """Printed width in PDF points from the figure layout."""
    return layout['widths'][panel] * layout['textwidth_bp']


def render_graph(make_dot, pdf, printed_bp, scale=0.6, tries=8):
    """Lay out make_dot(scale) with graphviz dot into pdf, where scale is printed width over natural width.

    make_dot sets strokes for scale (graphviz_node, graphviz_edge). The natural width moves a little with
    the strokes, so iterate until scale agrees with the PDF it produces. Returns scale.
    """
    import subprocess

    for _ in range(tries):
        subprocess.run(["dot", "-Tpdf", "-o", str(pdf)], input=make_dot(scale), text=True, check=True)
        new = printed_bp / pdf_size_bp(pdf)[0]
        if abs(new - scale) < 1e-4:
            return scale
        scale = new
    raise RuntimeError(f"stroke scale of {pdf} did not settle")


def pdf_size_bp(path):
    """Page size of a PDF in bp, from pdfinfo (graphviz keeps it in a compressed object stream)."""
    import subprocess

    info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
    found = re.findall(r"^Page size:\s*([\d.]+) x ([\d.]+) pts", info, flags=re.M)
    assert len(found) == 1, path
    return float(found[0][0]), float(found[0][1])


if __name__ == "__main__":
    for k, (c, t) in enumerate(zip(MORSE, TINT)):
        print(k, c, t)
