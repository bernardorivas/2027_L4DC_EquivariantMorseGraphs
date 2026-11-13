"""Render CMGDB cells and graphs for the explicit and fitted sphere maps."""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
_MPL_TMP = tempfile.TemporaryDirectory(prefix="sphere-morse-mpl-")
os.environ["MPLCONFIGDIR"] = _MPL_TMP.name

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_style as style

import CMGDB

FOLDER = "sphere/morse"
LAYOUT = HERE / "layout.json"
RESULTS = HERE / "data/sphere"
PANELS = (("explicit", "exact-K128"), ("learned", "seed0-K128"))
NODE = {"T-": 0, "S": 1, "T+": 2, "E": 3, "N": 4}
HEIGHT = {"T-": -0.4, "S": -1.0, "T+": 0.4, "E": 0.0, "N": 1.0}


def check_nodes(folder):
    """Node ids of the saved computation agree with NODE, and each set sits at its height."""
    nodes = json.loads((folder / "native-result.json").read_text())["nodes"]
    names = {row["geometric_name"]: row["node"] for row in nodes}
    assert names == NODE, (folder, names)
    for row in nodes:
        z = HEIGHT[row["geometric_name"]]
        lo = min(a for a, _ in row["boxes"])
        hi = max(b for _, b in row["boxes"])
        assert lo - .03 <= z <= hi + .03, (folder, row)


def cells(folder, path):
    fig, ax = CMGDB.PlotMorseSets1D(str(folder / "morse-sets.csv"), clist=style.MORSE, fig_w=7, fig_h=1.5,
                                    xlim=[-1.02, 1.02], xlabel="$z$", axis_arrow=False, height=.20,
                                    label_sets=True, show=False)
    for text in ax.texts:
        text.set_color(style.INK)
    ax.spines["bottom"].set_edgecolor(style.GRAPHITE)
    ax.tick_params(color=style.GRAPHITE, labelcolor=style.INK)
    ax.xaxis.label.set_color(style.INK)
    style.save_panel(fig, path)
    plt.close(fig)


NODE_LINE = re.compile(r'^(\d+) \[label="([^"]*)", shape=(\w+), style=filled, fillcolor="#[0-9a-fA-F]{8}", '
                       r'margin="([^"]*)"\];$')


def restyle(dot, scale):
    """The saved DOT with node attributes style.graphviz_node(k, scale) and edges style.graphviz_edge(scale).

    Label, shape, margin, the default font and the rank constraints are kept; the graph gets
    style.GRAPH_FRAME.
    """
    def attrs(d):
        return ", ".join(f'{key}="{value}"' for key, value in d.items())

    out, seen = [], set()
    for line in dot.splitlines():
        m = NODE_LINE.match(line.strip())
        if m:
            k, label, shape, margin = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            out.append(f'{k} [label="{label}", shape={shape}, {attrs(style.graphviz_node(k, scale))}, '
                       f'margin="{margin}"];')
            seen.add(k)
            continue
        assert "fillcolor" not in line, line
        out.append(line)
        if line.strip() == "digraph {":
            out.append(f"graph [{style.GRAPH_FRAME}];")
            out.append(f"edge [{attrs(style.graphviz_edge(scale))}];")
    assert seen == set(NODE.values()), seen
    return "\n".join(out) + "\n"


def graph(folder, path, printed_bp):
    """Render the restyled DOT with strokes for its printed width."""
    dot = (folder / "morse-graph.dot").read_text()
    return style.render_graph(lambda scale: restyle(dot, scale), path, printed_bp)


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="figures directory; writes OUT/" + FOLDER)
    parser.add_argument("--layout", type=Path, default=LAYOUT)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    args = parser.parse_args()

    layout = json.loads(args.layout.read_text())
    RESULTS = args.data / "sphere"
    plt.rcParams.update(style.mpl_rc())
    out = args.out / FOLDER
    out.mkdir(parents=True, exist_ok=True)
    for prefix, result in PANELS:
        folder = RESULTS / result
        check_nodes(folder)
        cells(folder, out / f"{prefix}-cells.pdf")
        printed = style.printed_width_bp(layout, f"{FOLDER}/{prefix}-graph.pdf")
        scale = graph(folder, out / f"{prefix}-graph.pdf", printed)
        print(f"{out / prefix}-{{cells,graph}}.pdf (graph printed at scale {scale:.4f})")


if __name__ == "__main__":
    main()
