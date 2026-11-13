"""Render original and learned reduced-attitude panels from frozen inputs."""

from argparse import ArgumentParser
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix)/".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import CMGDB

from . import figure_style as style
from .workflow import read, verify_inputs, write
from .sphere_panels import sphere_axis
from .aggregate_interval import from_saved, save as save_aggregate

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent/"example3/periodic/reference"
DEFAULT_LAYOUT = HERE/"layout.json"


def roles(record, policy):
    result = {}
    for row in record["nodes"]:
        a, b = row["interval"]
        if a == -1.:
            role = "S"
        elif b == 1.:
            role = "N"
        elif row["node"] in record["sink_nodes"]:
            role = "C"
        else:
            role = "A"
        assert role not in result.values()
        result[row["node"]] = role
    assert set(result.values()) == ({"S", "N"} if policy == "nominal" else
                                    ({"S", "C", "N"} if len(result) == 3 else {"S", "C", "A", "N"}))
    return result


def sphere_indices(role, policy):
    if role == "A":
        return ["0", "0", "0"]
    if role == "C":
        return ["x-1", "x-1", "0"]
    if role == "N" and policy == "nominal":
        return ["x-1", "0", "0"]
    return ["0", "0", "x-1"]


def graph_source(record, policy, reconstructed, scale):
    names = roles(record, policy)
    edge = style.graphviz_edge(scale)
    lines = ["digraph {", f"graph [{style.GRAPH_FRAME}, nodesep=.15, ranksep=.25];",
             'node [fontname="Times-Roman", fontsize=15, shape=ellipse, margin=".065,.04"];',
             f'edge [color="{edge["color"]}", penwidth="{edge["penwidth"]}", arrowsize=.8];']
    for row in record["nodes"]:
        node, role = row["node"], names[row["node"]]
        annotations = sphere_indices(role, policy) if reconstructed else row["annotations"]
        label = role+" : ("+", ".join(annotations)+")"
        attrs = ", ".join(f'{key}="{value}"' for key, value in style.graphviz_node(node, scale).items())
        lines.append(f'{node} [label="{label}", {attrs}];')
    lines += [f"{a} -> {b};" for a, b in record["edges"]]
    sources = set(names)-{b for _, b in record["edges"]}
    lines.append("{rank=same; "+"; ".join(map(str, sorted(sources)))+";}")
    return "\n".join(lines+["}"])+"\n"


def sphere_panel(phase, coordinate, path, svg):
    count = len(phase["boxes"])
    order = np.clip(np.floor((phase["boxes"].mean(axis=1)+1)*count/2).astype(int), 0, count-1)
    assert np.array_equal(np.sort(order), np.arange(count))
    membership, basins = np.empty(count, int), np.empty(count, int)
    membership[order], basins[order] = phase["membership"], phase["basin_node"]
    fig, ax = plt.subplots(figsize=(2.8, 3.))
    sphere_axis(ax, membership, basins, coordinate["knots"], coordinate["values"])
    for collection in ax.collections:
        collection.set_rasterized(True)
    style.save_panel(fig, path, dpi=600, svg=svg)
    plt.close(fig)


def interval_panel(csv, path, variable, svg):
    fig, ax = CMGDB.PlotMorseSets1D(str(csv), clist=style.MORSE, fig_w=5.8, fig_h=1.15,
                                  xlim=[-1.04, 1.04], xlabel=f"${variable}$", fontsize=11,
                                  axis_arrow=False, height=.16, label_sets=False, show=False)
    ax.set_yticks([])
    style.save_panel(fig, path, svg=svg)
    plt.close(fig)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def render(data, out, layout, policies=("nominal", "coning"), record_evidence=True, svg=True):
    presentation = data/"presentation"
    verify_inputs(presentation)
    selection = read(presentation/"selection.json")
    plt.rcParams.update(style.mpl_rc() | {"font.family": "serif", "font.serif": ["STIXGeneral"],
                                         "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42,
                                         "svg.fonttype": "none"})
    evidence = {}
    for policy in policies:
        target = out/"attitude"/policy
        target.mkdir(parents=True, exist_ok=True)
        seed = selection[policy]["selected_seed"]
        for side in ("original", "learned"):
            if side == "original":
                record = read(presentation/f"{policy}-original-graph.json")
                phase = load_arrays(presentation/f"{policy}-original-phase.npz")
                coordinate = dict(knots=np.array([-1., 1.]), values=np.array([-1., 1.]))
            else:
                folder = data/policy/f"seed{seed}"
                record = read(folder/"cmgdb/results.json")
                phase = load_arrays(folder/"cmgdb/phase.npz")
                coordinate = load_arrays(presentation/f"{policy}-coordinate.npz")
                metrics = read(folder/"metrics.json")
                assert metrics["minimum_latitude_derivative"] > 0
                assert metrics["endpoint_images"] == [-1., 1.]
                assert len(metrics["interior_fixed_latitudes"]) == (0 if policy == "nominal" else 1)
            for row in record["nodes"]:
                assert np.array_equal(np.flatnonzero(phase["membership"] == row["node"]), row["cell_ids"])
            native_count = len(record["nodes"])
            record, phase = from_saved(record, phase, policy == "coning")
            with tempfile.TemporaryDirectory(prefix="attitude-aggregate-") as temporary:
                aggregate_dir = Path(temporary)
                save_aggregate(record, phase, aggregate_dir)
                sphere_panel(phase, coordinate, target/f"{side}-sphere.pdf", svg)
                interval_panel(aggregate_dir/"morse-sets.csv", target/f"{side}-interval.pdf",
                               "z" if side == "original" else "u", svg)
            for reconstructed, suffix in ((False, "quotient"), (True, "reconstructed")):
                relative = f"attitude/{policy}/{side}-{suffix}.pdf"
                width = layout["widths"].get(relative, .455)*layout["textwidth_bp"]
                scale = style.render_graph(lambda scale: graph_source(record, policy, reconstructed, scale),
                                           target/f"{side}-{suffix}.pdf", width)
                if svg:
                    subprocess.run(["dot", "-Tsvg", "-o", str(target/f"{side}-{suffix}.svg")],
                                   input=graph_source(record, policy, reconstructed, scale), text=True, check=True)
            evidence[f"{policy}/{side}"] = dict(nodes=record["nodes"], edges=record["edges"],
                aggregation=record["aggregation"], raw_components=native_count,
                sphere_indices={roles(record, policy)[row["node"]]: sphere_indices(roles(record, policy)[row["node"]], policy)
                                for row in record["nodes"]},
                index_status="analytic sphere indices" if side == "original" else
                             "numerical reconstruction on the specified latent sphere; no physical transfer certificate")
    if record_evidence:
        write(out/"attitude/verification.json", evidence)
    return evidence


def build_presentation(out):
    package = HERE.parent/"example3/periodic"
    shutil.copy2(package/"presentation.tex", out/"presentation.tex")
    shutil.copy2(package/"presentation-figure.tex", out/"presentation-figure.tex")
    if shutil.which("latexmk"):
        with (out/"presentation-build.log").open("w") as log:
            subprocess.run(["latexmk", "-pdf", "-halt-on-error", "-interaction=nonstopmode", "presentation.tex"],
                           cwd=out, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(out/"presentation.pdf", flush=True)
    else:
        print(out/"presentation.tex", flush=True)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--presentation", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=("nominal", "coning"), default=["nominal", "coning"])
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--no-svg", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    render(args.data, args.out, read(args.layout), args.policies, not args.no_record, not args.no_svg)
    if args.figure:
        shutil.copy2(HERE.parent/"example3/periodic/presentation-figure.tex", args.out/"attitude-figure.tex")
    if args.presentation:
        build_presentation(args.out)


if __name__ == "__main__":
    main()
