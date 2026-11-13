"""Render data-map basins, raw graphs, and physical sampling comparisons."""

import argparse
from pathlib import Path

from .training import read, write
from .figures import basin_page, cell_axis, foot, image_axis, record, save
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def density_page(pdf, run, seed, depth, sizes, page):
    fig, axes = plt.subplots(2, len(sizes), figsize=(11.7, 8.3), squeeze=False,
                             gridspec_kw={"height_ratios": [1.1, 1.4]})
    for column, size in enumerate(sizes):
        folder = run / f"seed{seed}/S{size}/K{2**depth}"
        summary = read(folder / "sampling-summary.json")
        title = f"{size} x {size} physical section + 4 controls\n{summary['supported_cells']} supported latent cells"
        cell_axis(axes[0, column], folder, title, "phase")
        image_axis(axes[1, column], folder / "native-morse-graph.png",
                   f"Native graph: {len(record(folder)['nodes'])} nodes")
    fig.suptitle(f"Example 2, seed {seed}: physical sampling density at {2**depth} x {2**depth} latent cells", fontsize=14)
    fig.tight_layout(rect=(.015, .09, .985, .94))
    fig.text(.04, .05, "A node number without a tuple means CMGDB returned no index annotation; it does not mean a zero index.", fontsize=8)
    foot(fig, page, "BoxMapData uses only (E(x), f(E(x))) pairs. Empty cells map outside; interpolation and padding are disabled.")
    pdf.savefig(fig)
    save(fig, run / "figures" / f"seed{seed}-sampling")
    plt.close(fig)


def summary_page(pdf, run, selection, depth, primary, page):
    rows = read(run / "data-map-summary.json")
    rows = [row for row in rows if row["grid"] == 2**depth]
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.05, .95, "Example 2: sampling density and regions of attraction", fontsize=16, weight="bold")
    fig.text(.05, .90, f"Latent grid: {2**depth} x {2**depth}. Selected model: seed {selection['selected_seed']} by validation. Primary physical grid: {primary} x {primary}.", fontsize=10)
    cells = []
    for row in rows:
        counts = (row["basin_counts"] or {}).get("active", {})
        cells.append([row["seed"], row["source_size"], row["nodes"], row["supported_cells"],
                      row["minimum_sources_per_supported_cell"], counts.get("minus", "n/a"),
                      counts.get("plus", "n/a"), counts.get("unresolved", "n/a"), counts.get("unknown", "n/a")])
    ax = fig.add_axes([.05, .47, .90, .37])
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["Seed", "Sources/axis", "Nodes", "Supported", "Min. samples", "A- cells", "A+ cells", "Unresolved", "Unknown"],
                     loc="center", cellLoc="center", colWidths=[.06, .13, .07, .11, .12, .10, .10, .12, .10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#eeeeee")
    fig.text(.07, .40, "The networks and validation ranking are unchanged. Physical section samples are independent of the training splits.\n"
             "Every source lies in E(X). Targets are f(E(x)), so this computes the learned dynamics.\n"
             "Each source grid includes its boundary, both axes, and four physical equilibrium-orbit controls.", fontsize=10, linespacing=1.6)
    fig.text(.07, .27, "A basin label requires a unique reachable sink and no path to an unsupported cell.\n"
             "Unknown cells are included in the unresolved count. An empty cell is not evidence that it lies outside E(X).\n"
             "Source count and finite-graph transitions are checked for inclusion as the nested sample grids grow.", fontsize=10, linespacing=1.6)
    fig.text(.07, .14, "These are numerical image hulls, not validated enclosures. Physical panels sample preimages under E.\n"
             "Native quotient annotations remain separate from original-space indices. Every raw component is retained.\n"
             "An absent native index annotation does not mean a zero index.", fontsize=10, linespacing=1.6)
    foot(fig, page)
    pdf.savefig(fig)
    save(fig, run / "figures/summary")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    config = read(run / "requested-config.json")
    selection = read(run / "selection.json")
    depth = config["analysis"]["primary_depth"]
    settings = config["analysis"]["data_map"]
    sizes, primary = settings["source_sizes"], settings["primary_source_size"]
    seeds = [row["seed"] for row in selection["ranking"]]
    (run / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    with PdfPages(run / "results.pdf", metadata={"Title": "Example 2: BoxMapData results", "CreationDate": None}) as pdf:
        page = 1
        for seed in seeds:
            basin_page(pdf, run, seed, depth, page, run / f"seed{seed}/S{primary}/K{2**depth}", primary)
            page += 1
            density_page(pdf, run, seed, depth, sizes, page)
            page += 1
        summary_page(pdf, run, selection, depth, primary, page)
    with PdfPages(run / "regions-of-attraction.pdf", metadata={"Title": "Example 2: data-map attraction regions", "CreationDate": None}) as pdf:
        for page, seed in enumerate(seeds, 1):
            basin_page(pdf, run, seed, depth, page, run / f"seed{seed}/S{primary}/K{2**depth}", primary)
    with PdfPages(run / "sampling-study.pdf", metadata={"Title": "Example 2: physical sampling density", "CreationDate": None}) as pdf:
        for page, seed in enumerate(seeds, 1):
            density_page(pdf, run, seed, depth, sizes, page)
    write(run / "figure-summary.json", {"primary_depth": depth, "primary_source_size": primary,
          "results_pages": 2*len(seeds)+1, "selected_seed": selection["selected_seed"],
          "physical_preimages": "sampled under E; no decoder used"})


if __name__ == "__main__":
    main()
