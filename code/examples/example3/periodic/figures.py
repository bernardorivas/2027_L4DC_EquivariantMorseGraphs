"""Scientific panels and a standalone draft for the constructed controller."""

from pathlib import Path
from io import BytesIO
import os
import shutil
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix)/".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter

from common import figure_style as style
from common.workflow import read
from example3.figures import native_panels, phase_by_bin, sphere_axis, place
from . import controller as ctl

HERE = Path(__file__).resolve().parent


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=.04, dpi=450, metadata={"CreationDate": None})
    plt.close(fig)


def build(out):
    out = Path(out)
    panels = out/"panels"
    panels.mkdir(exist_ok=True)
    summary, selected = read(out/"summary.json"), read(out/"selection.json")
    plt.rcParams.update(style.mpl_rc() | {"font.size": 10, "axes.spines.top": False,
                                        "axes.spines.right": False, "pdf.fonttype": 42})
    writer, native_writer = PdfWriter(), PdfWriter()
    for row in summary:
        policy, seed = row["policy"], row["seed"]
        folder, name = out/f"{policy}/seed{seed}", f"{policy}-seed{seed}"
        graph_folder = "cmgdb/aggregate" if (folder/"cmgdb/aggregate/results.json").exists() else "cmgdb"
        graph, _ = native_panels(folder, panels, name, graph_folder)
        raw_graph, _ = native_panels(folder, panels, name+"-native")
        for collection, panel, kind in ((writer, graph, "Aggregated"), (native_writer, raw_graph, "Native")):
            title = plt.figure(figsize=(8.5, 5.))
            title.text(.06, .92, f"{policy.capitalize()}, seed {seed}", fontsize=14)
            title.text(.06, .86, f"{kind} interval graph | 128 cells | 1 second | coefficients in F5", fontsize=9)
            buffer = BytesIO()
            title.savefig(buffer, format="pdf")
            plt.close(title)
            page = collection.add_page(PdfReader(buffer).pages[0])
            place(page, panel, (30, 45, 552, 280))
    for collection, name in ((writer, "all-aggregate-graphs.pdf"), (native_writer, "all-native-graphs.pdf")):
        with (out/name).open("wb") as stream:
            collection.write(stream)
    for policy, selection in selected.items():
        seed = selection["selected_seed"]
        folder = out/f"{policy}/seed{seed}"
        membership, basins = phase_by_bin(folder, "cmgdb/aggregate" if (folder/"cmgdb/aggregate/results.json").exists() else "cmgdb")
        with np.load(folder/"diagnostics.npz") as saved:
            d = {name: saved[name] for name in saved.files}
        fig, ax = plt.subplots(figsize=(2.8, 3.0))
        sphere_axis(ax, membership, basins, d["knots"], d["coordinate_values"])
        # Rasterize the contiguous fill only, avoiding PDF polygon seams.
        # The outline, labels, graph edges, and native annotations stay vector.
        for collection in ax.collections:
            collection.set_rasterized(True)
        save(fig, panels/f"{policy}-sphere.pdf")
        z = d["heights"].astype(float)
        image = z.copy()
        for _ in range(10):
            image = np.interp(image, z, d["latitude"])
        u = d["coordinates"]
        learned = np.interp(image, d["knots"], d["coordinate_values"])
        physical = np.interp(ctl.height_flow(z, 1., policy), d["knots"], d["coordinate_values"])
        fig, ax = plt.subplots(figsize=(4.7, 2.7))
        ax.axhline(0, color=style.LIGHT, lw=.8)
        ax.plot(u, physical-u, color=style.GRAPHITE, lw=1.5, label="Physical reference")
        ax.plot(u, learned-u, color=style.MORSE[0], ls="--", lw=1.1, label="Learned")
        ax.set(xlabel="$u$", ylabel="$\\bar f^{10}(u)-u$", xlim=(-1, 1))
        ax.legend(frameon=False, fontsize=8)
        save(fig, panels/f"{policy}-drift.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.8))
    for row in summary:
        policy, seed = row["policy"], row["seed"]
        history = read(out/f"{policy}/seed{seed}/model/history.json")
        color = style.MORSE[0 if policy == "nominal" else 1]
        axes[0].semilogy([r["step"] for r in history], [r["validation"]["objective"] for r in history],
                         color=color, alpha=.45+.2*seed, lw=.9,
                         label=policy.capitalize() if seed == 0 else None)
        axes[1].semilogy(np.arange(33)*.1, row["metrics"]["forecast_rms_by_horizon"], color=color,
                         alpha=.45+.2*seed, lw=.9)
    axes[0].set(xlabel="Training updates", ylabel="Validation objective")
    axes[1].set(xlabel="Prediction time (s)", ylabel="Held-out axis RMS error")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, panels/"learning.pdf")
    physical_folder = out/"controller" if (out/"controller/validation.json").exists() else out.parent/"controller"
    if (physical_folder/"trajectories.npz").exists():
        with np.load(physical_folder/"trajectories.npz") as saved:
            physical = {name: saved[name] for name in saved.files}
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.8))
        time = physical["times"]
        for radius, ls in ((0., "-"), (6., "--")):
            q = physical[f"coning-r{radius}-axis"]
            for index in (0, 2, 10, 15, 20, 25, 30):
                axes[0].plot(time, q[:, index, 2], ls=ls, color=style.MORSE[0 if radius == 0 else 1], lw=.8, alpha=.7)
            eta = np.linalg.norm(physical[f"coning-r{radius}-eta"], axis=-1)
            if radius:
                axes[1].semilogy(time[:121], eta[:121, [0, 10, 20]], color=style.MORSE[1], lw=.9)
                axes[1].semilogy(time[:121], 6*np.exp(-ctl.DAMPING*time[:121]), color=style.GRAPHITE,
                                 lw=1.2, ls="--", label="$6e^{-2t}$")
        axes[0].axhline(ctl.CONE_HEIGHT, color=style.GRAPHITE, lw=.9)
        axes[0].set(xlabel="Time (s)", ylabel="Physical axis height $q_3$", ylim=(-1.04, 1.04))
        axes[1].set(xlabel="Time (s)", ylabel="Rate error $\\|\\eta\\|$", ylim=(1e-10, 10))
        axes[1].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        save(fig, panels/"physical.pdf")
        fig = plt.figure(figsize=(4.2, 3.5))
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        angle = np.linspace(0, 2*np.pi, 200)
        radius = np.sqrt(1-ctl.CONE_HEIGHT**2)
        ax.plot(radius*np.cos(angle), radius*np.sin(angle), np.full_like(angle, ctl.CONE_HEIGHT),
                color=style.MORSE[1], lw=2., zorder=10)
        q = physical["coning-r2.0-axis"]
        indices = [int(np.argmin(np.abs(q[0, :, 2]-z))) for z in (-.8, -.3, .6, .9)]
        for index in indices:
            ax.plot(*q[:121, index].T, lw=.7, color=style.MORSE[0], alpha=.8)
        ax.scatter([0, 0], [0, 0], [-1, 1], color=style.GRAPHITE, s=12)
        ax.set(xlabel="$q_1$", ylabel="$q_2$", zlabel="$q_3$", xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(22, -55)
        ax.grid(False)
        save(fig, panels/"physical-cone.pdf")
    table = []
    for row in summary:
        m = row["metrics"]
        shown = row.get("aggregate_graph", row["graph"])
        table.append(f"{row['policy'].capitalize()} & {row['seed']} & "
            f"{m['forecast_rms_by_horizon'][1]:.2g} & {m['forecast_rms_by_horizon'][10]:.2g} & "
            f"{m['test']['encoder_calibration_rms']:.2g} & {m['one_step_physical_closure_rms']:.2g} & "
            f"{len(shown['nodes'])} & {len(shown['sink_nodes'])} " + r"\\")
    (out/"results-table.tex").write_text(
        r"\begin{tabular}{llrrrrrr}"+"\n"+r"\toprule"+"\n"
        +r"Policy&Seed&$0.1$ s error&$1$ s error&$E$ error&Closure&Nodes&Sinks\\"+"\n"
        +r"\midrule"+"\n"+"\n".join(table)+"\n"+r"\bottomrule"+"\n"+r"\end{tabular}"+"\n")
    figure = plt.figure(figsize=(8.5, 5.))
    figure.text(.06, .9, "Constructed periodic-axis stabilization", fontsize=15)
    figure.text(.06, .83, "Task-aware EMLPs on the invariant sphere subsystem, three seeds per controller", fontsize=10)
    ax = figure.add_axes([.05, .21, .9, .52])
    ax.axis("off")
    rows = [[row["policy"], row["seed"], f"{row['metrics']['forecast_rms_by_horizon'][1]:.2g}",
             f"{row['metrics']['forecast_rms_by_horizon'][10]:.2g}", len(row.get("aggregate_graph", row["graph"])["nodes"]),
             len(row.get("aggregate_graph", row["graph"])["sink_nodes"])] for row in summary]
    table_plot = ax.table(cellText=rows, colLabels=["Policy", "Seed", "0.1 s RMS", "1 s RMS", "Nodes", "Sinks"],
                          loc="center", cellLoc="center")
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1, 1.8)
    figure.text(.06, .1, "Analytic controller proofs are in periodic-orbit-stabilization.tex.\n"
                "Numerical graph agreement is observed. No physical transfer certificate is claimed.", fontsize=9)
    summary_pdf = panels/"summary.pdf"
    figure.savefig(summary_pdf)
    plt.close(figure)
    report = PdfWriter()
    report.append(summary_pdf)
    report.append(out/"all-aggregate-graphs.pdf")
    with (out/"results.pdf").open("wb") as stream:
        report.write(stream)
    selected_rows = {row["policy"]: row for row in summary if row["seed"] == selected[row["policy"]]["selected_seed"]}
    text = []
    for policy, row in selected_rows.items():
        m = row["metrics"]
        fixed = ", ".join(f"{value:.6f}" for value in m["interior_fixed_latitudes"]) or "none"
        text.append(f"The selected {policy} fit is seed {row['seed']}. Its sampled minimum latitude derivative is "
                    f"{m['minimum_latitude_derivative']:.4f}, and its interior fixed latitudes are {fixed}. "
                    f"All {m['supported_interval_cells']} of 128 interval cells contain encoded training or validation samples. "
                    f"Its native learned graph has {len(row['graph']['nodes'])} components and "
                    f"{len(row['graph']['sink_nodes'])} sink components. Its displayed aggregate has "
                    f"{len(row.get('aggregate_graph', row['graph'])['nodes'])} components. "
                    f"The physical reference in its learned coordinate has {len(row['matched_reference']['nodes'])} raw components.\n")
    (out/"results-text.tex").write_text("\n".join(text))
    (out/"selected-panels.tex").write_text("\n".join(
        f"\\def\\{policy}graph{{panels/{policy}-seed{selection['selected_seed']}-morse-graph.pdf}}\n"
        f"\\def\\{policy}cells{{panels/{policy}-seed{selection['selected_seed']}-morse-cells.pdf}}"
        for policy, selection in selected.items())
        + f"\n\\def\\fitcount{{{len(summary)}}}\n\\def\\seedcount{{{len(summary)//len(selected)}}}\n")
    if set(selected) != {"nominal", "coning"}:
        print("Prepared numeric reports; both policies are required for the full mathematical draft.", flush=True)
        return out/"results.pdf"
    shutil.copyfile(HERE/"draft.tex", out/"periodic-orbit-stabilization.tex")
    if shutil.which("latexmk") is None:
        print("Prepared the mathematical draft source; latexmk is required to compile its PDF.", flush=True)
        return out/"results.pdf"
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                    "periodic-orbit-stabilization.tex"], cwd=out, check=True,
                   stdout=(out/"latex-build.log").open("w"), stderr=subprocess.STDOUT)
    return out/"periodic-orbit-stabilization.pdf"
