"""Recompute the released sphere graph and sampled pullback check.

The neural maps are evaluated in the pinned EMLP environment.  CMGDB is run in
the notebook environment.  The public entry point, :func:`replay`, verifies the
checkpoint hash and checks every reported value against the release manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "reproducibility" / "sphere-passing"
DEFAULT_OUTPUT = ROOT / "build" / "notebooks" / "sphere-passing"


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker(checkpoint: Path, output: Path):
    """Evaluate the learned maps and save arrays for the CMGDB environment."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    sys.path.insert(0, str(ROOT / "src"))
    from quotient_runtime import load_quotient

    _, encode, decode, advance = load_quotient(checkpoint)

    def batch(function, values, size=2048):
        return np.concatenate(
            [np.asarray(function(values[i : i + size])) for i in range(0, len(values), size)]
        )

    def iterate(values, steps=10):
        result = values
        for _ in range(steps):
            result = batch(advance, result)
        return result

    def meridian(z):
        return np.column_stack(
            (np.sqrt(np.maximum(0.0, 1.0 - z * z)), np.zeros_like(z), z)
        )

    # Fixed points of the learned ten-step map in the latent interval.
    root_grid = np.linspace(-1.0, 1.0, 8193)
    root_values = iterate(root_grid[:, None])[:, 0] - root_grid

    def fixed_point_equation(q):
        return float(iterate(np.array([[q]]))[0, 0] - q)

    roots = [-1.0]
    for left, right, f_left, f_right in zip(
        root_grid[:-1], root_grid[1:], root_values[:-1], root_values[1:]
    ):
        if f_left * f_right < 0.0:
            roots.append(brentq(fixed_point_equation, left, right, xtol=1e-13))
    roots.append(1.0)
    roots = sorted(set(roots))

    # Dense cell images for the CMGDB graph.
    edges = np.linspace(-1.0, 1.0, 129)
    graph_points = np.array(
        [np.linspace(left, right, 1025) for left, right in zip(edges[:-1], edges[1:])]
    )
    graph_values = iterate(graph_points.reshape(-1, 1))[:, 0].reshape(graph_points.shape)
    graph_images = np.column_stack(
        (
            np.maximum(-1.0, np.nextafter(graph_values.min(axis=1) - 1e-10, -np.inf)),
            np.minimum(1.0, np.nextafter(graph_values.max(axis=1) + 1e-10, np.inf)),
        )
    )

    # Independently sampled cell images used for the block tolerances.
    block_points = np.array(
        [np.linspace(left, right, 65) for left, right in zip(edges[:-1], edges[1:])]
    )
    block_values = iterate(block_points.reshape(-1, 1))[:, 0].reshape(block_points.shape)
    block_images = np.column_stack(
        (
            np.maximum(-1.0, np.nextafter(block_values.min(axis=1) - 1e-10, -np.inf)),
            np.minimum(1.0, np.nextafter(block_values.max(axis=1) + 1e-10, np.inf)),
        )
    )

    # Physical meridian samples, refined near both poles.
    theta_base = np.linspace(0.0, np.pi, 65537)
    theta_poles = np.geomspace(1e-9, 1e-2, 2000)
    theta = np.sort(np.concatenate((theta_base, theta_poles, np.pi - theta_poles)))
    z = np.cos(theta)
    interior = (theta > 0.0) & (theta < np.pi)
    w0 = -np.log(np.tan(theta[interior] / 2.0))
    solution = solve_ivp(
        lambda _time, w: np.tanh(w) * (0.16 - np.tanh(w) ** 2),
        (0.0, 10.0),
        w0,
        method="DOP853",
        rtol=2e-13,
        atol=2e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    true_height = z.copy()
    true_height[interior] = np.tanh(solution.y[:, -1])

    encoded = batch(encode, meridian(z))[:, 0]
    encoded_true_next = batch(encode, meridian(true_height))[:, 0]
    predicted = iterate(encoded[:, None])[:, 0]
    residual = np.abs(predicted - encoded_true_next)

    # Reader-facing prediction curve: decode the ten-step latent prediction.
    plot_z = np.linspace(-1.0, 1.0, 2001)
    plot_encoded = batch(encode, meridian(plot_z))
    plot_predicted_height = batch(decode, iterate(plot_encoded))[:, 0]

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "model-arrays.npz",
        edges=edges,
        roots=np.asarray(roots),
        graph_images=graph_images,
        block_images=block_images,
        physical_z=z,
        encoded_z=encoded,
        physical_residual=residual,
        plot_z=plot_z,
        plot_encoded=plot_encoded[:, 0],
        plot_predicted_height=plot_predicted_height,
    )


def _blocks(images, edges):
    """Select cells that can reach exactly one attracting Morse component."""
    adjacency = (images[:, 0, None] <= edges[1:]) & (images[:, 1, None] >= edges[:-1])
    reachability = adjacency.copy()
    np.fill_diagonal(reachability, True)
    for middle in range(len(images)):
        reachability |= reachability[:, middle, None] & reachability[middle, None, :]

    _, labels = connected_components(adjacency, directed=True, connection="strong")
    components = [np.where(labels == label)[0] for label in np.unique(labels)]
    components = [
        component
        for component in components
        if len(component) > 1 or adjacency[component[0], component[0]]
    ]
    sinks = [
        component
        for component in components
        if not any(
            reachability[np.ix_(component, other)].any()
            for other in components
            if not np.array_equal(component, other)
        )
    ]
    selected = []
    for component in sorted(sinks, key=lambda values: values[0]):
        other = np.concatenate(
            [values for values in components if not np.array_equal(component, values)]
        )
        selected.append(
            np.where(
                reachability[:, component].any(axis=1)
                & ~reachability[:, other].any(axis=1)
            )[0]
        )
    return selected, len(components)


def _plot_prediction(output, arrays):
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12}
    ):
        fig, axis = plt.subplots(figsize=(4.4, 3.5), layout="constrained")
        axis.plot([-1, 1], [-1, 1], color="0.82", linewidth=1.0)
        axis.plot(
            arrays["plot_z"],
            arrays["plot_predicted_height"],
            color="0.12",
            linewidth=2.0,
        )
        colors = ["#D55E00", "#0072B2", "#CC79A7", "#009E73", "#E69F00"]
        encoded = arrays["plot_encoded"]
        if encoded[0] < encoded[-1]:
            physical_roots = np.interp(arrays["roots"], encoded, arrays["plot_z"])
        else:
            physical_roots = np.interp(
                arrays["roots"], encoded[::-1], arrays["plot_z"][::-1]
            )
        for z_value, color in zip(physical_roots, colors):
            y_value = np.interp(z_value, arrays["plot_z"], arrays["plot_predicted_height"])
            axis.scatter([z_value], [y_value], s=34, color=color, zorder=3)
        axis.set(
            xlim=(-1.03, 1.03),
            ylim=(-1.03, 1.03),
            xticks=[-1, -0.4, 0, 0.4, 1],
            yticks=[-1, 0, 1],
            xlabel="$z$",
            ylabel="Predicted height after ten steps",
        )
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"predicted-height.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _inverse_height(q, arrays):
    """Invert the released monotone encoder along the reference meridian."""
    encoded = arrays["plot_encoded"]
    heights = arrays["plot_z"]
    if encoded[0] < encoded[-1]:
        return float(np.interp(q, encoded, heights))
    return float(np.interp(q, encoded[::-1], heights[::-1]))


def _plot_reconstruction(output, arrays, aggregate):
    """Plot exact encoder preimages and the specified sphere-lift indices."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyArrowPatch

    from sphere_pullback import draw_transparent_sphere

    palette = {
        0: "#D55E00",
        1: "#0072B2",
        2: "#CC79A7",
        3: "#009E73",
        4: "#E69F00",
    }
    names = {0: "S", 1: "A-", 2: "C", 3: "A+", 4: "N"}
    reverse_group = {
        int(native): int(group)
        for group, members in aggregate["groups"].items()
        for native in members
    }
    native_rows = np.loadtxt(
        output / "graph" / "rectangle-128-cells.csv", delimiter=",", ndmin=2
    )
    sphere_rows = []
    for q_lo, q_hi, native_node in native_rows:
        group = reverse_group[int(native_node)]
        sphere_rows.append(
            {
                "node": group,
                "z": sorted(
                    (_inverse_height(float(q_lo), arrays), _inverse_height(float(q_hi), arrays))
                ),
            }
        )

    positions = {
        0: (-1.4, 0.8),
        2: (0.0, 0.8),
        4: (1.4, 0.8),
        1: (-0.7, -0.6),
        3: (0.7, -0.6),
    }
    index_labels = {
        0: r"$S:\;(0,0,x-1)$",
        1: r"$A_{-}:\;(x-1,x-1,0)$",
        2: r"$C:\;(0,x-1,x-1)$",
        3: r"$A_{+}:\;(x-1,x-1,0)$",
        4: r"$N:\;(0,0,x-1)$",
    }

    with plt.rc_context(
        {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12}
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
        sphere_axis, graph_axis = axes
        draw_transparent_sphere(sphere_axis, sphere_rows, palette)
        sphere_axis.set(aspect="equal", xlim=(-1.45, 1.55), ylim=(-1.2, 1.2))
        sphere_axis.axis("off")
        sphere_axis.set_title(r"Encoder preimages in $S^2$")

        label_targets = {
            0: ((-0.12, -0.99), (-0.58, -1.08)),
            1: ((0.78, -0.52), (1.16, -0.78)),
            2: ((0.98, 0.01), (1.34, 0.0)),
            3: ((0.78, 0.5), (1.16, 0.72)),
            4: ((0.0, 0.98), (0.0, 1.12)),
        }
        for node, (target, text_position) in label_targets.items():
            label = {
                0: "$S$",
                1: "$A_{-}$",
                2: "$C$",
                3: "$A_{+}$",
                4: "$N$",
            }[node]
            sphere_axis.annotate(
                label,
                xy=target,
                xytext=text_position,
                color=palette[node],
                ha="center",
                va="center",
                arrowprops={
                    "arrowstyle": "->",
                    "color": palette[node],
                    "linewidth": 1.0,
                    "connectionstyle": "arc3,rad=0.22" if node in (1, 2, 3) else "arc3",
                },
            )

        for source, target in aggregate["edges"]:
            start = positions[int(source)]
            end = positions[int(target)]
            graph_axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.1,
                    color="0.22",
                    shrinkA=34,
                    shrinkB=38,
                    zorder=1,
                )
            )
        for node, (x_value, y_value) in positions.items():
            width = 1.18 if node in (0, 2, 4) else 1.4
            graph_axis.add_patch(
                Ellipse(
                    (x_value, y_value),
                    width,
                    0.48,
                    facecolor=palette[node],
                    edgecolor=palette[node],
                    alpha=0.34,
                    linewidth=1.2,
                    zorder=2,
                )
            )
            graph_axis.text(
                x_value,
                y_value,
                index_labels[node],
                ha="center",
                va="center",
                fontsize=10.5,
                zorder=3,
            )
        graph_axis.set(xlim=(-2.1, 2.1), ylim=(-1.05, 1.25), aspect="equal")
        graph_axis.axis("off")
        graph_axis.set_title(r"Reconstructed Conley--Morse graph in $S^2$")
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"reconstruction.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def replay(checkpoint=DEFAULT_CHECKPOINT, output=DEFAULT_OUTPUT):
    """Recompute and verify the released Example 5.1 numerical evidence."""
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    manifest = _read_json(checkpoint / "manifest.json")
    actual_hash = _sha256(checkpoint / "models.npz")
    if actual_hash != manifest["checkpoint_sha256"]:
        raise AssertionError(f"checkpoint hash {actual_hash} does not match the manifest")

    emlp_python = ROOT / ".venv-emlp" / "bin" / "python"
    if not emlp_python.is_file():
        raise FileNotFoundError(f"missing pinned EMLP interpreter: {emlp_python}")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(ROOT / "build" / "matplotlib"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(
        [
            str(emlp_python),
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )

    with np.load(output / "model-arrays.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}

    from quotient_domain import native_run

    roots = [dict(id=index, q=[float(value)]) for index, value in enumerate(arrays["roots"])]
    graph_dir = output / "graph"
    graph_dir.mkdir(exist_ok=True)
    graph = native_run(
        graph_dir,
        128,
        arrays["graph_images"],
        arrays["graph_images"],
        np.ones(128, dtype=bool),
        np.array([-1.0]),
        np.array([1.0]),
        roots,
        True,
    )

    selected, component_count = _blocks(arrays["block_images"], arrays["edges"])
    if len(selected) != 2:
        raise AssertionError(f"expected two attracting blocks, found {len(selected)}")
    checks = {}
    for label, cells in zip(("A-", "A+"), selected):
        if not np.all(np.diff(cells) == 1):
            raise AssertionError(f"{label} is not a contiguous block")
        lo = float(arrays["edges"][cells[0]])
        hi = float(arrays["edges"][cells[-1] + 1])
        tolerance = float(
            min(
                arrays["block_images"][cells, 0].min() - lo,
                hi - arrays["block_images"][cells, 1].max(),
            )
        )
        mask = (arrays["encoded_z"] >= lo) & (arrays["encoded_z"] <= hi)
        indices = np.flatnonzero(mask)
        worst = int(indices[np.argmax(arrays["physical_residual"][mask])])
        residual = float(arrays["physical_residual"][worst])
        checks[label] = dict(
            cells=int(len(cells)),
            first_cell=int(cells[0]),
            last_cell=int(cells[-1]),
            interval=[lo, hi],
            tolerance=tolerance,
            maximum_sampled_residual=residual,
            worst_sampled_height=float(arrays["physical_z"][worst]),
            passes=bool(residual < tolerance),
        )

    expected = manifest["expected"]
    np.testing.assert_allclose(arrays["roots"], expected["fixed_points"], rtol=0, atol=5e-10)
    if graph["rectangle"]["components"] != expected["native_morse_components"]:
        raise AssertionError("native Morse component count changed")
    aggregate = graph["aggregate"]
    if len(aggregate["groups"]) != expected["grouped_morse_components"]:
        raise AssertionError("grouped Morse component count changed")
    observed_groups = {
        str(group): [int(member) for member in members]
        for group, members in aggregate["groups"].items()
    }
    if observed_groups != expected["grouped_members"]:
        raise AssertionError("grouped Morse component membership changed")
    if aggregate["edges"] != expected["grouped_edges"] or not aggregate["matches_analytical"]:
        raise AssertionError("grouped Conley--Morse graph changed")
    if component_count != expected["native_morse_components"]:
        raise AssertionError("block graph component count changed")
    for label, observed in checks.items():
        pinned = expected["blocks"][label]
        for key in ("cells", "first_cell", "last_cell", "interval"):
            if observed[key] != pinned[key]:
                raise AssertionError(f"{label} {key} changed: {observed[key]} != {pinned[key]}")
        np.testing.assert_allclose(
            [observed["tolerance"], observed["maximum_sampled_residual"]],
            [pinned["tolerance"], pinned["maximum_sampled_residual"]],
            rtol=0,
            atol=5e-10,
        )
        if not observed["passes"]:
            raise AssertionError(f"{label} residual is not below its tolerance")

    verification = dict(
        checkpoint_sha256=actual_hash,
        map_iterations=10,
        grid_cells=128,
        graph_image_samples_per_cell=1025,
        block_image_samples_per_cell=65,
        physical_residual_samples=int(len(arrays["physical_z"])),
        fixed_points=arrays["roots"].tolist(),
        native_morse_components=graph["rectangle"]["components"],
        grouped_graph=aggregate,
        blocks=checks,
        sampled_estimates=True,
        uniform_bound=False,
    )
    _write_json(output / "verification.json", verification)
    _plot_prediction(output, arrays)
    _plot_reconstruction(output, arrays, aggregate)
    return verification


def show_dynamics(output=DEFAULT_OUTPUT):
    """Display the learned ten-step map and its recomputed grouped graph."""
    from IPython.display import HTML, Image, display

    output = Path(output)
    display(Image(filename=str(output / "predicted-height.png")))
    display(Image(filename=str(output / "graph" / "aggregate-128.png")))


def show_pullback_check(verification, output=DEFAULT_OUTPUT):
    """Display the sampled residual--tolerance comparison for both blocks."""
    from IPython.display import HTML, display
    rows = []
    for label in ("A-", "A+"):
        check = verification["blocks"][label]
        rows.append(
            "<tr>"
            f"<td>{label}</td><td>{check['cells']}</td>"
            f"<td>{check['tolerance']:.5f}</td>"
            f"<td>{check['maximum_sampled_residual']:.5f}</td>"
            f"<td>{'yes' if check['passes'] else 'no'}</td></tr>"
        )
    display(
        HTML(
            "<table><thead><tr><th>Block</th><th>Cells</th><th>Tolerance</th>"
            "<th>Maximum sampled residual</th><th>Residual below tolerance</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<p>The maxima and tolerances are sampled numerical estimates, not uniform error bounds.</p>"
        )
    )


def show_reconstruction(output=DEFAULT_OUTPUT):
    """Display encoder preimages and the reconstructed sphere indices."""
    from IPython.display import Image, display

    display(Image(filename=str(Path(output) / "reconstruction.png")))


def show(verification, output=DEFAULT_OUTPUT):
    """Display all verified Example 5.1 results in a notebook."""
    show_dynamics(output)
    show_pullback_check(verification, output)
    show_reconstruction(output)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.worker:
        _worker(arguments.checkpoint, arguments.output)
    else:
        print(json.dumps(replay(arguments.checkpoint, arguments.output), indent=2))


if __name__ == "__main__":
    _main()
