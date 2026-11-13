"""Audit finite graph structure and the official EMLP implementation."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from common.workflow import read, write
from example3.periodic import model as learning


def graph_audit(folder):
    result = read(folder/"cmgdb/results.json")
    with np.load(folder/"cmgdb/phase.npz") as saved:
        phase = {name: saved[name] for name in saved.files}
    offsets, targets = phase["csr_offsets"].astype(int), phase["csr_targets"].astype(int)
    count = len(phase["boxes"])
    matrix = csr_matrix((np.ones(len(targets)), targets, offsets), shape=(count, count))
    _, components = connected_components(matrix, directed=True, connection="strong")
    cyclic = []
    for component in np.unique(components):
        ids = np.flatnonzero(components == component)
        if len(ids) > 1 or matrix[ids[0], ids[0]]:
            cyclic.append(tuple(map(int, ids)))
    assert sorted(cyclic) == sorted(tuple(row["cell_ids"]) for row in result["nodes"])
    native_membership = phase["membership"]
    def reachable_cells(source):
        seen, pending = set(), [source]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(map(int, targets[offsets[current]:offsets[current+1]]))
        return seen
    reachable_nodes = []
    for source in range(count):
        reachable_nodes.append({int(native_membership[i]) for i in reachable_cells(source) if native_membership[i] >= 0})
    singleton = np.array([next(iter(values)) if len(values) == 1 else -1 for values in reachable_nodes])
    basins = np.where(np.isin(singleton, result["sink_nodes"]), singleton, -1)
    assert np.array_equal(basins, phase["basin_node"])
    graph_edges = {row["node"]: {b for a, b in result["edges"] if a == row["node"]} for row in result["nodes"]}
    for row in result["nodes"]:
        source = row["node"]
        seen, pending = set(), [source]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(graph_edges[node])
        assert seen == reachable_nodes[row["cell_ids"][0]]
    model = learning.load(folder/"model")
    p = model.prepare(model.params)
    function = jax.jit(lambda u: model.quotient(p, u, steps=10))
    metrics = read(folder/"metrics.json")
    orbit_geometry = []
    for z in metrics["interior_fixed_latitudes"]:
        point = jnp.array([[np.sqrt(1-z*z), 0., z]], dtype=jnp.float32)
        decoded = np.asarray(model.decode(p, point))[0]
        _, angle = model.heads(p, jnp.asarray(z, dtype=jnp.float32))
        multiplier = float(jax.grad(lambda height: model.latitude(p, height))(jnp.asarray(z, dtype=jnp.float32)))
        orbit_geometry.append(dict(latent_height=z, decoded_physical_height=float(decoded[2]),
                                   angle_per_step=float(angle), transverse_multiplier=multiplier))
    boxes = phase["boxes"]
    maximum_violation = 0.
    for start in range(0, count, 16):
        chosen = boxes[start:start+16]
        points = chosen[:, :1]+(chosen[:, 1:]-chosen[:, :1])*np.linspace(0, 1, 4097)
        values = np.asarray(function(jnp.asarray(points.reshape(-1), dtype=jnp.float32))).reshape(points.shape)
        images = phase["images"][start:start+16]
        maximum_violation = max(maximum_violation, float(np.max(images[:, 0]-values.min(axis=1))),
                                 float(np.max(values.max(axis=1)-images[:, 1])))
    assert maximum_violation <= 0
    return dict(independent_cyclic_components=len(cyclic), all_128_sink_labels_checked=True,
                all_morse_reachability_checked=True, dense_image_samples=count*4097,
                dense_image_violation=maximum_violation, learned_orbit_geometry=orbit_geometry)


def kernel_audit(folder, data_dir, policy):
    fast = learning.load(folder/"model")
    native = learning.load(folder/"model", backend="native")
    with np.load(data_dir/f"{policy}-test.npz") as saved:
        x = ((saved["observations"][:128, 0]-fast.mean)/fast.scale).astype(np.float32)
    a, b = fast.functions(), native.functions()
    inputs = [x, x[:, 12:15], x[:, 12:15], x[:, 14], x[:, 14], x[:, 14], x[:, 14]]
    errors = [float(np.max(np.abs(np.asarray(f(jnp.asarray(value)))-np.asarray(g(jnp.asarray(value))))))
              for f, g, value in zip(a, b, inputs, strict=True)]
    assert max(errors) < 3e-5
    return dict(official_emlp_output_maximum_errors=errors, maximum_error=max(errors), samples=128)


def run(out, data_dir):
    out, data_dir = Path(out), Path(data_dir)
    summary = read(out/"summary.json")
    checks = {}
    for row in summary:
        policy, seed = row["policy"], row["seed"]
        folder = out/f"{policy}/seed{seed}"
        checks[f"{policy}/seed{seed}"] = graph_audit(folder)
    for policy, selection in read(out/"selection.json").items():
        checks[f"{policy}/official-emlp"] = kernel_audit(out/f"{policy}/seed{selection['selected_seed']}", data_dir, policy)
    write(out/"independent-audit.json", checks)
    print(checks, flush=True)
    return checks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.data)
