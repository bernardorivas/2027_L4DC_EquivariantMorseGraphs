"""Held-out diagnostics and native regular-interval CMGDB computations."""

from pathlib import Path

import CMGDB
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from common import systems, figure_style as style
from common.workflow import read, write
from common.aggregate_interval import aggregate, save as save_aggregate
from example3 import sphere
from . import controller as ctl, model as learning


def graph(function, folder, time=1., samples=257, padding=2e-6, cells=128, aggregate_interior=False):
    folder.mkdir(parents=True, exist_ok=True)
    cache = {}
    def batch(boxes):
        missing = np.asarray([box for box in boxes if tuple(box) not in cache])
        if len(missing):
            points = missing[:, :1]+(missing[:, 1:]-missing[:, :1])*np.linspace(0, 1, samples)
            values = np.asarray(function(points.reshape(-1)), dtype=np.float64).reshape(points.shape)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite scalar time map.")
            images = np.column_stack((np.maximum(-1., values.min(axis=1)-padding),
                                      np.minimum(1., values.max(axis=1)+padding)))
            for box, image in zip(missing, images, strict=True):
                cache[tuple(box)] = image.tolist()
        return [cache[tuple(box)] for box in boxes]
    def box_map(box):
        return batch([box])[0]
    depth = cells.bit_length()-1
    native = CMGDB.Model(depth, depth, depth, 10000, [-1.], [1.], box_map)
    native.set_batch_map(batch)
    morse, phase = CMGDB.ComputeConleyMorseGraph(native, cache_transition_graph=True, cache_map_graph=True,
        batch_chunk_size=128, reserve_min_edges=0)
    boxes = np.asarray([morse.phase_space_box(i) for i in range(phase.num_vertices())])
    nodes, membership = [], np.full(len(boxes), -1, dtype=int)
    for node in morse.vertices():
        ids = sorted(map(int, morse.morse_set(node)))
        membership[ids] = node
        nodes.append(dict(node=int(node), cell_ids=ids,
            interval=[float(boxes[ids, 0].min()), float(boxes[ids, 1].max())],
            annotations=list(morse.annotations(node))))
    edges = sorted(map(list, morse.edges()))
    sinks = sorted(set(map(int, morse.vertices()))-{int(a) for a, _ in edges})
    singleton = np.asarray(CMGDB.MorseSingletonReachability(phase, morse, np.arange(cells, dtype=np.uint64)))
    basins = np.where(np.isin(singleton, sinks), singleton, -1)
    offsets, targets = [np.asarray(values) for values in phase.csr_view()]
    rows = np.repeat(np.arange(cells), np.diff(offsets).astype(int))
    known = basins[rows] >= 0
    assert np.all(basins[targets[known]] == basins[rows[known]])
    result = dict(cells=cells, time=time, nodes=nodes, edges=edges, sink_nodes=sinks,
        basin_cells={str(node): int(np.sum(basins == node)) for node in sinks},
        unresolved_cells=int(np.sum(basins < 0)), padding=padding, samples_per_cell=samples,
        coefficient_field=5, annotation_space="interval quotient",
        image_status="sampled function hulls; not validated enclosures",
        basin_status="singleton sink reachability in the finite graph")
    write(folder/"results.json", result)
    arrays = dict(boxes=boxes, images=np.asarray(batch(boxes)), membership=membership,
                  basin_node=basins, csr_offsets=offsets, csr_targets=targets)
    np.savez_compressed(folder/"phase.npz", **arrays)
    CMGDB.SaveMorseSets(morse, str(folder/"morse-sets.csv"))
    native_graph = CMGDB.PlotMorseGraph(morse, clist=[style.MORSE[i % len(style.MORSE)] for i in range(len(nodes))])
    (folder/"morse-graph.dot").write_text(native_graph.source)
    (folder/"morse-graph.pdf").write_bytes(native_graph.pipe(format="pdf"))
    (folder/"morse-graph.svg").write_bytes(native_graph.pipe(format="svg"))
    aggregated, aggregate_arrays = aggregate(native, morse, phase, result, arrays, aggregate_interior)
    save_aggregate(aggregated, aggregate_arrays, folder/"aggregate")
    print(folder, f"{len(nodes)} raw components; {len(aggregated['nodes'])} aggregate components; "
          f"aggregate basin cells {aggregated['basin_cells']}", flush=True)
    return result


def diagnostics(model, data_dir, policy, folder):
    encode, decode, advance, coordinate, inverse, orbit_decode, latitude = model.functions()
    p = model.prepare(model.params)
    grid = np.linspace(-1., 1., 16385, dtype=np.float32)
    test_points = np.stack((np.sqrt(np.maximum(0., 1-grid*grid)), np.zeros_like(grid), grid), axis=-1)
    image = sphere.batched(advance, test_points)
    coordinates = np.asarray(coordinate(jnp.asarray(grid)))
    values = np.asarray(model.coordinate_values(p))
    knots = np.asarray(model.knots)
    slopes = np.diff(values)/np.diff(knots)
    boosts, angles = jax.jit(lambda z: model.heads(p, z))(jnp.asarray(grid))
    boosts, angles = np.asarray(boosts), np.asarray(angles)
    derivative_b = np.asarray(jax.jit(jax.vmap(jax.grad(lambda z: model.heads(p, z)[0])))(jnp.asarray(grid)))
    denominators = np.cosh(boosts)+grid*np.sinh(boosts)
    derivatives = (1+(1-grid*grid)*derivative_b)/denominators**2
    roots = []
    scalar_boost = jax.jit(lambda z: model.heads(p, z)[0])
    for a, b, fa, fb in zip(grid[:-1], grid[1:], boosts[:-1], boosts[1:], strict=True):
        if fa*fb < 0:
            root = brentq(lambda z: float(scalar_boost(jnp.asarray(z, dtype=jnp.float32))), float(a), float(b), xtol=1e-8)
            roots.append(root)
    observed = {}
    physical_data = {}
    for split in ("validation", "test"):
        with np.load(data_dir/f"{policy}-{split}.npz") as saved:
            physical = saved["observations"]
        normalized = ((physical-model.mean)/model.scale).astype(np.float32)
        starts = np.random.default_rng(8181 if split == "validation" else 9191).integers(23, size=len(physical))
        observed[split] = learning.evaluate(model, model.params,
            normalized[np.arange(len(physical))[:, None], starts[:, None]+learning.HORIZONS])
        if split == "test":
            raw_encoder = jax.jit(lambda x: model.network(0, p[0], x))
            raw = sphere.batched(raw_encoder, normalized.reshape(-1, 18))
            minimum_encoder_norm = float(np.linalg.norm(raw, axis=1).min())
            code = sphere.batched(encode, normalized[:, 0])
            next_true = sphere.batched(encode, normalized[:, 1])
            next_pred = sphere.batched(advance, code)
            predicted = code.copy()
            forecasts = [sphere.batched(decode, predicted)]
            closures = [float(np.sqrt(np.mean((code-sphere.batched(encode, normalized[:, 0]))**2)))]
            for step in range(1, 33):
                predicted = sphere.batched(advance, predicted)
                forecasts.append(sphere.batched(decode, predicted))
                true_code = sphere.batched(encode, normalized[:, step])
                closures.append(float(np.sqrt(np.mean((predicted-true_code)**2))))
            forecasts = np.stack(forecasts, axis=1)
            forecast_rms = np.sqrt(np.mean((forecasts-physical[..., 12:15])**2, axis=(0, 2)))
            angle = .731
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                 [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
            rotated = systems.observation_action("attitude", physical[:256, 0], rotation)
            rotated_code = sphere.batched(encode, ((rotated-model.mean)/model.scale).astype(np.float32))
            encoded_equivariance = float(np.max(np.abs(rotated_code-code[:256] @ rotation.T)))
            map_equivariance = float(np.max(np.abs(sphere.batched(advance, test_points @ rotation.T)-image @ rotation.T)))
            direct_lower = np.asarray(coordinate(jnp.asarray(next_pred[:, 2])))
            indirect_lower = np.asarray(jax.jit(lambda u: model.quotient(p, u))(coordinate(jnp.asarray(code[:, 2]))))
            physical_data = dict(test_physical_axis=physical[:, 0, 12:15], test_latent_axis=code,
                                 forecast_rms=forecast_rms, closure_rms=np.asarray(closures),
                                 test_forecasts=forecasts[:32], test_targets=physical[:32, :, 12:15])
    support = []
    for split in ("train", "validation"):
        with np.load(data_dir/f"{policy}-{split}.npz") as saved:
            physical = saved["observations"][:, [0, 1, 4, 10, 32]].reshape(-1, 18)
        code = sphere.batched(encode, ((physical-model.mean)/model.scale).astype(np.float32))
        support.append(np.asarray(coordinate(jnp.asarray(code[:, 2]))))
    counts = np.bincount(np.clip(np.floor((np.concatenate(support)+1)*64).astype(int), 0, 127), minlength=128)
    endpoint_images = np.asarray(latitude(jnp.array([-1., 1.])))
    raw_decoder = jax.jit(lambda x: model.network(1, p[1], x))
    minimum_decoder_norm = float(np.linalg.norm(sphere.batched(raw_decoder, test_points), axis=1).min())
    results = dict(**observed, one_step_physical_closure_rms=float(np.sqrt(np.mean((next_pred-next_true)**2))),
        forecast_rms_by_horizon=forecast_rms.tolist(), closure_rms_by_horizon=closures,
        sampled_encoder_equivariance_error=encoded_equivariance,
        sampled_map_equivariance_error=map_equivariance,
        maximum_sphere_norm_error=float(np.max(np.abs(np.linalg.norm(image, axis=1)-1))),
        lower_square_maximum_error=float(np.max(np.abs(direct_lower-indirect_lower))),
        endpoint_images=endpoint_images.tolist(), minimum_latitude_derivative=float(derivatives.min()),
        minimum_coordinate_slope=float(slopes.min()), maximum_coordinate_slope=float(slopes.max()),
        minimum_test_raw_encoder_norm=minimum_encoder_norm, minimum_sampled_raw_decoder_norm=minimum_decoder_norm,
        inverse_identity_maximum=float(np.max(np.abs(np.asarray(inverse(coordinate(jnp.asarray(grid))))-grid))),
        interior_fixed_latitudes=roots, supported_interval_cells=int(np.sum(counts > 0)),
        pole_boosts=[float(boosts[0]), float(boosts[-1])],
        orbit_decoder_rms=float(np.sqrt(np.mean((np.asarray(orbit_decode(coordinate(jnp.asarray(grid))))-grid)**2))),
        numerical_status="held-out samples and floating-point structural checks; no physical transfer certificate")
    assert results["endpoint_images"] == [-1., 1.]
    assert results["maximum_sphere_norm_error"] < 2e-6
    assert results["lower_square_maximum_error"] < 2e-6
    assert results["minimum_coordinate_slope"] > 0
    assert minimum_encoder_norm > 1e-9 and minimum_decoder_norm > 1e-9
    np.savez_compressed(folder/"diagnostics.npz", heights=grid, boost=boosts, angle=angles,
                        latitude=image[:, 2], coordinates=coordinates, knots=knots, coordinate_values=values,
                        physical_support_counts=counts, **physical_data)
    write(folder/"metrics.json", results)
    return results


def run(out, data_dir, policies, seeds):
    out, data_dir = Path(out), Path(data_dir)
    if not (out/"controller/validation.json").exists() and not (out.parent/"controller/validation.json").exists():
        from .validation import physical_checks
        physical_checks(out/"controller")
    ranking, models = {}, {}
    # Freeze validation selection before any graph calculation.
    for policy in policies:
        rows = []
        for seed in seeds:
            folder = out/f"{policy}/seed{seed}"
            models[(policy, seed)] = learning.load(folder/"model")
            config = read(folder/"model/config.json")
            rows.append(dict(seed=seed, validation_objective=config["best_validation"]))
        rows.sort(key=lambda row: (row["validation_objective"], row["seed"]))
        ranking[policy] = dict(selected_seed=rows[0]["seed"], ranking=rows,
                               criterion="joint validation objective, before graph computation")
    write(out/"selection.json", ranking)
    summary = []
    for policy in policies:
        reference_graph = graph(lambda z: ctl.height_flow(z, 1., policy), out/f"physical-reference/{policy}",
                                aggregate_interior=policy == "coning")
        for seed in seeds:
            folder = out/f"{policy}/seed{seed}"
            model = models[(policy, seed)]
            metrics = diagnostics(model, data_dir, policy, folder)
            p = model.prepare(model.params)
            advance = jax.jit(lambda u: model.quotient(p, u, steps=10))
            coning_structure = (policy == "coning" and len(metrics["interior_fixed_latitudes"]) == 1
                                and metrics["pole_boosts"][0] > 0 > metrics["pole_boosts"][1])
            native_graph = graph(lambda u: np.asarray(advance(jnp.asarray(u, dtype=jnp.float32))), folder/"cmgdb",
                                 aggregate_interior=coning_structure)
            values, knots = np.asarray(model.coordinate_values(p)), np.asarray(model.knots)
            def reference_in_coordinate(u):
                z = np.interp(u, values, knots)
                return np.interp(ctl.height_flow(z, 1., policy), knots, values)
            matched = graph(reference_in_coordinate, folder/"matched-reference", aggregate_interior=policy == "coning")
            summary.append(dict(policy=policy, seed=seed, metrics=metrics,
                                graph=native_graph, matched_reference=matched, physical_reference=reference_graph,
                                aggregate_graph=read(folder/"cmgdb/aggregate/results.json")))
            write(out/"summary.json", summary)
    from . import figures
    figures.build(out)
    return summary
