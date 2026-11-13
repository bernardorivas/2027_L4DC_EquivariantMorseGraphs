"""Regular interval box maps, native basins, and two-stage diagnostics."""

from pathlib import Path
from collections import deque

import CMGDB
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from common import systems
from common.workflow import read, write
from . import sphere, interval

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#A6761D"]


def reference(policy, folder, settings):
    """Match the grid, time, and padding on the analytic thrust-axis subsystem."""
    folder.mkdir(parents=True, exist_ok=False)
    count = settings["cells"]
    vertices = np.linspace(-1, 1, count+1)
    def rhs(_, z):
        return systems.attitude_gain(z, policy)[0]*(1-z*z)
    answer = solve_ivp(rhs, (0., .1), vertices, method="DOP853", rtol=2e-13, atol=2e-14)
    if not answer.success:
        raise RuntimeError("The numerical thrust-axis reference integration failed.")
    images = answer.y[:, -1]
    roots = [-1., 1.] if policy == "nominal" else [-1., .25, .875, 1.]
    for z in roots:
        images[vertices == z] = z
    if policy == "nominal":
        direct = (vertices+np.tanh(.1))/(1+vertices*np.tanh(.1))
        assert np.max(np.abs(images-direct)) < 1e-12
    assert np.all(np.diff(images) > 0)
    values, padding = dict(zip(vertices, images, strict=True)), settings["image_padding"]
    def box_map(box):
        return [max(-1., values[box[0]]-padding), min(1., values[box[1]]+padding)]
    depth = count.bit_length()-1
    model = CMGDB.Model(depth, depth, depth, 10000, [-1.], [1.], box_map)
    morse, _ = CMGDB.ComputeConleyMorseGraph(model)
    nodes = [dict(node=int(node), intervals=morse.morse_set_boxes(node), annotations=list(morse.annotations(node)))
             for node in morse.vertices()]
    nonzero = [node for node in nodes if any(value != "0" for value in node["annotations"])]
    result = dict(policy=policy, cells=count, time_step=.1, image_padding=padding, nodes=nodes,
                  nonzero_nodes=nonzero, edges=sorted(map(list, morse.edges())),
                  evidence_status="DOP853 numerical reference with monotone endpoint hulls; not a validated enclosure")
    write(folder / "results.json", result)
    CMGDB.SaveMorseSets(morse, str(folder / "morse-sets.csv"))
    graph = CMGDB.PlotMorseGraph(morse)
    (folder / "morse-graph.dot").write_text(graph.source)
    (folder / "morse-graph.pdf").write_bytes(graph.pipe(format="pdf"))
    return result


class RegularIntervalMap:
    """Sample a learned interval function inside each requested rectangle."""

    def __init__(self, model, samples=257, padding=2e-6):
        _, _, self.advance, _ = interval.functions(model)
        self.fractions = np.linspace(0., 1., samples)
        self.padding, self.cache = padding, {}

    def __call__(self, box):
        key = tuple(box)
        if key not in self.cache:
            self.batch([box])
        return self.cache[key]

    def batch(self, boxes):
        boxes = np.asarray(boxes, dtype=np.float64)
        missing = np.array([box for box in boxes if tuple(box) not in self.cache])
        if len(missing):
            points = missing[:, :1]+(missing[:, 1:]-missing[:, :1])*self.fractions[None]
            values = np.asarray(self.advance(jnp.asarray(points.reshape(-1), dtype=jnp.float32))).reshape(points.shape)
            if not np.isfinite(values).all():
                raise FloatingPointError("The learned interval map became nonfinite.")
            images = np.column_stack((np.maximum(-1., values.min(axis=1).astype(float)-self.padding),
                                      np.minimum(1., values.max(axis=1).astype(float)+self.padding)))
            for box, image in zip(missing, images, strict=True):
                self.cache[tuple(box)] = image.tolist()
        return [self.cache[tuple(box)] for box in boxes]


def native(model, folder, settings):
    folder.mkdir(parents=True, exist_ok=True)
    count = settings["cells"]
    if count not in (64, 128):
        raise ValueError("Choose 64 or 128 interval cells for Example 3.")
    function = RegularIntervalMap(model, settings["samples_per_cell"], settings["image_padding"])
    depth = count.bit_length()-1
    native_model = CMGDB.Model(depth, depth, depth, 10000, [-1.], [1.], function)
    native_model.set_batch_map(function.batch)
    morse, phase = CMGDB.ComputeConleyMorseGraph(native_model, cache_transition_graph=True, cache_map_graph=True,
                                               batch_chunk_size=128, reserve_min_edges=0)
    boxes = np.asarray([morse.phase_space_box(i) for i in range(phase.num_vertices())])
    images = np.asarray(function.batch(boxes))
    membership = np.full(len(boxes), -1, dtype=int)
    nodes = []
    for node in morse.vertices():
        ids = sorted(map(int, morse.morse_set(node)))
        membership[ids] = node
        nodes.append(dict(node=int(node), cell_ids=ids, cells=len(ids),
                          interval=[float(boxes[ids, 0].min()), float(boxes[ids, 1].max())],
                          annotations=list(morse.annotations(node))))
    edges = sorted(map(list, morse.edges()))
    offsets, targets = [np.asarray(values) for values in phase.csr_view()]
    cells = np.arange(len(boxes), dtype=np.uint64)
    singleton = np.asarray(CMGDB.MorseSingletonReachability(phase, morse, cells))
    masks = np.asarray(CMGDB.MorseReachabilityMasks(phase, morse, cells))
    nonempty = masks != 0
    one = nonempty & ((masks & (masks-np.uint64(1))) == 0)
    assert np.array_equal(singleton >= 0, one)
    sinks = sorted({int(node) for node in morse.vertices()}-{int(a) for a, _ in edges})
    basins = np.where(np.isin(singleton, sinks), singleton, -1)
    rows = np.repeat(np.arange(len(boxes)), np.diff(offsets).astype(int))
    assert np.all(basins[targets[basins[rows] >= 0]] == basins[rows[basins[rows] >= 0]])
    result = dict(cells=count, domain=[-1., 1.], nodes=nodes, edges=edges, sink_nodes=sinks,
          basin_cells={str(node): int(np.sum(basins == node)) for node in sinks},
          unresolved_cells=int(np.sum(basins < 0)), map_constructor="regular learned scalar box callback",
          samples_per_cell=settings["samples_per_cell"], padding=settings["image_padding"],
          time_step=.1, coefficient_field=5, annotation_space="learned interval quotient only",
          image_status="sampled function hulls; not validated",
          finite_graph_basin_forward_invariance_checked=True)
    np.savez_compressed(folder / "phase.npz", boxes=boxes, images=images, membership=membership,
                        singleton=singleton, reach_masks=masks, basin_node=basins,
                        csr_offsets=offsets, csr_targets=targets)
    write(folder / "results.json", result)
    graph = CMGDB.PlotMorseGraph(morse, clist=[COLORS[i % len(COLORS)] for i in range(max(1, len(nodes)))])
    (folder / "morse-graph.dot").write_text(graph.source)
    for kind in ("pdf", "png"):
        (folder / f"morse-graph.{kind}").write_bytes(graph.pipe(format=kind))
    CMGDB.SaveMorseSets(morse, str(folder / "morse-sets.csv"))
    print(folder, f"{len(nodes)} native nodes; basin cells {result['basin_cells']}; unresolved {result['unresolved_cells']}", flush=True)
    return result


def sphere_diagnostics(model, reduced, data_dir, policy, folder):
    encode, decode, advance = model.functions()
    interval_encode, _, interval_advance, _ = interval.functions(reduced)
    rng = np.random.default_rng(1929)
    with np.load(data_dir / f"{policy}-validation.npz") as data:
        valid = ((data["observations"]-model.mean)/model.scale).astype(np.float32)
        starts = np.random.default_rng(8181).integers(17, size=len(valid))
        validation = sphere.evaluate(model, model.params, valid[np.arange(len(valid))[:, None], starts[:, None]+sphere.HORIZONS])
    with np.load(data_dir / f"{policy}-test.npz") as data:
        physical = data["observations"]
        test = ((physical-model.mean)/model.scale).astype(np.float32)
        starts = np.random.default_rng(9191).integers(17, size=len(test))
        windows = test[np.arange(len(test))[:, None], starts[:, None]+sphere.HORIZONS]
        test_metrics = sphere.evaluate(model, model.params, windows)
        code = sphere.batched(encode, test[:, 0])
        true_next = sphere.batched(encode, test[:, 1])
        predicted = sphere.batched(advance, code)
        h = np.asarray(interval_encode(jnp.asarray(code)))
        hnext = np.asarray(interval_encode(jnp.asarray(true_next)))
        scalar_next = np.asarray(interval_advance(jnp.asarray(h)))
        scalar_model_target = np.asarray(interval_encode(jnp.asarray(predicted)))
        angle = .731
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        transformed = systems.observation_action("attitude", physical[:256, 0], rotation)
        transformed_code = sphere.batched(encode, ((transformed-model.mean)/model.scale).astype(np.float32))
        equivariance = float(np.max(np.abs(transformed_code-code[:256]@rotation.T)))
        physical_height = physical[:, 0, 14].copy()
        initial = data["initial_state"].copy()
    effective = model.prepare(model.params)
    raw_encoder = jax.jit(lambda x: model.apply(0, effective[0], x, normalized=False))
    raw_advance = jax.jit(lambda x: model.apply(2, effective[2], x, normalized=False))
    height = np.linspace(-1, 1, 513, dtype=np.float32)
    angles = np.linspace(0, 2*np.pi, 65, dtype=np.float32)
    zz, aa = np.meshgrid(height, angles, indexing="ij")
    radius = np.sqrt(np.maximum(0., 1-zz*zz))
    points = np.column_stack((radius.ravel()*np.cos(aa).ravel(), radius.ravel()*np.sin(aa).ravel(), zz.ravel()))
    advanced = sphere.batched(advance, points)
    raw = sphere.batched(raw_advance, points)
    raw_encoded = sphere.batched(raw_encoder, test[:4096, 0])
    defect = float(np.max(np.abs(sphere.batched(advance, points @ rotation.T)-advanced @ rotation.T)))
    scalar_sources = sphere.batched(interval_encode, points)
    predicted_scalar = sphere.batched(interval_advance, scalar_sources)
    actual_scalar = sphere.batched(interval_encode, advanced)
    azimuth = np.arctan2(advanced[:, 1], advanced[:, 0])-np.arctan2(points[:, 1], points[:, 0])
    azimuth = np.arctan2(np.sin(azimuth), np.cos(azimuth)).reshape(513, 65)
    coverage = []
    for split in ("train", "validation"):
        with np.load(data_dir / f"{policy}-{split}.npz") as data:
            observed = data["observations"][:, [0, 1, 4, 16, 32]].reshape(-1, 18)
            codes = sphere.batched(encode, ((observed-model.mean)/model.scale).astype(np.float32))
            coverage.append(np.asarray(interval_encode(jnp.asarray(codes))))
    support = np.concatenate(coverage)
    with np.load(folder / "cmgdb/phase.npz") as phase:
        count = len(phase["boxes"])
        bins = np.clip(np.floor((support+1)*count/2).astype(int), 0, count-1)
        counts = np.bincount(bins, minlength=count)
        canonical = np.clip(np.floor((phase["boxes"].mean(axis=1)+1)*count/2).astype(int), 0, count-1)
        assert np.array_equal(np.sort(canonical), np.arange(count))
        inverse_order = np.empty(count, dtype=int)
        inverse_order[canonical] = np.arange(count)
        physical_active = counts[canonical] > 0
        offsets, targets = phase["csr_offsets"], phase["csr_targets"]
        predecessors = [[] for _ in range(count)]
        for source in range(count):
            for destination in targets[offsets[source]:offsets[source+1]]:
                predecessors[int(destination)].append(source)
        unknown = ~physical_active
        queue = deque(map(int, np.flatnonzero(unknown)))
        while queue:
            destination = queue.popleft()
            for source in predecessors[destination]:
                if not unknown[source]:
                    unknown[source] = True
                    queue.append(source)
        physical_basin = np.where(physical_active & ~unknown, phase["basin_node"], -1)
        source_rows = np.repeat(np.arange(count), np.diff(offsets).astype(int))
        safe = physical_basin[source_rows] >= 0
        assert np.all(physical_basin[targets[safe]] == physical_basin[source_rows[safe]])
        test_bins = np.clip(np.floor((h+1)*count/2).astype(int), 0, count-1)
        test_basins = physical_basin[inverse_order[test_bins]]
    result = dict(validation=validation, test=test_metrics,
          physical_one_step_sphere_consistency_rms=float(np.sqrt(np.mean(np.sum((predicted-true_next)**2, axis=-1)))),
          physical_one_step_interval_consistency_rms=float(np.sqrt(np.mean((scalar_next-hnext)**2))),
          physical_second_stage_commutation_rms=float(np.sqrt(np.mean((scalar_next-scalar_model_target)**2))),
          sphere_second_stage_commutation_maximum=float(np.max(np.abs(predicted_scalar-actual_scalar))),
          encoder_equivariance_maximum=equivariance, sphere_map_equivariance_maximum=defect,
          minimum_raw_encoder_norm=float(np.linalg.norm(raw_encoded, axis=-1).min()),
          minimum_raw_sphere_map_norm=float(np.linalg.norm(raw, axis=-1).min()),
          sphere_norm_maximum_error=float(np.max(np.abs(np.linalg.norm(advanced, axis=-1)-1))),
          physical_source_interval_range=[float(support.min()), float(support.max())],
          physical_source_supported_cells=int(np.sum(counts > 0)),
          physical_basin_labeled_cells=int(np.sum(physical_basin >= 0)),
          physical_cells_reaching_unobserved_support=int(np.sum(unknown & physical_active)),
          support_source="training and validation observations at indices 0, 1, 4, 16, 32",
          test_latent_height_physical_tilt_correlation=float(np.corrcoef(code[:, 2], physical_height)[0, 1])
                if np.std(code[:, 2]) > 1e-12 and np.std(physical_height) > 1e-12 else None,
          physical_sources=len(support), evidence_status="sampled diagnostics; no physical transfer certificate")
    if not all(np.isfinite(v) for v in (result["minimum_raw_encoder_norm"], result["minimum_raw_sphere_map_norm"])):
        raise FloatingPointError("Nonfinite sphere normalization diagnostic.")
    if min(result["minimum_raw_encoder_norm"], result["minimum_raw_sphere_map_norm"]) <= 1e-8:
        raise FloatingPointError("A sampled raw sphere output is too close to zero.")
    np.savez_compressed(folder / "physical-diagnostics.npz", test_codes=code, test_interval=h,
          test_next_interval=hnext, test_predicted_interval=scalar_next, physical_tilt=physical_height,
          test_initial_state=initial, physical_occupancy=counts, latitude=height, angle=angles,
          physical_basin_node=physical_basin, physical_support_unknown=unknown,
          test_basin_node=test_basins,
          sphere_map_height=advanced[:, 2].reshape(513, 65), sphere_phase_increment=azimuth)
    write(folder / "sphere-metrics.json", result)
    return result


def compare_metrics(expected, actual, location=""):
    """Compare float32 model diagnostics while keeping discrete fields exact."""
    checked = 0
    for key, value in expected.items():
        label = f"{location}/{key}"
        if isinstance(value, dict):
            checked += compare_metrics(value, actual[key], label)
        elif isinstance(value, list):
            if len(value) != len(actual[key]):
                raise ValueError(f"Diagnostic length changed at {label}")
            checked += compare_metrics(dict(enumerate(value)), dict(enumerate(actual[key])), label)
        elif isinstance(value, float):
            if not np.isclose(value, actual[key], rtol=1e-4, atol=3e-6):
                raise ValueError(f"Float32 diagnostic changed at {label}: {value} versus {actual[key]}")
            checked += 1
        else:
            if value != actual[key]:
                raise ValueError(f"Discrete diagnostic changed at {label}")
            checked += isinstance(value, int) and not isinstance(value, bool)
    return checked


def compare_saved(folder, reference, tolerance=3e-6):
    expected, actual = read(reference / "results.json"), read(folder / "results.json")
    assert expected == actual
    with np.load(reference / "phase.npz") as old, np.load(folder / "phase.npz") as new:
        assert set(old.files) == set(new.files)
        maximum = float(np.max(np.abs(old["images"]-new["images"])))
        assert maximum < tolerance
        assert all(np.array_equal(old[key], new[key]) for key in old.files if key != "images")
    result = dict(passed=True, image_endpoint_maximum_difference=maximum,
                  cells_edges_annotations_and_basins_match=True)
    write(folder / "replay-verification.json", result)
    return result
