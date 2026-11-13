"""Native Morse graphs, attracting cell regions, and numerical tolerance margins."""

from pathlib import Path
import sys
import time

import numpy as np
import CMGDB

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from latent_equivariance import quotient_certificates as qc

from .workflow import write

NATIVE_OPTIONS = dict(cache_transition_graph=True, cache_map_graph=True,
                      batch_chunk_size=512, reserve_min_edges=0)


def elapsed(function):
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter()-start


def measure(function, repeats=3):
    """Retain the first call separately from repeated warm calls."""
    if repeats < 0:
        raise ValueError("The repetition count must be nonnegative.")
    value, first = elapsed(function)
    values = []
    for _ in range(repeats):
        value, seconds = elapsed(function)
        values.append(seconds)
    median = float(np.median(values)) if values else first
    return value, dict(first_seconds=first, warm_seconds=values, median_seconds=median,
                      median_kind="warm" if values else "first call",
                      relative_spread=(max(values)-min(values))/median if values and median else None)


def plain(value):
    """Keep reports strict JSON, with an explicit flag accompanying infinite bounds."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def table_model(boxes, images):
    dimension = boxes.shape[1]//2
    lower, upper = boxes[:, :dimension], boxes[:, dimension:]
    lookup = {tuple(box): image.tolist() for box, image in zip(boxes, images, strict=True)}
    return make_model(boxes, lambda box: lookup[tuple(box)],
                      lambda batch: [lookup[tuple(box)] for box in batch])


def make_model(boxes, callback, batch):
    dimension = boxes.shape[1]//2
    count = len(boxes)
    if count < 2 or count & (count-1):
        raise ValueError("Use a product grid with a power-of-two cell count.")
    depth = count.bit_length()-1
    model = CMGDB.Model(depth, depth, depth, 10000,
                        boxes[:, :dimension].min(axis=0).tolist(),
                        boxes[:, dimension:].max(axis=0).tolist(), callback)
    model.set_batch_map(batch)
    return model


def compute_graph(boxes, images, *, boundary_faces, map_time, support=None,
                  data_map=None, aggregation=None, repeats=5):
    """Recompute the native graph and its conservative minimal-node regions.

    A data map is a native BoxMapData built from actual encoded physical sources.
    No interpolation or learned-map evaluations in unavailable cells are used.
    """
    boxes, images = np.asarray(boxes, float), np.asarray(images, float)
    dimension = boxes.shape[1]//2
    if images.shape != boxes.shape or not np.isfinite(images).all():
        raise ValueError("Each cell needs a finite image box.")
    index = qc.CellIndex(boxes[:, :dimension], boxes[:, dimension:])
    support = np.ones(len(boxes), bool) if support is None else np.asarray(support, bool)
    if support.shape != (len(boxes),):
        raise ValueError("Support needs one boolean value per cell.")
    faces = np.asarray(boundary_faces, bool)
    if faces.shape != (dimension, 2):
        raise ValueError("Boundary faces need one pair per coordinate.")
    def run():
        native = table_model(boxes, images) if data_map is None else make_model(boxes, data_map, data_map.batch)
        morse, phase = CMGDB.ComputeConleyMorseGraph(native, **NATIVE_OPTIONS)
        return native, morse, phase
    (native, morse, phase), native_time = measure(run, repeats)
    phase_boxes = np.asarray([morse.phase_space_box(i) for i in range(len(boxes))])
    # CMGDB orders a planar grid differently from a coordinate product.
    centers = (phase_boxes[:, :dimension]+phase_boxes[:, dimension:])/2
    order = index.containing(centers, slack=0)[:, 0]
    if np.any(order < 0) or len(np.unique(order)) != len(boxes):
        raise RuntimeError("The native grid does not match the supplied cells.")
    if not np.array_equal(phase_boxes, boxes[order]):
        raise RuntimeError("The native cell boxes differ from the supplied grid.")
    boxes, images, support = boxes[order], images[order], support[order]
    offsets, targets = [np.asarray(values, dtype=np.int64) for values in phase.csr_view()]
    nodes, membership = [], np.full(len(boxes), -1, dtype=int)
    for node in morse.vertices():
        ids = sorted(map(int, morse.morse_set(node)))
        membership[ids] = node
        row = dict(node=int(node), cell_ids=ids, annotations=list(morse.annotations(node)),
                   hull=np.r_[boxes[ids, :dimension].min(axis=0), boxes[ids, dimension:].max(axis=0)].tolist())
        if dimension == 1:
            row["interval"] = row["hull"]
        nodes.append(row)
    edges = sorted(map(list, morse.edges()))
    sinks = sorted({row["node"] for row in nodes}-{int(a) for a, _ in edges})
    singleton = np.asarray(CMGDB.MorseSingletonReachability(phase, morse,
                           np.arange(len(boxes), dtype=np.uint64)))
    labels = np.where(np.isin(singleton, sinks), singleton, -1)
    record = dict(cells=len(boxes), time=map_time, nodes=nodes, edges=edges, sink_nodes=sinks,
                  basin_cells={str(node): int(np.sum(labels == node)) for node in sinks},
                  unresolved_cells=int(np.sum(labels < 0)), coefficient_field=5,
                  image_status="sampled image hulls; not validated enclosures")
    arrays = dict(boxes=boxes, images=images, membership=membership, basin_node=labels,
                  active=support, csr_offsets=offsets, csr_targets=targets)
    raw_record, raw_arrays = dict(record), dict(arrays)
    start = time.perf_counter()
    if aggregation is not None:
        if dimension != 1 or aggregation not in {"coning", "endpoints"}:
            raise ValueError("The supplied structural aggregate is an interval controller aggregate.")
        from .aggregate_interval import aggregate
        try:
            if aggregation == "coning":
                south = [row for row in nodes if row["interval"][0] == -1.]
                north = [row for row in nodes if row["interval"][1] == 1.]
                if len(south) != 1 or len(north) != 1 or south[0]["node"] == north[0]["node"]:
                    raise ValueError("The coning structure requires two separate pole components.")
                if any(row["node"] in sinks or row["annotations"] != ["0", "0"] for row in south+north):
                    raise ValueError("The native pole components do not have the prescribed repelling structure.")
            record, arrays = aggregate(native, morse, phase, record, arrays, aggregation == "coning")
            record["aggregation"]["applied"] = True
        except (ValueError, RuntimeError) as error:
            record = dict(raw_record, aggregation=dict(applied=False, requested=aggregation,
                          reason=str(error), native_components=len(nodes), aggregate_components=len(nodes)))
            arrays = dict(raw_arrays)
    else:
        record["aggregation"] = dict(applied=True, identity=True, native_components=len(nodes),
                                      aggregate_components=len(nodes))
    labels = arrays["basin_node"].copy()
    outside = np.any(((images[:, :dimension] < boxes[:, :dimension].min(axis=0)) & ~faces[:, 0]) |
                     ((images[:, dimension:] > boxes[:, dimension:].max(axis=0)) & ~faces[:, 1]), axis=1)
    unsafe = qc.forward_closure(*qc.reverse(offsets, targets), np.flatnonzero(~support | outside))
    labels[unsafe] = -1
    sources = np.repeat(np.arange(len(boxes)), np.diff(offsets))
    known = labels[sources] >= 0
    if not np.all(labels[targets[known]] == labels[sources[known]]):
        raise RuntimeError("The conservative basin regions are not forward invariant.")
    record = dict(record, basin_cells={str(node): int(np.sum(labels == node)) for node in record["sink_nodes"]},
                  unresolved_cells=int(np.sum(labels < 0)), active_cells=int(support.sum()),
                  unsafe_cells=int(unsafe.sum()), forward_invariance_checked=True)
    arrays = dict(arrays, basin_node=labels, unsafe=unsafe)
    timings = dict(native_cmgdb=native_time,
                   region_extraction_including_aggregation_seconds=time.perf_counter()-start,
                   native_scope="graph and quotient indices; includes data queries when BoxMapData is supplied")
    return raw_record, raw_arrays, record, arrays, timings


def boundary_distance(boxes, images, selected, metric, true_faces, geometry):
    """Independent check using the exposed faces of the region."""
    dimension = boxes.shape[1]//2
    index = qc.CellIndex(boxes[:, :dimension], boxes[:, dimension:])
    positions = np.rint((boxes[:, :dimension]-index.domain_lower)/index.side).astype(int)
    faces, owners = [], []
    for cell in np.flatnonzero(selected):
        for axis in range(dimension):
            for side in (0, 1):
                neighbor = positions[cell].copy()
                neighbor[axis] += 2*side-1
                inside = bool(np.all((neighbor >= 0) & (neighbor < np.array(index.shape))))
                if inside and selected[index.ids[tuple(neighbor)]]:
                    continue
                if not inside and true_faces[axis, side]:
                    continue
                low, high = geometry[cell, :dimension].copy(), geometry[cell, dimension:].copy()
                low[axis] = high[axis] = low[axis] if side == 0 else high[axis]
                faces.append(np.r_[low, high])
                owners.append(int(cell))
    if not faces:
        return dict(value=float("inf"), witness=None)
    faces = np.asarray(faces)
    ids = np.flatnonzero(selected)
    best, witness = np.inf, None
    for start in range(0, len(ids), 128):
        cells = ids[start:start+128]
        distance = metric.lower(images[cells, None, :dimension], images[cells, None, dimension:],
                                faces[None, :, :dimension], faces[None, :, dimension:])
        row, col = np.unravel_index(distance.argmin(), distance.shape)
        if distance[row, col] < best:
            best = float(distance[row, col])
            witness = dict(source_cell=int(cells[row]), boundary_cell=owners[col],
                           boundary=faces[col].tolist(), image_hull=images[cells[row]].tolist())
    return dict(value=best, witness=witness)


def compare_regions(record, arrays, codes, specifications, *, boundary_faces):
    """Compare sampled residuals with margins in matching metrics.

    A specification contains a metric, cell and image boxes in its coordinates,
    and predicted and encoded physical images in those same coordinates. Region
    membership is always tested in the original grid coordinates `codes`.
    """
    boxes, labels = arrays["boxes"], arrays["basin_node"]
    dimension = boxes.shape[1]//2
    codes = np.asarray(codes).reshape(-1, dimension)
    index = qc.CellIndex(boxes[:, :dimension], boxes[:, dimension:])
    results, residuals = [], {}
    for name, spec in specifications.items():
        residuals[name] = spec["metric"].point(spec["predicted"], spec["encoded_image"])
        if residuals[name].shape != (len(codes),) or not np.isfinite(residuals[name]).all():
            raise ValueError("Residual samples must be finite and match the encoded starts.")
    for node in record["sink_nodes"]:
        selected = labels == node
        if not selected.any():
            results.append(dict(node=int(node), cells=0, available=False, comparisons={},
                                reason="no safe cells", certified=False))
            continue
        member = index.points_in(codes, selected)
        row = dict(node=int(node), cells=int(selected.sum()), cell_ids=np.flatnonzero(selected).tolist(),
                   latent_hull=np.r_[boxes[selected, :dimension].min(axis=0),
                                     boxes[selected, dimension:].max(axis=0)].tolist(),
                   available=True, comparisons={}, certified=False)
        for name, spec in specifications.items():
            geometry, image = spec["boxes"], spec["images"]
            margin, seconds = elapsed(lambda: qc.tolerance_lower(
                geometry[:, :dimension], geometry[:, dimension:], selected,
                image[:, :dimension], image[:, dimension:], spec["metric"], boundary_faces=boundary_faces))
            independent, check_seconds = elapsed(lambda: boundary_distance(
                boxes, image, selected, spec["metric"], np.asarray(boundary_faces), geometry))
            if not np.isclose(margin["lower"], independent["value"], rtol=2e-10, atol=2e-12):
                raise RuntimeError("The complement and boundary-face tolerance calculations disagree.")
            comparison = qc.compare(margin["lower"], residuals[name], member, "sampled")
            samples = np.flatnonzero(member)
            maximum = comparison["max_residual"]
            row["comparisons"][name] = dict(comparison,
                infinite_tolerance=bool(np.isinf(margin["lower"])), tolerance=margin,
                independent_boundary_distance=independent,
                residual_over_tolerance=maximum/margin["lower"] if maximum is not None and margin["lower"] else None,
                worst_sample=int(samples[np.argmax(residuals[name][samples])]) if len(samples) else None,
                tolerance_seconds=seconds, independent_check_seconds=check_seconds, certified=False)
        results.append(row)
    return results, residuals


def save_result(out, raw_record, raw_arrays, record, arrays, regions, residual_arrays, metadata, timings):
    """Save numerical results without changing any input checkpoint or reference."""
    out = Path(out)
    write(out/"raw-graph.json", plain(raw_record))
    np.savez_compressed(out/"raw-phase.npz", **raw_arrays)
    write(out/"graph.json", plain(record))
    np.savez_compressed(out/"phase.npz", **arrays)
    write(out/"regions.json", plain(regions))
    np.savez_compressed(out/"residuals.npz", **residual_arrays)
    write(out/"metadata.json", plain(metadata))
    write(out/"timings.json", plain(timings))
    lines = ["# Attracting-region tolerances", "",
             f"Example: {metadata['example']}. Physical map time: {metadata['map_time']:.12g}.", "",
             f"Primary metric: `{metadata['primary_metric']}`. " + " ".join(f"{key}: {value}." for key, value in metadata["metrics"].items()), "",
             "Each region is a compact union of cells whose only reachable Morse node is a specified minimal node. Unavailable cells, paths to them, and artificial-window exits are excluded. The graph is recomputed from the supplied checkpoint.", "",
             "Image hulls and residual maxima are sampled. The reported comparisons are numerical and do not establish uniform physical certificates.", "",
             "| Node | Region cells | Metric | Numerical tolerance | Maximum sampled residual | Samples in region | Observed comparison |",
             "|---:|---:|---|---:|---:|---:|---|" ]
    for row in regions:
        if not row["available"]:
            lines.append(f"| {row['node']} | 0 | unavailable | unavailable | unavailable | 0 | {row['reason']} |")
            continue
        ordered = sorted(row["comparisons"], key=lambda name: name != metadata["primary_metric"])
        for name in ordered:
            item = row["comparisons"][name]
            bound = "infinity" if item["infinite_tolerance"] else f"{item['tolerance_lower']:.9g}"
            error = "unavailable" if item["max_residual"] is None else f"{item['max_residual']:.9g}"
            lines.append(f"| {row['node']} | {row['cells']} | {name} | {bound} | {error} | {item['count']} | {'pass' if item['passed'] else 'fail'} |")
    aggregation = record["aggregation"]
    if not aggregation["applied"]:
        aggregation_text = "The requested structural aggregate is unavailable: " + aggregation["reason"]
    elif aggregation.get("identity"):
        aggregation_text = "Each native component is retained."
    else:
        aggregation_text = (f"The structural aggregate has {aggregation['aggregate_components']} components "
                            f"from {aggregation['native_components']} native components. Connecting cells are included, "
                            "and native CMGDB checks and recomputes the aggregate index.")
    lines += ["", aggregation_text, "",
              "Metric definitions, sampling settings, checkpoint hashes, and numerical versions are in `metadata.json`. Native graph, image or source-pair construction, residual evaluation, and initialization costs are separate in `timings.json`. First calls and warm repetitions are retained. Training, plotting, and full-space index reconstruction are excluded.", "",
              "`raw-graph.json` retains every native component. `graph.json` includes the prescribed structural aggregate when its checks succeed. An unsupported aggregate remains explicitly unavailable, with the raw graph retained. `regions.json` includes the limiting boundary witnesses and every failed comparison. An infinite tolerance denotes the whole domain relative to its genuine boundary and is not an informative separation test.", ""]
    (out/"REPORT.md").write_text("\n".join(lines))
    print(out/"REPORT.md", flush=True)
