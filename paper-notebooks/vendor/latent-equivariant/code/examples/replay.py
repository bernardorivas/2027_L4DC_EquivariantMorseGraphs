"""Recompute CMGDB graphs, indices, and displayed regions from saved box images."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

_MPL_CACHE = tempfile.TemporaryDirectory(prefix="example-replay-mpl-")
os.environ["MPLCONFIGDIR"] = _MPL_CACHE.name

import CMGDB
from CMGDB.morse_lattice import DerivedGraph, transitive_reduction
import numpy as np

from render import verify_data

HERE = Path(__file__).resolve().parent
CMGDB_VERSION = "1.5.3+fork.7"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_graph(morse, record, sphere=False):
    assert sorted(map(tuple, morse.edges())) == sorted(map(tuple, record["edges"]))
    assert sorted(morse.vertices()) == sorted(row["node"] for row in record["nodes"])
    for row in record["nodes"]:
        key = "cell_indices" if sphere else "cell_ids"
        assert sorted(morse.morse_set(row["node"])) == sorted(row[key])
        annotations = row["native_annotations"] if sphere else row["annotations"]
        assert list(morse.annotations(row["node"])) == annotations


def replay_sphere(source, output, recompute_explicit=False):
    output.mkdir(parents=True)
    reports = {}
    for case in ("exact-K128", "seed0-K128"):
        src, dst = source / case, output / case
        dst.mkdir()
        record = read(src / "native-result.json")
        callbacks = read(src / "callback-images.json")
        images = {tuple(row["rectangle"]): row["image"] for row in callbacks}
        if recompute_explicit and case == "exact-K128":
            from exact_enclosures import sphere_box_images
            rectangles = list(images)
            values = sphere_box_images(np.asarray(rectangles))
            assert np.array_equal(values, np.asarray(list(images.values())))
            images = dict(zip(rectangles, values.tolist(), strict=True))
        depth = record["uniform_depth"]
        model = CMGDB.Model(depth, depth, depth, 10000, [-1.], [1.],
                            lambda rectangle: images[tuple(rectangle)])
        morse, graph = CMGDB.ComputeConleyMorseGraph(model)
        assert graph.num_vertices() == 2**depth == len(images)
        check_graph(morse, record, sphere=True)
        CMGDB.SaveMorseSets(morse, str(dst / "morse-sets.csv"))
        (dst / "morse-graph.dot").write_text(CMGDB.PlotMorseGraph(morse).source)
        for name in ("native-result.json", "callback-images.json", "vertex-images.json"):
            shutil.copy2(src / name, dst / name)
        expected = np.loadtxt(src / "morse-sets.csv", delimiter=",")
        actual = np.loadtxt(dst / "morse-sets.csv", delimiter=",")
        assert np.array_equal(expected, actual)
        reports[case] = {"cells": 2**depth, "morse_nodes": len(record["nodes"]),
                         "edges_cells_and_annotations_match": True}
        print(f"sphere {case}: {len(record['nodes'])} nodes, cells and annotations match", flush=True)
    return reports


def basins(morse, phase, boxes, record, expected, path):
    roles = {row["anchors"][0]: row["node"] for row in record["nodes"] if len(row["anchors"]) == 1}
    singleton = CMGDB.MorseSingletonReachability(phase, morse, np.arange(len(boxes), dtype=np.uint64))
    selected = CMGDB.MorseDirectedPathCells(phase, morse, [roles["O"]], [roles["C"]])
    _, targets = phase.csr_view()
    values = {"boxes": boxes, "singleton": singleton, "path_cells": selected,
              "image_cells": np.unique(targets)}
    with np.load(expected, allow_pickle=False) as saved:
        for key, value in values.items():
            assert np.array_equal(value, saved[key]), key
    np.savez_compressed(path, **values)


def aggregate(model, morse, phase, boxes, record, source, destination, expected):
    saddle, = [row["node"] for row in record["nodes"] if row["anchors"] == ["C"]]
    extra, = [row["node"] for row in record["nodes"] if not row["anchors"]]
    assert list(morse.annotations(extra)) == ["0", "0", "0"]
    assert [extra, saddle] in record["edges"]
    cells = sorted(map(int, CMGDB.MorseDirectedPathCells(phase, morse, [extra], [saddle])))
    assert set(cells) == set(morse.morse_set(extra)) | set(morse.morse_set(saddle))
    annotation = list(CMGDB.ComputeConleyIndexForCells(model, morse, cells))
    assert annotation == ["0", "x-1", "0"]
    kept = sorted(node for node in morse.vertices() if node != extra)
    new_id = {node: i for i, node in enumerate(kept)}
    new_id[extra] = new_id[saddle]
    edges = {i: set() for i in range(len(kept))}
    for src, dst in morse.edges():
        src, dst = new_id[src], new_id[dst]
        if src != dst:
            edges[src].add(dst)
    edges = {node: sorted(children) for node, children in transitive_reduction(list(edges), edges).items()}
    nodes = []
    for row in record["nodes"]:
        old = row["node"]
        if old == extra:
            continue
        merged = old == saddle
        ids = cells if merged else row["cell_ids"]
        nodes.append(dict(row, node=new_id[old], original_nodes=[saddle, extra] if merged else [old],
                          cell_ids=ids, cells=len(ids), annotations=annotation if merged else row["annotations"],
                          hull=[*boxes[ids, :2].min(axis=0), *boxes[ids, 2:].max(axis=0)]))
    reference = read(expected / "aggregate-results.json")
    assert nodes == reference["nodes"]
    assert sorted((src, dst) for src, children in edges.items() for dst in children) == sorted(map(tuple, reference["edges"]))
    graph = DerivedGraph(nodes=list(range(len(kept))), edges=edges,
                         labels={row["node"]: f'({", ".join(row["annotations"])})' for row in nodes})
    destination.mkdir()
    (destination / "aggregate-morse-graph.dot").write_text(CMGDB.plot_derived_graph(graph).source)
    rows = [[*boxes[cell], node["node"]] for node in nodes for cell in node["cell_ids"]]
    np.savetxt(destination / "aggregate-morse-sets.csv", rows, delimiter=",", fmt=["%.17g"] * 4 + ["%d"])
    reference["source_result_sha256"] = digest(source / "native-results.json")
    write(destination / "aggregate-results.json", reference)


def replay_o3(source, output, recompute_explicit=False):
    output.mkdir(parents=True)
    reports = {}
    for case in ("explicit", "fitted"):
        src, dst = source / case, output / case
        dst.mkdir()
        record = read(src / "native-results.json")
        with np.load(src / "native-phase.npz", allow_pickle=False) as saved:
            boxes, images = saved["boxes"], saved["image_boxes"]
        if recompute_explicit and case == "explicit":
            from exact_enclosures import o3_box_images
            computed = o3_box_images(boxes)
            assert np.array_equal(images, computed)
            images = computed
        image_by_box = dict(zip(map(tuple, boxes), images.tolist(), strict=True))
        depth = sum(n.bit_length() - 1 for n in record["grid_shape"])
        model = CMGDB.Model(depth, depth, depth, 10000, record["domain_lower"], record["domain_upper"],
                            lambda box: image_by_box[tuple(box)])
        model.set_batch_map(lambda batch: [image_by_box[tuple(box)] for box in batch])
        morse, phase = CMGDB.ComputeConleyMorseGraph(model, cache_transition_graph=True,
                            cache_map_graph=True, batch_chunk_size=512, reserve_min_edges=0)
        assert phase.num_vertices() == len(boxes)
        assert np.array_equal(boxes, np.asarray([morse.phase_space_box(i) for i in range(len(boxes))]))
        check_graph(morse, record)
        CMGDB.SaveMorseSets(morse, str(dst / "native-morse-sets.csv"))
        (dst / "native-morse-graph.dot").write_text(CMGDB.PlotMorseGraph(morse).source)
        for name in ("native-phase.npz", "native-results.json"):
            shutil.copy2(src / name, dst / name)
        basins(morse, phase, boxes, record, source / f"{case}-basins.npz", output / f"{case}-basins.npz")
        if case == "explicit":
            aggregate(model, morse, phase, boxes, record, dst,
                      output / "explicit-aggregate", source / "explicit-aggregate")
        reports[case] = {"cells": len(boxes), "morse_nodes": len(record["nodes"]),
                         "edges_cells_annotations_and_regions_match": True}
        print(f"o3 {case}: {len(record['nodes'])} native nodes, cells, annotations, and regions match", flush=True)
    summary = read(source / "summary.json")
    for case, row in summary["runs"].items():
        row["inputs_sha256"] = {name: digest(output / case / name) for name in row["inputs_sha256"]}
    write(output / "summary.json", summary)
    return reports


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recompute-explicit", action="store_true",
                        help="Recompute the explicit image enclosures before running CMGDB.")
    args = parser.parse_args()
    if version("CMGDB") != CMGDB_VERSION:
        raise RuntimeError(f"These saved results require CMGDB {CMGDB_VERSION}")
    verify_data(args.data)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"cmgdb_version": version("CMGDB"), "explicit_images_recomputed": args.recompute_explicit,
              "sphere": replay_sphere(args.data / "sphere", args.out / "sphere", args.recompute_explicit),
              "o3": replay_o3(args.data / "o3", args.out / "o3", args.recompute_explicit)}
    write(args.out / "verification.json", report)
    manifest = {p.relative_to(args.out).as_posix(): digest(p) for p in sorted(args.out.rglob("*")) if p.is_file()}
    write(args.out / "manifest.json", manifest)
    print("All graph, index, cell, and displayed-region comparisons passed.", flush=True)


if __name__ == "__main__":
    main()
