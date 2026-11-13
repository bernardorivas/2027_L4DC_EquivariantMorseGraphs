"""Analyze learned dynamics using only encoded physical source samples."""

from hashlib import sha256
from pathlib import Path
import shutil

import CMGDB
import numpy as np

from common.workflow import compare_metrics, digest, new_output, read, run_module, write
from . import analysis, basins, training
from .run import basin_anchor, geometry, load_data

HERE = Path(__file__).resolve().parent
CONTROLS = np.array([[0., 0.], [0., 1.], [-1., 0.], [1., 0.]])


def source_pairs(model, folder, sizes):
    """Encode one physical representative per orbit on nested section grids."""
    largest = max(sizes)
    a, r = np.linspace(-1.5, 1.5, largest), np.linspace(0., 1.5, largest)
    aa, rr = np.meshgrid(a, r, indexing="ij")
    physical = training.section(np.column_stack((aa.ravel(), rr.ravel())))
    encode = training.evaluator(model, model.encode)
    advance = training.evaluator(model, model.advance)
    codes = training.batched(encode, physical).reshape(largest, largest, 2)
    values = training.batched(advance, codes.reshape(-1, 2)).reshape(largest, largest, 2)
    controls = training.batched(encode, training.section(CONTROLS))
    control_images = training.batched(advance, controls)
    # Physical equilibrium controls supplement every grid, without using test data.
    np.savez_compressed(folder / "source-pairs.npz", physical_a=a, physical_r=r,
                        encoded_sources=codes, learned_images=values,
                        physical_controls=CONTROLS, encoded_controls=controls, control_images=control_images)
    for size in sizes:
        stride = (largest-1)//(size-1)
        x = np.concatenate((codes[::stride, ::stride].reshape(-1, 2), controls))
        y = np.concatenate((values[::stride, ::stride].reshape(-1, 2), control_images))
        assert len(x) == size*size + len(CONTROLS)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert np.all((x >= training.LOWER) & (x <= training.UPPER))
        yield size, x, y


def box_map(x, y):
    return CMGDB.BoxMapData(x, y, map_empty="outside",
            lower_bounds=training.LOWER.tolist(), upper_bounds=training.UPPER.tolist(),
            domain_padding=False, padding=False)


def image_boxes(function, boxes):
    """Use native data-map queries for images and exact closed-cell occupancy."""
    images = np.asarray(function.batch(boxes))
    counts = np.asarray([len(function.map_points(box.tolist())) for box in boxes], dtype=np.int64)
    assert np.all(images[counts == 0, :2] > training.UPPER)
    return images, counts


def compare_reference(folder, frozen):
    if not frozen.exists():
        return {"available": False}
    expected, actual = read(frozen / "native-results.json"), read(folder / "native-results.json")
    pair_hashes_match = True
    for key in ("encoded_sources_sha256", "learned_images_sha256"):
        pair_hashes_match &= expected["construction"].pop(key) == actual["construction"].pop(key)
    assert expected == actual, f"Data-map record changed: {folder}"
    with np.load(frozen / "native-phase.npz") as old, np.load(folder / "native-phase.npz") as new:
        assert set(old.files) == set(new.files)
        difference = float(np.max(np.abs(old["image_boxes"]-new["image_boxes"])))
        assert difference < 2e-13
        for key in old.files:
            if key not in {"image_boxes", "unmasked_image_boxes"}:
                assert np.array_equal(old[key], new[key]), (folder, key)
    assert read(frozen / "basins.json").get("available", True) == read(folder / "basins.json").get("available", True)
    if (frozen / "basins.npz").exists():
        with np.load(frozen / "basins.npz") as old, np.load(folder / "basins.npz") as new:
            assert set(old.files) == set(new.files)
            assert all(np.array_equal(old[key], new[key]) for key in old.files)
    result = {"available": True, "passed": True, "maximum_image_endpoint_difference": difference,
              "source_pair_hashes_match": pair_hashes_match,
              "cells_edges_annotations_occupancy_and_basins_match": True}
    write(folder / "data-replay-verification.json", result)
    return result


def density_change(previous, current):
    """Check inclusion for the nested sample sets and compare their cell graphs."""
    if previous is None:
        return None
    with np.load(previous / "native-phase.npz") as old, np.load(current / "native-phase.npz") as new:
        assert np.array_equal(old["boxes"], new["boxes"])
        active = old["active"]
        assert np.all(new["occupancy_counts"] >= old["occupancy_counts"])
        assert np.all(new["active"][active])
        assert np.all(new["image_boxes"][active, :2] <= old["image_boxes"][active, :2])
        assert np.all(new["image_boxes"][active, 2:] >= old["image_boxes"][active, 2:])
        counts = new["occupancy_counts"][new["active"]]
        old_edges = {(i, int(j)) for i in np.flatnonzero(active)
                     for j in old["csr_targets"][old["csr_offsets"][i]:old["csr_offsets"][i+1]]}
        new_edges = {(i, int(j)) for i in range(len(active))
                     for j in new["csr_targets"][new["csr_offsets"][i]:new["csr_offsets"][i+1]]}
        assert old_edges <= new_edges
        old_record, new_record = read(previous / "native-results.json"), read(current / "native-results.json")
        result = {"nested_sample_image_inclusion_checked": True,
                  "new_supported_cells": int(np.sum(new["active"] & ~active)),
                  "added_transition_edges": len(new_edges-old_edges),
                  "same_morse_cells_edges_and_annotations": old_record["nodes"] == new_record["nodes"] and old_record["edges"] == new_record["edges"],
                  "minimum_sources_per_supported_cell": int(counts.min())}
    if (previous / "basins.npz").exists() and (current / "basins.npz").exists():
        with np.load(previous / "basins.npz") as old, np.load(current / "basins.npz") as new:
            # Compare signed sinks rather than numeric node IDs, which can change.
            def signs(values):
                labels = values["safe_sink_node"]
                return np.where(labels == values["sink_nodes"][0], -1,
                                np.where(labels == values["sink_nodes"][1], 1, 0))
            result["changed_signed_basin_cells"] = int(np.sum(signs(old) != signs(new)))
    return result


def configuration(args):
    frozen = HERE / "reference/data-map/config.json"
    config = read(frozen if args.command == "replay" and frozen.exists() else
                  HERE / "config.json" if args.command == "replay" else args.config)
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if "data_map" not in config["analysis"]:
        raise ValueError("Use the current Example 2 config with analysis.data_map settings.")
    settings = config["analysis"]["data_map"]
    if args.source_sizes is not None:
        settings["source_sizes"] = args.source_sizes
    sizes = settings["source_sizes"]
    if not sizes or sizes != sorted(set(sizes)) or any(s < 3 or (s-1) & (s-2) for s in sizes):
        raise ValueError("Use increasing distinct source sizes of the form 2^k+1, with k >= 1.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Choose distinct model seeds.")
    depths = config["analysis"]["depths"]
    if depths != [6] or config["analysis"]["primary_depth"] != 6:
        raise ValueError("Example 2 uses a fixed 64 x 64 latent grid.")
    if settings["map_empty"] != "outside" or settings["domain_padding"] or settings["padding"]:
        raise ValueError("The data construction requires map_empty=outside, domain_padding=false, and padding=false.")
    if config["analysis"]["physical_section_points_per_axis"] != 513:
        raise ValueError("The physical display uses 513 section points per axis.")
    if not 0 <= config["recipe"]["encoder_gain_log_bound"] < np.log(4/3):
        raise ValueError("The encoder gain must keep all physical codes inside the latent rectangle.")
    if args.command == "replay" and any(s not in (0, 1, 2) for s in config["seeds"]):
        raise ValueError("Bundled replay supports seeds 0, 1, and 2.")
    if args.command == "train" and args.steps is not None:
        config["training"]["steps"] = args.steps
    if config["training"]["steps"] < 1:
        raise ValueError("Training steps must be positive.")
    if settings["primary_source_size"] not in sizes:
        settings["primary_source_size"] = max(sizes)
    config["analysis"] = {"depths": [6], "primary_depth": 6,
                           "physical_section_points_per_axis": 513, "data_map": settings}
    return config


def joint(args):
    config = configuration(args)
    replay = args.command == "replay"
    source = HERE / "reference/joint"
    frozen = HERE / "reference/data-map"
    data_source = HERE / "data/trajectories.npz" if replay or args.data is None else args.data
    with load_data(data_source) as data:
        split_sizes = {name: len(data[f"{name}_x"]) for name in ("train", "validation", "test")}
    out = new_output(args.out)
    shutil.copy2(data_source, out / "physical-trajectories.npz")
    write(out / "requested-config.json", config)
    depths, seeds = config["analysis"]["depths"], config["seeds"]
    sizes = config["analysis"]["data_map"]["source_sizes"]
    physics = training.verify_physics(out)
    checks, summary = {}, []
    # The validation ranking is fixed before any learned graph is constructed.
    if replay:
        ranking = [{"seed": seed, "validation": read(source / f"seed{seed}/config.json")["best_validation"]}
                   for seed in seeds]
    else:
        ranking = []
        with load_data(data_source) as data:
            for seed in seeds:
                training.train(out / f"seed{seed}", data, seed, recipe=config["recipe"], **config["training"])
                ranking.append({"seed": seed, "validation": read(out / f"seed{seed}/config.json")["best_validation"]})
    ranking.sort(key=lambda row: row["validation"])
    write(out / "selection.json", {"criterion": "lowest validation loss, fixed before CMGDB",
                                   "selected_seed": ranking[0]["seed"], "ranking": ranking})
    for seed in seeds:
        folder = out / f"seed{seed}"
        if replay:
            folder.mkdir()
            for name in ("config.json", "weights.npz", "metrics.json", "history.json", "geometry.npz"):
                shutil.copy2(source / f"seed{seed}" / name, folder / name)
        model = training.load(folder)
        with load_data(data_source) as data:
            metrics = training.diagnostics(model, data, folder, restrict_axes=True)
            parts = training.evaluator(model, lambda x, y: training.loss_parts(model, x, y))
            score = float(np.asarray(parts(data["validation_x"][:, 0], data["validation_x"][:, 1])).sum())
        geometry(model, folder, replay)
        check = {"validation_objective": score}
        if replay:
            check["diagnostic_values_checked"] = compare_metrics(read(source / f"seed{seed}/metrics.json"), metrics)
            compare_metrics({"validation": read(folder / "config.json")["best_validation"]}, {"validation": score})
            check["diagnostics_match"] = True
        previous = {}
        seed_rows = []
        for size, x, y in source_pairs(model, folder, sizes):
            function = box_map(x, y)
            construction = {"method": "CMGDB.BoxMapData", "pairs": "(E(x), f(E(x)))",
                "source_section_size": size, "source_pairs": len(x),
                "physical_domain": [[-1.5, 0.], [1.5, 1.5]],
                "source_recipe": "uniform physical orbit section plus four physical equilibrium-orbit controls",
                "map_empty": "outside", "domain_padding": False, "padding": False,
                "learned_map_evaluations_outside_encoded_sources": 0,
                "encoded_sources_sha256": sha256(x.tobytes()).hexdigest(),
                "learned_images_sha256": sha256(y.tobytes()).hexdigest()}
            for depth in depths:
                n = 2**depth
                target = folder / f"S{size}/K{n}"
                boxes = analysis.grid(n, training.LOWER, training.UPPER)
                images, counts = image_boxes(function, boxes)
                record, *_ = analysis.native(target, boxes, images, training.LOWER, training.UPPER, n,
                    "sampled images of encoded physical sources; not validated", support_counts=counts,
                    box_map=function, construction=construction)
                analysis.physical_preimages(model, target, n, folder / "geometry.npz")
                radial, reason = basin_anchor(target, metrics)
                if reason is None:
                    basin = basins.compute(target, radial)
                else:
                    basin = {"available": False, "reason": reason, "grid": n}
                    write(target / "basins.json", basin)
                replay_check = compare_reference(target, frozen / f"seed{seed}/S{size}/K{n}") if replay else {"available": False}
                change = density_change(previous.get(n), target)
                previous[n] = target
                covered = counts[counts > 0]
                row = {"seed": seed, "source_size": size, "source_pairs": len(x), "grid": n,
                       "nodes": len(record["nodes"]), "edges": record["edges"],
                       "annotations": [node["annotations"] for node in record["nodes"]],
                       "supported_cells": int(len(covered)), "unsupported_cells": int(np.sum(counts == 0)),
                       "minimum_sources_per_supported_cell": int(covered.min()),
                       "median_sources_per_supported_cell": float(np.median(covered)),
                       "known_to_unsampled_edges": record["known_to_unsampled_edges"],
                       "basins_available": basin.get("available", True),
                       "basin_counts": basin.get("counts"), "replay": replay_check,
                       "change_from_previous_density": change}
                seed_rows.append(row)
                write(target / "sampling-summary.json", row)
        check["data_map_replays"] = [row["replay"] for row in seed_rows]
        checks[str(seed)] = check
        summary.extend(seed_rows)
    write(out / "data-map-summary.json", summary)
    run_module("example2.data_figures", "--run", out)
    write(out / "verification.json", {"experiment": "joint-quotient-data-map-v1", "mode": args.command,
          "construction": "CMGDB.BoxMapData on (E(x), f(E(x)))", "versions": args.versions,
          "physical_data_sha256": digest(data_source), "split_sizes": split_sizes,
          "reference_checks": checks, "physical_verification": physics,
          "source_sizes": sizes, "primary_depth": config["analysis"]["primary_depth"],
          "primary_source_size": config["analysis"]["data_map"]["primary_source_size"],
          "empty_cells": "unavailable; excluded from basin labels and never interpolated",
          "image_status": "sampled data hulls; not validated"})
    print(f"Completed {args.command}: {out}", flush=True)
    print(f"Results: {out / 'results.pdf'}", flush=True)
    print(f"Regions of attraction: {out / 'regions-of-attraction.pdf'}", flush=True)
