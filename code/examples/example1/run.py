"""Replay the frozen interval fit, train a new fit, or reproduce manuscript panels."""

import argparse
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.workflow import (compare_metrics, digest, new_output, read, require_environment,
                             run_module, verify_inputs, write)


def joint(args):
    import numpy as np
    from example1 import training

    config = read(args.config)
    if args.command == "train" and args.steps is not None:
        if args.steps < 1:
            raise ValueError("The number of training steps must be positive.")
        config["training"]["steps"] = args.steps
    if config["analysis"]["time_map_iterates"] != 10 or config["analysis"]["image_padding"] != 1e-10:
        raise ValueError("The current analysis uses ten iterates and image padding 1e-10.")
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    config["seeds"] = seeds
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be distinct.")
    source = HERE / "reference/joint"
    if args.command == "replay" and any(not (source / f"seed{seed}").is_dir() for seed in seeds):
        raise ValueError("The bundled replay supports seeds 0, 1, and 2.")
    out = new_output(args.out)
    shutil.copy2(HERE / "data/trajectories.npz", out / "physical-trajectories.npz")
    write(out / "requested-config.json", config)
    if args.command == "replay":
        shutil.copytree(source / "reference", out / "reference")
        for seed in seeds:
            shutil.copytree(source / f"seed{seed}", out / f"seed{seed}")
    data = np.load(out / "physical-trajectories.npz", allow_pickle=False)
    depths = config["analysis"]["depths"]
    training.reference(out, depths)
    reports, checks = [], {}
    for seed in seeds:
        folder = out / f"seed{seed}"
        if args.command == "replay":
            model = training.load(folder)
        else:
            settings = dict(config["training"])
            if args.steps is not None:
                settings["steps"] = args.steps
            model = training.train(folder, data, seed, recipe=config["recipe"], **settings)
        metrics = training.diagnostics(model, data, folder)
        if not metrics["sampled_encoder_increasing"]:
            raise ValueError(f"Seed {seed} failed the sampled encoder ordering check. The failed fit is saved.")
        graphs = training.analyze(model, folder, depths, config["analysis"]["image_samples"])
        if (source / f"seed{seed}").is_dir():
            mismatch = None
            try:
                count = compare_metrics(read(source / f"seed{seed}/metrics.json"), metrics)
            except ValueError as error:
                if args.command == "replay":
                    raise
                count, mismatch = 0, str(error)
            expected = read(source / f"seed{seed}/config.json")["best_validation"]
            validation = float(training.objective(model, *[data[f"validation_{key}"]
                               for key in ("p0", "p1", "w0", "w1")]))
            validation_match = bool(np.isclose(expected, validation, rtol=1e-6, atol=1e-9))
            if args.command == "replay" and not validation_match:
                raise ValueError(f"Seed {seed} has a different validation objective.")
            graph_match = True
            for depth in depths:
                saved = source / f"seed{seed}/K{2**depth}/native-result.json"
                if not saved.exists():
                    graph_match = False
                    break
                old, new = read(saved), read(folder / f"K{2**depth}/native-result.json")
                graph_match &= old["nodes"] == new["nodes"] and old["edges"] == new["edges"]
            checks[str(seed)] = {"diagnostic_values_checked": count,
                                "diagnostics_match": mismatch is None, "diagnostic_mismatch": mismatch,
                                "validation_objective": validation, "reference_validation_objective": expected,
                                "validation_matches": validation_match,
                                "cells_edges_and_annotations_match": bool(graph_match),
                                "morse_sets": {str(row["cells"]): row["morse_sets"] for row in graphs}}
        reports.append({"seed": seed, "metrics": metrics, "cmgdb": graphs})
    data.close()
    write(out / "summary.json", reports)
    run_module("example1.figures", "--run", out, "--seeds", *seeds)
    verification = {"experiment": config["experiment"], "mode": args.command,
                    "physical_data_sha256": digest(out / "physical-trajectories.npz"),
                    "reference_checks": checks, "versions": args.versions,
                    "image_status": "sampled learned image hulls; not validated enclosures"}
    write(out / "verification.json", verification)
    print(f"Completed {args.command}: {out}", flush=True)
    print(f"Comparison: {out / 'sphere-emlp-comparison.pdf'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "train"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--seeds", type=int, nargs="+")
        if name == "train":
            command.add_argument("--config", type=Path, default=HERE / "config.json")
            command.add_argument("--steps", type=int)
        else:
            command.set_defaults(config=HERE / "reference/joint/config.json")
    manuscript = commands.add_parser("manuscript", help="Recompute and redraw the older saved manuscript fit")
    manuscript.add_argument("--out", type=Path, required=True)
    tolerances = commands.add_parser("tolerances", help="Recompute attracting-region margins for a checkpoint")
    from common.workflow import attracting_region_arguments
    attracting_region_arguments(tolerances, 1)
    args = parser.parse_args()
    try:
        args.versions = require_environment(graphics=args.command != "tolerances")
        if args.command == "tolerances":
            from common.attracting_region_models import example1
            example1(args)
        elif args.command == "manuscript":
            from common.workflow import manuscript_run
            manuscript_run("example1", args.out)
        else:
            verify_inputs(HERE)
            joint(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
