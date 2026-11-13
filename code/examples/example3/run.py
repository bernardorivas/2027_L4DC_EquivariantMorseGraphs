"""Two learned equivariant reductions and regular CMGDB on the interval."""

import argparse
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.workflow import (new_output, read, require_environment,
                             run_module, verify_inputs, write)


def experiment(args):
    from example3 import sphere, interval, analysis, data
    reference = HERE / "reference"
    replay = args.command == "replay"
    config = read(reference / "config.json" if replay else args.config)
    policies = config["policies"] if args.policy == "both" else [args.policy]
    seeds = config["seeds"] if args.seeds is None else args.seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose distinct model seeds.")
    if (replay or args.stage == "interval") and any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Bundled sphere models support seeds 0, 1, and 2.")
    if not replay and args.steps is not None:
        config["interval"]["steps"] = args.steps
    if not replay and args.sphere_steps is not None:
        config["sphere"]["steps"] = args.sphere_steps
    if args.cells is not None:
        config["analysis"]["cells"] = args.cells
    if config["interval"]["steps"] < 1 or config["sphere"]["steps"] < 1:
        raise ValueError("Training update counts must be positive.")
    if config["analysis"]["cells"] not in (64, 128):
        raise ValueError("Choose 64 or 128 interval cells for Example 3.")
    data_dir = HERE / "data" if args.data is None else args.data
    actual_sizes = {}
    for policy in policies:
        actual_sizes[policy] = {}
        for split in data.SPLITS:
            observations, *_ = data.load(data_dir, policy, split)
            actual_sizes[policy][split] = len(observations)
    config["data"]["actual_split_sizes"] = actual_sizes
    config["policies"], config["seeds"] = policies, seeds
    out = new_output(args.out)
    write(out / "requested-config.json", config)
    selection, summary, checks, models = {}, [], {}, {}
    for policy in policies:
        ranking = []
        for seed in seeds:
            folder = out / f"{policy}/seed{seed}"
            folder.mkdir(parents=True)
            source = reference / f"{policy}/seed{seed}"
            if replay or args.stage == "interval":
                shutil.copytree(source / "sphere", folder / "sphere")
                model = sphere.load(folder / "sphere")
            else:
                train, *_ = data.load(data_dir, policy, "train")
                validation, *_ = data.load(data_dir, policy, "validation")
                model = sphere.train(folder / "sphere", train, validation, seed, config["sphere"])
            models[(policy, seed)] = model
            ranking.append(dict(seed=seed, validation_prediction=read(folder / "sphere/config.json")["best_validation_prediction"]))
        ranking.sort(key=lambda row: (row["validation_prediction"], row["seed"]))
        selection[policy] = dict(selected_seed=ranking[0]["seed"], ranking=ranking,
                                criterion="full-plant validation prediction, fixed before CMGDB")
    write(out / "selection.json", selection)
    reference_checks = {}
    for policy in policies:
        result = analysis.reference(policy, out / f"thrust-axis-reference/{policy}", config["analysis"])
        saved = reference / f"time01/{policy}-K{config['analysis']['cells']}.json"
        expected = read(saved)
        if expected["image_padding"] == result["image_padding"]:
            if expected != result:
                raise ValueError(f"The matched thrust-axis reference changed for {policy}.")
            reference_checks[policy] = dict(passed=True, native_nodes=len(result["nodes"]),
                                             nonzero_nodes=len(result["nonzero_nodes"]))
        else:
            reference_checks[policy] = dict(status="new padding", native_nodes=len(result["nodes"]),
                                             nonzero_nodes=len(result["nonzero_nodes"]))
    for policy in policies:
        for seed in seeds:
            folder = out / f"{policy}/seed{seed}"
            source = reference / f"{policy}/seed{seed}"
            model = models[(policy, seed)]
            pairs = interval.make_pairs(model, folder, config["interval_data"]["sizes"])
            if replay:
                shutil.copytree(source / "interval", folder / "interval")
                reduced = interval.load(folder / "interval")
            else:
                reduced = interval.train(folder / "interval", pairs, seed, config["interval"])
            reduced_metrics = interval.diagnostics(reduced, pairs, folder / "interval")
            graph = analysis.native(reduced, folder / "cmgdb", config["analysis"])
            physical_metrics = analysis.sphere_diagnostics(model, reduced, data_dir, policy, folder)
            verification = {}
            if replay:
                verification["interval_values_checked"] = analysis.compare_metrics(read(source / "interval/metrics.json"), reduced_metrics)
                expected = read(source / "sphere-metrics.json")
                reference_cells = read(source / "cmgdb/results.json")["cells"]
                graph_reference = source / "cmgdb" if graph["cells"] == reference_cells else reference / f"cmgdb{graph['cells']}/{policy}/seed{seed}"
                if (graph_reference / "results.json").exists():
                    verification["cmgdb"] = analysis.compare_saved(folder / "cmgdb", graph_reference)
                else:
                    verification["cmgdb"] = dict(status="new resolution", cells=graph["cells"], reference_cells=reference_cells)
                if graph["cells"] != reference_cells:
                    dependent = ("physical_source_supported_cells", "physical_basin_labeled_cells",
                                 "physical_cells_reaching_unobserved_support")
                    expected = {key: value for key, value in expected.items() if key not in dependent}
                    verification["resolution_dependent_metrics_recomputed"] = list(dependent)
                verification["sphere_values_checked"] = analysis.compare_metrics(expected, physical_metrics)
            checks[f"{policy}/seed{seed}"] = verification
            summary.append(dict(policy=policy, seed=seed, interval=reduced_metrics, graph=graph,
                                validation_prediction=physical_metrics["validation"]["prediction"],
                                test_prediction=physical_metrics["test"]["prediction"],
                                physical_interval_consistency_rms=physical_metrics["physical_one_step_interval_consistency_rms"]))
            write(out / "summary.json", summary)
    run_module("example3.figures", "--run", out)
    write(out / "verification.json", dict(mode=args.command, policies=policies, seeds=seeds,
          sphere_training="retained completed fits" if replay or args.stage == "interval" else "fresh joint training",
          interval_training="retained newly trained fits" if replay else "fresh joint training",
          cells=config["analysis"]["cells"], versions=args.versions, reference_checks=checks,
          thrust_axis_reference_checks=reference_checks,
          numerical_status="sampled function hulls and physical diagnostics; no physical transfer certificate"))
    print(f"Completed {args.command}: {out}", flush=True)
    print(out / "results.pdf", flush=True)


def main():
    if sys.argv[1:2] == ["periodic"]:
        from example3.periodic.run import main as periodic_main
        return periodic_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("periodic", help="Constructed autonomous coning control, joint learning, and checked replay")
    for name in ("replay", "train"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--policy", choices=("nominal", "coning", "both"), default="both")
        command.add_argument("--seeds", type=int, nargs="+")
        command.add_argument("--data", type=Path)
        command.add_argument("--cells", type=int, choices=(64, 128),
                             help="Interval cells; bundled references use 64")
        command.set_defaults(stage="interval", steps=None, sphere_steps=None)
        if name == "train":
            command.add_argument("--config", type=Path, default=HERE / "config.json")
            command.add_argument("--stage", choices=("both", "interval"), default="both")
            command.add_argument("--steps", type=int, help="Second-stage update count")
            command.add_argument("--sphere-steps", type=int, help="First-stage update count")
    data = commands.add_parser("data")
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--policy", choices=("nominal", "coning", "both"), default="both")
    data.add_argument("--sizes", type=int, nargs=3, default=[16384, 2048, 4096])
    args = parser.parse_args()
    try:
        args.versions = require_environment(graphics=args.command != "data")
        if args.command == "data":
            if min(args.sizes) < 8:
                raise ValueError("Each split needs at least eight trajectories.")
            from example3.data import generate
            out = new_output(args.out)
            generate(out, ["nominal", "coning"] if args.policy == "both" else [args.policy], args.sizes)
        else:
            verify_inputs(HERE)
            experiment(args)
    except (FileExistsError, FileNotFoundError, FloatingPointError, RuntimeError, ValueError) as error:
        parser.exit(1, str(error)+"\n")


if __name__ == "__main__":
    main()
