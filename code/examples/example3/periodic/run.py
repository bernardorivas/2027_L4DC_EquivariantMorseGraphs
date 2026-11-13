"""Run the separate autonomous coning stabilization experiment."""

import argparse
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
from common.workflow import new_output, read, require_environment, run_module, verify_inputs, write


def replay(destination):
    from example3.periodic import analysis, audit, data, validation
    verify_inputs(HERE)
    reference = HERE/"reference"
    config = read(reference/"requested-config.json")
    out = new_output(destination)
    data.generate(out/"data")
    validation.physical_checks(out/"controller")
    trained = out/"trained"
    trained.mkdir()
    write(trained/"requested-config.json", config)
    for policy in config["policies"]:
        for seed in config["seeds"]:
            relative = Path(policy)/f"seed{seed}"
            shutil.copytree(reference/relative/"model", trained/relative/"model")
    summary = analysis.run(trained, out/"data", config["policies"], config["seeds"])
    expected_rows = {(row["policy"], row["seed"]): row for row in read(reference/"summary.json")}
    checks = {}
    def compare(expected, actual):
        if isinstance(expected, dict):
            return all(compare(value, actual[key]) for key, value in expected.items())
        if isinstance(expected, list):
            return len(expected) == len(actual) and all(compare(a, b) for a, b in zip(expected, actual, strict=True))
        if isinstance(expected, float):
            return bool(np.isclose(expected, actual, rtol=1e-4, atol=3e-6))
        return expected == actual
    for row in summary:
        relative = Path(row["policy"])/f"seed{row['seed']}"
        expected = read(reference/relative/"metrics.json")
        assert compare(expected, row["metrics"]), f"Metric mismatch for {relative}"
        assert row["graph"] == read(reference/relative/"cmgdb/results.json"), f"Graph mismatch for {relative}"
        expected_row = expected_rows[(row["policy"], row["seed"])]
        assert row["matched_reference"] == expected_row["matched_reference"]
        assert row["physical_reference"] == expected_row["physical_reference"]
        aggregate = row["aggregate_graph"]
        assert aggregate["aggregation"]["index_pairs_checked"]
        assert len(aggregate["nodes"]) == (2 if row["policy"] == "nominal" else 3)
        with np.load(reference/relative/"cmgdb/phase.npz") as a, np.load(trained/relative/"cmgdb/phase.npz") as b:
            for name in a.files:
                if name == "images":
                    np.testing.assert_allclose(a[name], b[name], rtol=1e-4, atol=3e-6)
                else:
                    np.testing.assert_array_equal(a[name], b[name])
        checks[str(relative)] = dict(metrics=True, native_graph=True, transition_graph=True, basin_labels=True,
                                     aggregate_graph=True, aggregate_index_pairs=True)
    assert read(trained/"selection.json") == read(reference/"selection.json")
    audit.run(trained, out/"data")
    write(trained/"replay-verification.json", dict(passed=True, checks=checks, input_files=verify_inputs(HERE)))
    print(trained/"periodic-orbit-stabilization.pdf", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("replay")
    frozen.add_argument("--out", type=Path, required=True)
    presentation = commands.add_parser("presentation", help="Render the concise example and paired original/learned panels")
    presentation.add_argument("--out", type=Path, required=True)
    data = commands.add_parser("data")
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--sizes", type=int, nargs=3, default=[8192, 2048, 4096])
    for name in ("train", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        command.add_argument("--policies", nargs="+", choices=("nominal", "coning"), default=["nominal", "coning"])
        if name == "train":
            command.add_argument("--steps", type=int, default=6000)
            command.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args(argv)
    versions = require_environment(graphics=args.command != "data")
    if args.command == "presentation":
        verify_inputs(HERE)
        out = new_output(args.out)
        run_module("common.render_attitude", "--out", out, "--presentation")
        return
    if args.command == "replay":
        replay(args.out)
        return
    if args.command == "data":
        from example3.periodic.data import generate
        generate(args.out, args.sizes)
        return
    from example3.periodic import model
    if args.command == "train":
        out = new_output(args.out)
        write(out/"requested-config.json", dict(seeds=args.seeds, policies=args.policies,
              settings=model.DEFAULTS | dict(steps=args.steps), versions=versions,
              experiment="constructed periodic-axis stabilization with task-aware calibration",
              data_hashes=read(args.data/"metadata.json")))
        for policy in args.policies:
            with np.load(args.data/f"{policy}-train.npz") as saved:
                observations = saved["observations"]
            with np.load(args.data/f"{policy}-validation.npz") as saved:
                validation = saved["observations"]
            for seed in args.seeds:
                model.train(out/f"{policy}/seed{seed}/model", observations, validation, seed, dict(steps=args.steps))
        if args.skip_analysis:
            return
    from example3.periodic import analysis
    analysis.run(args.out, args.data, args.policies, args.seeds)


if __name__ == "__main__":
    main()
