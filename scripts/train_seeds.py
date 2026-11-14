"""Train several seeds of one example in parallel processes, then rank them by validation error.

    python scripts/train_seeds.py ex1_so2_flow --seeds 0 1 2 3 4 --parallel 5

Each seed trains in its own process on the device chosen by `morsegraphs.device`, with the
settings of `data/<example>/config.json` and the data in `data/<example>/`. Results go to
`outputs/sweeps/<example>/<run>/`, where `status.json` reports progress while it trains.
The ranking by validation objective, made before any CMGDB computation, is written to
`outputs/sweeps/<example>/selection.json`; `scripts/select_seed.py` then stores the fits.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXAMPLES = {"ex1_so2_flow": ("so2_flow", "float64"),
            "ex2_o3_flow": ("o3_flow", "float64"),
            "ex3_attitude_control": ("attitude_control", "float32")}
# Section 5.3 trains each seed once for each gain, and selects a fit for each gain.
VARIANTS = {"ex3_attitude_control": ("policy", ("nominal", "coning"))}


def settings_of(example, module, config=None):
    path = Path(config) if config else ROOT / "data" / example / "config.json"
    return json.loads(path.read_text()) if path.exists() else module.DEFAULTS


def runs(example, seeds):
    """The runs of a sweep: one per seed, and per gain in Section 5.3."""
    if example not in VARIANTS:
        return [dict(name=f"seed{seed}", seed=seed, group="", data_options={}) for seed in seeds]
    key, values = VARIANTS[example]
    return [dict(name=f"{value}-seed{seed}", seed=seed, group=value, data_options={key: value})
            for value in values for seed in seeds]


def rank(folders_by_group):
    """The ranking by validation objective within each group of runs."""
    from morsegraphs import training
    result = {}
    for group, folders in folders_by_group.items():
        if folders:
            result[group or "all"] = training.select(folders)
    return result


def child(args):
    from morsegraphs import device, training
    name, precision = EXAMPLES[args.example]
    choice = device.configure(precision, args.device)
    module = importlib.import_module(f"morsegraphs.{name}")
    settings = settings_of(args.example, module, args.config)
    options = settings.get("training", settings)  # Section 5.3 keeps its training settings at the top level
    if args.steps is not None:
        options["steps"] = args.steps
    run = json.loads(args.run)
    folder = Path(args.out) / run["name"]
    progress = training.Progress(folder, args.example, run["name"], options["steps"], choice["device"])
    try:
        data = module.load_data(ROOT / "data" / args.example, **run["data_options"])
        progress.update(status="training")
        module.train(folder, data, run["seed"], settings, progress)
    except Exception as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--parallel", type=int, default=5, help="processes at a time")
    parser.add_argument("--steps", type=int, help="training updates; the default is the example's config")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="the default tries MPS, CUDA, then CPU")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path, help="settings to use instead of data/<example>/config.json")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = args.out or ROOT / "outputs" / "sweeps" / args.example
    if args.child:
        child(args)
        return
    planned = runs(args.example, args.seeds)
    if any((args.out / run["name"]).exists() for run in planned):
        sys.exit(f"{args.out} already holds some of these runs; choose another --out.")
    args.out.mkdir(parents=True, exist_ok=True)
    pending, active, failed = list(planned), [], []
    start = time.perf_counter()
    while pending or active:
        while pending and len(active) < args.parallel:
            run = pending.pop(0)
            command = [sys.executable, __file__, args.example, "--child", "--run", json.dumps(run),
                       "--out", str(args.out)]
            if args.steps is not None:
                command += ["--steps", str(args.steps)]
            if args.device is not None:
                command += ["--device", args.device]
            if args.config is not None:
                command += ["--config", str(args.config.resolve())]
            log = open(args.out / f"{run['name']}.log", "w")
            active.append((run, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(2)
        for item in list(active):
            run, process, log = item
            if process.poll() is not None:
                log.close()
                active.remove(item)
                if process.returncode:
                    failed.append(run["name"])
                print(f"{run['name']}: {'failed' if process.returncode else 'trained'} "
                      f"after {time.perf_counter() - start:.0f} s", flush=True)
    from morsegraphs import training
    groups = {}
    for run in planned:
        if run["name"] not in failed:
            groups.setdefault(run["group"], []).append(args.out / run["name"])
    selection = rank(groups)
    if selection:
        training.write(args.out / "selection.json", dict(selection, failed=failed))
        for group, chosen in selection.items():
            for row in chosen["ranking"]:
                print(f"{row['run']}: validation {row['validation']:.6g}", flush=True)
            print(f"selected {chosen['selected']}", flush=True)
    if failed:
        sys.exit(f"failed runs: {', '.join(failed)}; see their .log files in {args.out}")


if __name__ == "__main__":
    main()
