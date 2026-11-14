"""Store the fits of a sweep in `data/<example>/models/` and record the selection.

    python scripts/select_seed.py ex1_so2_flow

Every fit of `outputs/sweeps/<example>/` is kept, with its weights, settings, training
history and held-out diagnostics. The fit with the lowest validation objective is the
selected one; the ranking uses validation data only and precedes any CMGDB computation.
"""
import argparse
import importlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_seeds import EXAMPLES, VARIANTS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("--sweep", type=Path, help="the default is outputs/sweeps/<example>")
    args = parser.parse_args()
    from morsegraphs import device, training
    name, precision = EXAMPLES[args.example]
    device.configure(precision, "cpu")
    module = importlib.import_module(f"morsegraphs.{name}")
    sweep = args.sweep or ROOT / "outputs" / "sweeps" / args.example
    selection = training.read(sweep / "selection.json")
    target = ROOT / "data" / args.example / "models"
    if target.exists():
        sys.exit(f"{target.relative_to(ROOT)} exists; remove it to store another sweep.")
    key = VARIANTS.get(args.example, (None,))[0]
    for group, chosen in selection.items():
        if group == "failed":
            continue
        data = module.load_data(ROOT / "data" / args.example, **({key: group} if key else {}))
        for row in chosen["ranking"]:
            source, folder = sweep / row["run"], target / row["run"]
            folder.mkdir(parents=True)
            for item in source.iterdir():
                if item.name != "status.json" and item.is_file():
                    shutil.copy2(item, folder / item.name)
            metrics = module.diagnostics(module.load(folder), data)
            metrics.pop("arrays", None)
            training.write(folder / "metrics.json", metrics)
            print(f"{row['run']}: validation {row['validation']:.6g}", flush=True)
        print(f"selected {chosen['selected']}", flush=True)
    training.write(target.parent / "selection.json", selection)
    print(f"stored the fits in {target.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
