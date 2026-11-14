"""Generate the data of one example and write its settings.

    python scripts/generate_data.py ex1_so2_flow

Writes `data/<example>/config.json` (the example's settings, with the seeds of the sweep)
and the data arrays with a `metadata.json` of sizes, generator seeds and SHA-256 checksums.
The generators are deterministic, so a rerun reproduces the files.
"""
import argparse
import copy
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_seeds import EXAMPLES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    from morsegraphs import device, training
    name, precision = EXAMPLES[args.example]
    device.configure(precision, "cpu")
    module = importlib.import_module(f"morsegraphs.{name}")
    folder = ROOT / "data" / args.example
    settings = copy.deepcopy(module.DEFAULTS)
    settings["seeds"] = args.seeds
    folder.mkdir(parents=True, exist_ok=True)
    training.write(folder / "config.json", settings)
    module.generate_data(folder, settings)
    print(f"wrote {folder.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
