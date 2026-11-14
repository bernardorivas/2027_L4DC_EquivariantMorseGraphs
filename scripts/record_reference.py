"""Write the reference record that an example's notebook checks in replay.

    python scripts/record_reference.py ex3_attitude_control

Loads the selected fit of the example (of each gain in Section 5.3) and its bundled data,
recomputes every result the notebook checks with `reference_record(folder)` of the example
module, and writes it to `data/<example>/reference/results.json`. In replay the notebook
recomputes the same record and compares it with this file: Morse graphs exactly, numbers to
relative tolerance 1e-9. An existing record is replaced only with --overwrite.
"""
import argparse
import importlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_seeds import EXAMPLES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing record")
    args = parser.parse_args()
    from morsegraphs import device, training
    name, precision = EXAMPLES[args.example]
    device.configure(precision, "cpu")
    module = importlib.import_module(f"morsegraphs.{name}")
    if not hasattr(module, "reference_record"):
        sys.exit(f"morsegraphs.{name} has no reference_record(folder).")
    folder = ROOT / "data" / args.example
    target = folder / "reference" / "results.json"
    if target.exists() and not args.overwrite:
        sys.exit(f"{target.relative_to(ROOT)} exists; pass --overwrite to replace it.")
    start = time.perf_counter()
    record = module.reference_record(folder)
    training.write(target, record)
    print(f"wrote {target.relative_to(ROOT)} in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
