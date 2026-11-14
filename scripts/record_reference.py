"""Record every result of the selected fit of one example as the reference of its notebook.

    python scripts/record_reference.py ex1_so2_flow

Loads the fit named in `data/<example>/selection.json` with the settings of
`data/<example>/config.json`, computes the results that the example's notebook checks
(Morse graphs raw and merged, attracting regions, tolerances, residuals, reconstructed
indices, diagnostics) with the function `reference_record` of the example module, and
writes them to `data/<example>/reference/results.json`. In replay the notebook recomputes
the same record and compares it with this file: graphs and indices exactly, other numbers
to relative tolerance 1e-9. The analysis runs on the CPU, in the precision of the example.
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
    parser.add_argument("--replace", action="store_true", help="overwrite an existing results.json")
    args = parser.parse_args()
    from morsegraphs import device, training
    name, precision = EXAMPLES[args.example]
    device.configure(precision, "cpu")
    module = importlib.import_module(f"morsegraphs.{name}")
    if not hasattr(module, "reference_record"):
        sys.exit(f"morsegraphs.{name} has no reference_record function yet.")
    folder = ROOT / "data" / args.example
    target = folder / "reference" / "results.json"
    if target.exists() and not args.replace:
        sys.exit(f"{target.relative_to(ROOT)} exists; pass --replace to record it again.")
    start = time.perf_counter()
    record = module.reference_record(folder)
    training.write(target, record)
    print(f"wrote {target.relative_to(ROOT)} in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
