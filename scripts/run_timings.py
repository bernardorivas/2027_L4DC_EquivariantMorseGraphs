#!/usr/bin/env python3
"""Reproduce Section 6 and Table tab:timing.

From the repository root:
    python scripts/run_timings.py smoke
    python scripts/run_timings.py all

The first command runs the two quotient rows and uniform grids with 2^12
(sphere) and 2^11 (O(3)) cells. These are validation records only.
The second runs all six published rows, taking about 40 minutes. It refuses
to start unless the one-minute load average is below 2 and 16 GB of memory
is available. The same check applies before each benchmark row. No override is
provided. Use --out outputs/timings/RUN to preserve a previous set of records.
Every row runs in a fresh process with the four numerical thread counts set
to one. Original-space rows retain the published repeat and warmup counts.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# These must be set before importing the numerical package in this process.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
sys.path.insert(0, str(ROOT))
from morsegraphs import timing  # noqa: E402


def selected_rows(selection):
    if selection == "all":
        return ["sphere-quotient", "o3-quotient", "sphere-uniform", "sphere-adaptive", "o3-uniform", "o3-adaptive"]
    if selection == "smoke":
        return ["sphere-quotient", "o3-quotient", "sphere-uniform", "o3-uniform"]
    return [selection]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selection", choices=["all", "smoke", *timing.ROWS])
    parser.add_argument("--out", type=Path, help="new output folder; default data/timings, or data/timings/validation for smoke")
    parser.add_argument("--repeats", type=int, help="override timed repeats; the record retains this choice")
    parser.add_argument("--warmup", type=int, help="override warmups; the record retains this choice")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if ((args.repeats is not None and args.repeats < 1)
            or (args.warmup is not None and args.warmup < 0)):
        parser.error("repeats must be positive and warmup nonnegative")
    smoke = args.selection == "smoke" or args.smoke_worker
    if args.worker and args.selection in ("all", "smoke"):
        parser.error("a worker runs exactly one named row")
    out = args.out or ROOT / "data/timings" / ("validation" if smoke else "")
    out = out.resolve()
    rows = selected_rows(args.selection)
    if not smoke:
        try:
            timing.require_idle()
        except RuntimeError as error:
            parser.exit(2, str(error) + "\n")
    # Check every destination before launching any work. Existing records are preserved.
    existing = [row for row in rows if (out / f"{row}.json").exists()]
    if existing:
        parser.exit(2, f"Records already exist for {existing}; choose a new --out folder.\n")
    if args.worker:
        target = out / f"{args.selection}.json"
        out.mkdir(parents=True, exist_ok=True)
        with target.open("x") as handle:
            json.dump({"schema": timing.SCHEMA, "row": args.selection, "status": "running",
                       "purpose": "validation" if smoke else "benchmark"}, handle, indent=1)
            handle.write("\n")
        try:
            record = timing.measure_row(args.selection, smoke=smoke, repeats=args.repeats, warmup=args.warmup)
        except Exception as error:
            # Preserve a portable failure record even when native computation fails.
            target.write_text(json.dumps({"schema": timing.SCHEMA, "row": args.selection, "status": "failed",
                                          "error_type": type(error).__name__}, indent=1) + "\n")
            raise
        target.write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({"row": args.selection, "purpose": record["purpose"],
                          "totals": record["totals"], "morse_nodes": record["graph"]["morse_nodes"]}))
        return
    for row in rows:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), row, "--worker", "--out", str(out)]
        if smoke:
            command.append("--smoke-worker")
        for name in ("repeats", "warmup"):
            value = getattr(args, name)
            if value is not None:
                command += ["--" + name, str(value)]
        subprocess.run(command, check=True, cwd=ROOT, env={**os.environ, **timing.THREAD_ENV})


if __name__ == "__main__":
    main()
