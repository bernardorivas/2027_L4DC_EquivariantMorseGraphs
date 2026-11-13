"""Reproduce the saved Example 2 computations or generate physical trajectories."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.workflow import manuscript_run, new_output, require_environment, run_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="Recompute and redraw the supplied manuscript result")
    replay.add_argument("--out", type=Path, required=True)
    data = commands.add_parser("data", help="Generate new physical trajectories at spacing log(3)")
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--count", type=int, default=16384)
    data.add_argument("--seed", type=int, default=1)
    data.add_argument("--intervals", type=int, default=10)
    args = parser.parse_args()
    try:
        require_environment(graphics=args.command == "replay")
        if args.command == "replay":
            manuscript_run("example2", args.out)
        else:
            out = new_output(args.out)
            run_module("common.systems", "generate", "o3", out / "trajectories.npz",
                       "--count", args.count, "--seed", args.seed, "--intervals", args.intervals, "--physical-only")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
