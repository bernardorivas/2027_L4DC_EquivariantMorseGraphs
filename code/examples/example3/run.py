"""Compute the reduced reference graphs or generate full-plant trajectories."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.workflow import new_output, require_environment, run_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="Compute both reference thrust-axis quotient graphs at eta=0")
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--depth", type=int, default=7)
    data = commands.add_parser("data", help="Generate trajectories of the full plant")
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--count", type=int, default=1024)
    data.add_argument("--seed", type=int, default=2)
    data.add_argument("--intervals", type=int, default=32)
    data.add_argument("--policy", choices=("nominal", "coning"), default="nominal")
    args = parser.parse_args()
    try:
        require_environment(graphics=False)
        out = new_output(args.out)
        if args.command == "replay":
            run_module("common.systems", "attitude-baseline", out, "--depth", args.depth)
        else:
            run_module("common.systems", "generate", "attitude", out / "trajectories.npz",
                       "--count", args.count, "--seed", args.seed, "--intervals", args.intervals,
                       "--policy", args.policy)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
