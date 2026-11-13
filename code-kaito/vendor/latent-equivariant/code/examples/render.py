"""Render the sphere and O(3) example panels from the supplied results."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def verify_data(directory):
    manifest = directory / "manifest.json"
    if manifest.is_file():
        for name, expected in json.loads(manifest.read_text()).items():
            path = directory / name
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("sphere", "o3", "all"), default="all")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--layout", type=Path, default=HERE / "layout.json")
    parser.add_argument("--out", type=Path, default=HERE / "output")
    args = parser.parse_args()
    for command in ("dot", "pdftoppm", "pdfinfo"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command is unavailable: {command}")
    verify_data(args.data)
    scripts = []
    if args.example in ("sphere", "all"):
        scripts += ["render_sphere_flow.py", "render_sphere_morse.py"]
    if args.example in ("o3", "all"):
        scripts += ["render_o3.py"]
    for script in scripts:
        command = [sys.executable, "-B", str(HERE / script), "--data", str(args.data.resolve()),
                   "--out", str(args.out.resolve())]
        if script != "render_sphere_flow.py":
            command += ["--layout", str(args.layout.resolve())]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
