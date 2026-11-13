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
    parser.add_argument("--example", choices=("example1", "example2", "sphere", "o3", "all"), default="all")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--layout", type=Path, default=HERE / "layout.json")
    parser.add_argument("--out", type=Path, default=HERE / "output")
    args = parser.parse_args()
    for command in ("dot", "pdftoppm", "pdfinfo"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command is unavailable: {command}")
    cases = [("example1", "sphere", ("render_sphere_flow", "render_sphere_morse")),
             ("example2", "o3", ("render_o3",))]
    for example, model, scripts in cases:
        if args.example not in (example, model, "all"):
            continue
        source = HERE.parent / example / "reference/manuscript" if args.data is None else args.data
        if (source / model).is_dir():
            source = source / model
        verify_data(source)
        for script in scripts:
            command = [sys.executable, "-B", "-m", f"common.{script}", "--data", str(source.resolve()),
                       "--out", str(args.out.resolve())]
            if script != "render_sphere_flow":
                command += ["--layout", str(args.layout.resolve())]
            subprocess.run(command, cwd=HERE.parent, check=True)


if __name__ == "__main__":
    main()
