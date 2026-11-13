"""File and process helpers shared by the numbered example commands."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

EXAMPLES = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
NUMERICAL_VERSIONS = {"emlp": "1.0.3", "objax": "1.8.0", "jax": "0.4.38",
                      "jaxlib": "0.4.38", "numpy": "2.3.5", "scipy": "1.15.3",
                      "cmgdb": "1.5.3+fork.7"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    for name, expected in manifest.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"Bundled input checksum mismatch: {name}")
    return len(manifest)


def new_output(path):
    path = Path(path).resolve()
    if path.is_relative_to(EXAMPLES) or EXAMPLES.is_relative_to(path):
        raise ValueError("Choose an output directory outside examples/, such as outputs/example1-replay.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_environment(graphics=True):
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Use the Python 3.13 environment described in code/README.md.")
    if os.environ["JAX_PLATFORM_NAME"] != "cpu":
        raise RuntimeError("The reference workflows use CPU. Set JAX_PLATFORM_NAME=cpu.")
    found, problems = {}, []
    for name, expected in NUMERICAL_VERSIONS.items():
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "missing"
        if found[name] != expected:
            problems.append(f"{name}: expected {expected}, found {found[name]}")
    if graphics:
        problems += [f"Missing executable: {name}" for name in ("dot", "pdftoppm", "pdfinfo")
                     if shutil.which(name) is None]
    if problems:
        raise RuntimeError("Check the setup in code/README.md.\n" + "\n".join(problems))
    return found


def run_module(name, *arguments):
    subprocess.run([sys.executable, "-m", name, *map(str, arguments)], cwd=EXAMPLES, check=True)


def manuscript_run(example, destination):
    verify_inputs(EXAMPLES / example / "reference/manuscript")
    out = new_output(destination)
    run_module("common.paper_replay", "--example", example, "--out", out / "computation", "--recompute-explicit")
    run_module("common.paper_figures", "--example", example, "--data", out / "computation", "--out", out / "figures")
    print(f"Completed manuscript replay: {out}", flush=True)
