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


def attracting_region_arguments(command, example):
    """Arguments shared by the numerical attracting-region commands."""
    command.add_argument("--model", type=Path, help="Checkpoint directory containing config.json and weights.npz")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--samples", type=int, default={1: 65537, 2: 1024, 3: 4096}[example],
                         help="Meridian points, section points per axis, or trajectories, respectively")
    command.add_argument("--repeats", type=int, default=3, help="Warm image and residual timing repetitions; zero uses the first call")
    command.add_argument("--native-repeats", type=int, default=5)
    if example != 2:
        command.add_argument("--cells", type=int, default=128)
        command.add_argument("--image-samples", type=int, default=65 if example == 1 else 257)
    else:
        command.add_argument("--source-sizes", type=int, nargs="+", default=[1025])
    if example == 3:
        command.add_argument("--policy", choices=("nominal", "coning"), default="coning")
        command.add_argument("--data", type=Path, help="Optional held-out observation file or its data directory")


def run_module(name, *arguments):
    subprocess.run([sys.executable, "-m", name, *map(str, arguments)], cwd=EXAMPLES, check=True)


def manuscript_run(example, destination):
    verify_inputs(EXAMPLES / example / "reference/manuscript")
    out = new_output(destination)
    run_module("common.paper_replay", "--example", example, "--out", out / "computation", "--recompute-explicit")
    run_module("common.paper_figures", "--example", example, "--data", out / "computation", "--out", out / "figures")
    print(f"Completed manuscript replay: {out}", flush=True)


def compare_metrics(expected, actual, location=""):
    """Check nested numerical diagnostics with the shared replay tolerances."""
    import numpy as np
    checked = 0
    for key, value in expected.items():
        label = f"{location}/{key}"
        if isinstance(value, dict):
            checked += compare_metrics(value, actual[key], label)
        elif isinstance(value, list):
            if len(value) != len(actual[key]):
                raise ValueError(f"Diagnostic length mismatch at {label}")
            checked += compare_metrics(dict(enumerate(value)), dict(enumerate(actual[key])), label)
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if not np.isclose(value, actual[key], rtol=1e-6, atol=1e-9):
                raise ValueError(f"Diagnostic mismatch at {label}: {value} versus {actual[key]}")
            checked += 1
        elif value != actual[key]:
            raise ValueError(f"Diagnostic mismatch at {label}")
    return checked
