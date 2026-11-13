"""Joint O(3) learning and BoxMapData graphs at 64 x 64 latent cells."""

import argparse
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.workflow import new_output, read, require_environment, verify_inputs, write


def load_data(path):
    import numpy as np
    data = np.load(path, allow_pickle=False)
    if not {"train_x", "validation_x", "test_x"}.issubset(data.files):
        data.close()
        raise ValueError("Use the joint dataset with train_x, validation_x, and test_x. Generate it with example2/run.py data.")
    for split in ("train", "validation", "test"):
        values = data[f"{split}_x"]
        if values.ndim != 3 or values.shape[1:] != (11, 4) or len(values) < 8:
            raise ValueError("Each split must have shape (count, 11, 4), with at least eight starts.")
        if values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("Use finite float64 physical-state arrays.")
        if np.max(np.abs(values[..., 0])) > 1.5+1e-12 or np.max(np.linalg.norm(values[..., 1:], axis=-1)) > 1.5+1e-12:
            raise ValueError("Physical states must lie in the prescribed phase domain.")
    return data


def geometry(model, folder, replay):
    import numpy as np
    from example2 import training
    path = folder / "geometry.npz"
    if replay:
        with np.load(path) as saved, np.load(folder / "diagnostics.npz") as diagnostics:
            expected = saved["latent_codes"][::2, ::2].reshape(-1, 2)
            actual = diagnostics["section_codes"]
            difference = float(np.max(np.abs(expected-actual)))
            if difference > 1e-10:
                raise ValueError("The cached physical-section coordinates differ from the loaded encoder.")
        write(folder / "geometry-verification.json", {"points_checked": len(actual), "maximum_difference": difference})
    else:
        a, r = np.linspace(-1.5, 1.5, 513), np.linspace(0, 1.5, 513)
        aa, rr = np.meshgrid(a, r, indexing="ij")
        points = np.column_stack((aa.ravel(), rr.ravel()))
        encode = training.evaluator(model, model.encode)
        codes = training.batched(encode, training.section(points)).reshape(513, 513, 2)
        np.savez_compressed(path, physical_a=a, physical_r=r, latent_codes=codes)


def basin_anchor(folder, metrics):
    import numpy as np
    from example2.basins import component_at
    record = read(folder / "native-results.json")
    if len(record["nodes"]) > 64:
        return None, "This grid has more than 64 native components; the reachability-mask cross-check is unavailable."
    radial = [row["latent"][1] for row in metrics["observed_axis_fixed_points"] if row["kind"] == "radial"]
    sinks = {row["node"] for row in record["nodes"]} - {source for source, _ in record["edges"]}
    if len(radial) != 1 or len(sinks) != 2:
        return None, "The fit does not have one observed radial fixed point and two native sinks."
    with np.load(folder / "native-phase.npz") as data:
        boxes, membership = data["boxes"], data["morse_node"]
        try:
            component_at(np.array([0., 0.]), boxes, membership)
            component_at(np.array([0., radial[0]]), boxes, membership)
        except ValueError as error:
            return None, str(error)
        ordered = sorted(sinks, key=lambda node: float(boxes[membership == node, 0].mean()))
        if boxes[membership == ordered[0], 2].max() >= 0 or boxes[membership == ordered[1], 0].min() <= 0:
            return None, "The two native sinks are not separated by the scalar sign."
    return radial[0], None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "train"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--seeds", type=int, nargs="+")
        command.add_argument("--source-sizes", type=int, nargs="+", help="Physical-section grid sizes for BoxMapData")
        command.set_defaults(data=None)
        if name == "train":
            command.add_argument("--config", type=Path, default=HERE / "config.json")
            command.add_argument("--steps", type=int)
            command.add_argument("--data", type=Path)
    data = commands.add_parser("data", help="Generate the joint experiment's independently sampled splits")
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--sizes", type=int, nargs=3, default=[16384, 4096, 4096])
    args = parser.parse_args()
    try:
        args.versions = require_environment(graphics=args.command != "data")
        if args.command == "data":
            from example2.training import dataset
            if min(args.sizes) < 8:
                raise ValueError("Each split needs at least eight starts.")
            out = new_output(args.out)
            with dataset(out / "trajectories.npz", args.sizes):
                pass
            shutil.move(out / "trajectories.json", out / "metadata.json")
            print(out, flush=True)
        else:
            verify_inputs(HERE)
            from example2.data_map import joint
            joint(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
