"""Checkpoint adapters for numerical attracting-region calculations."""

from pathlib import Path
from hashlib import sha256
import os
import platform

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import numpy as np
from scipy.integrate import solve_ivp

from . import attracting_regions as regions
from .workflow import digest, new_output, read, write


def batched(function, values, size=8192):
    import jax.numpy as jnp
    return np.concatenate([np.asarray(function(jnp.asarray(values[start:start+size])))
                           for start in range(0, len(values), size)])


def begin(args, example, default_model):
    model = (args.model or default_model).resolve()
    if args.out.resolve().is_relative_to(model):
        raise ValueError("Choose an output directory outside the checkpoint directory.")
    if args.samples < 2 or args.repeats < 0 or args.native_repeats < 0:
        raise ValueError("Use at least two samples and nonnegative repetition counts.")
    if example != 2 and (args.cells < 2 or args.cells & (args.cells-1) or args.image_samples < 2):
        raise ValueError("Use a power-of-two interval cell count and at least two image samples.")
    filenames = ["config.json", "weights.npz"] + (["normalization.npz"] if example == 3 else [])
    hashes = {name: digest(model/name) for name in filenames}
    config = read(model/"config.json")
    out = new_output(args.out)
    code = Path(__file__).resolve().parents[2]
    files = ["examples/common/attracting_regions.py", "examples/common/attracting_region_models.py",
             "src/latent_equivariance/quotient_certificates.py"]
    files += [f"examples/example{example}/training.py"] if example != 3 else [
        "examples/example3/periodic/model.py", "examples/example3/periodic/controller.py",
        "examples/example3/periodic/data.py", "examples/example3/sphere.py", "examples/common/emlp_blocks.py"]
    if example == 2:
        files += ["examples/example2/data_map.py", "examples/example2/analysis.py"]
    if example == 3:
        files += ["examples/common/systems.py"]
    metadata = dict(example=f"example{example}", checkpoint=model.name, checkpoint_files=hashes,
                    checkpoint_configuration=config, settings={key: value for key, value in vars(args).items()
                    if key not in {"model", "out", "data", "versions"}}, versions=args.versions,
                    python_version=platform.python_version(), platform=platform.platform(),
                    source_files={name: digest(code/name) for name in files},
                    evidence="sampled image hulls and sampled residual maxima; no uniform physical certificate")
    write(out/"settings.json", regions.plain(metadata))
    return model, out, metadata


def specification(metric, boxes, images, predicted, encoded_image):
    return dict(metric=metric, boxes=boxes, images=images, predicted=predicted, encoded_image=encoded_image)


def finish(out, graph, codes, specifications, metadata, timings, faces, residual_arrays):
    raw, raw_arrays, record, arrays, native_timings = graph
    rows, residuals = regions.compare_regions(record, arrays, codes, specifications, boundary_faces=faces)
    timings.update(native_timings)
    residual_arrays = dict(residual_arrays, codes=codes,
                           **{f"residual_{name}": value for name, value in residuals.items()})
    regions.save_result(out, raw, raw_arrays, record, arrays, rows, residual_arrays, metadata, timings)
    return rows


def sphere_samples(count=65537):
    pole_samples = min(2000, max(8, count//4))
    theta = np.unique(np.r_[np.linspace(0., np.pi, count), np.geomspace(1e-9, .01, pole_samples),
                             np.pi-np.geomspace(1e-9, .01, pole_samples)])
    starts = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
    inside = (theta > 0) & (theta < np.pi)
    w = -np.log(np.tan(theta[inside]/2))
    answer = solve_ivp(lambda _, a: np.tanh(a)*(.16-np.tanh(a)**2), (0., 10.), w,
                       method="DOP853", rtol=2e-13, atol=2e-14)
    if not answer.success:
        raise RuntimeError(answer.message)
    radius, z = np.zeros(len(theta)), np.cos(theta)
    radius[inside], z[inside] = 1/np.cosh(answer.y[:, -1]), np.tanh(answer.y[:, -1])
    images = np.column_stack((radius*np.cos(6.), radius*np.sin(6.), z))
    starts[0], starts[-1] = [0., 0., 1.], [0., 0., -1.]
    return theta, starts, images


def example1(args):
    from example1 import training
    root = Path(training.__file__).parent
    model_path, out, metadata = begin(args, 1, root/"reference/joint/seed1")
    model, load_seconds = regions.elapsed(lambda: training.load(model_path))
    encode = training.evaluator(model, model.encode)
    def iterate(u):
        for _ in range(10):
            u = model.advance(u)
        return u
    advance = training.evaluator(model, iterate)
    (theta, starts, physical_images), physical_seconds = regions.elapsed(lambda: sphere_samples(args.samples))
    def residual_pipeline():
        codes = batched(encode, starts)
        return codes, batched(advance, codes), batched(encode, physical_images)
    (codes, predicted, encoded_image), residual_time = regions.measure(residual_pipeline, args.repeats)
    boundaries = np.linspace(-1., 1., args.cells+1)
    boxes = np.column_stack((boundaries[:-1], boundaries[1:]))
    points = boxes[:, :1]+(boxes[:, 1:]-boxes[:, :1])*np.linspace(0., 1., args.image_samples)
    def image_hulls():
        values = batched(advance, points.reshape(-1)).reshape(points.shape)
        return np.column_stack((np.maximum(-1., np.nextafter(values.min(axis=1)-1e-10, -np.inf)),
                                np.minimum(1., np.nextafter(values.max(axis=1)+1e-10, np.inf))))
    images, image_time = regions.measure(image_hulls, args.repeats)
    faces = np.array([[True, True]])
    graph = regions.compute_graph(boxes, images, boundary_faces=faces, map_time=10., repeats=args.native_repeats)
    ordered_boxes, ordered_images = graph[3]["boxes"], graph[3]["images"]
    specs = {"euclidean": specification(regions.qc.Euclidean(), ordered_boxes, ordered_images,
                                        predicted[:, None], encoded_image[:, None]),
             "sphere_height_lift": specification(regions.qc.SphereHeight(), ordered_boxes, ordered_images,
                                                 predicted[:, None], encoded_image[:, None])}
    metadata.update(map_time=10., primary_metric="euclidean", numerical_dtype="float64",
        sampling=dict(total=len(theta), meridian_points=args.samples,
                      pole_samples_per_end=min(2000, max(8, args.samples//4)),
                      angular_pole_range=[1e-9, .01], physical_rtol=2e-13, physical_atol=2e-14),
        metrics=dict(euclidean="distance in the learned invariant interval coordinate",
                     sphere_height_lift="chordal distance between circles of latitude in the sphere-height lift"),
        image_padding=1e-10, boundary_faces=faces.tolist())
    finish(out, graph, codes[:, None], specs, metadata,
           dict(model_load_seconds=load_seconds, physical_sample_generation_seconds=physical_seconds,
                image_construction=image_time, residual_evaluation=residual_time), faces,
           dict(theta=theta, predicted=predicted, encoded_image=encoded_image))


def o3_samples(count=1024):
    a, r = -1.5+3*(np.arange(count)+.5)/count, 1.5*(np.arange(count)+.5)/count
    aa, rr = np.meshgrid(a, r, indexing="ij")
    grid = np.column_stack((aa.reshape(-1), rr.reshape(-1)))
    near = np.geomspace(1e-10, .1, min(193, max(3, count//4)))
    axis_a = np.r_[-near[::-1], 0., near]
    aa, rr = np.meshgrid(axis_a, np.linspace(0., 1.5, min(257, count+1)), indexing="ij")
    edge = min(4097, 4*count+1)
    return np.concatenate((grid, np.column_stack((aa.reshape(-1), rr.reshape(-1))),
                           np.column_stack((np.linspace(-1.5, 1.5, edge), np.zeros(edge))),
                           np.array([[-1., 0.], [1., 0.], [0., 1.], [0., 0.]])))


def example2(args):
    from example2 import training, data_map, analysis
    sizes = args.source_sizes
    if sorted(set(sizes)) != sizes or any(size < 3 or (size-1) & (size-2) for size in sizes):
        raise ValueError("Source sizes must be increasing distinct values of the form 2^k+1.")
    root = Path(training.__file__).parent
    model_path, out, metadata = begin(args, 2, root/"reference/joint/seed1")
    model, load_seconds = regions.elapsed(lambda: training.load(model_path))
    encode, advance = [training.evaluator(model, method) for method in (model.encode, model.advance)]
    section, generation_seconds = regions.elapsed(lambda: o3_samples(args.samples))
    physical = training.section(section)
    physical_image, physical_seconds = regions.elapsed(lambda: training.physical_step(physical))
    def residual_pipeline():
        codes = batched(encode, physical)
        return codes, batched(advance, codes), batched(encode, physical_image)
    (codes, predicted, encoded_image), residual_time = regions.measure(residual_pipeline, args.repeats)
    largest = max(sizes)
    aa, rr = np.meshgrid(np.linspace(-1.5, 1.5, largest), np.linspace(0., 1.5, largest), indexing="ij")
    source = training.section(np.column_stack((aa.reshape(-1), rr.reshape(-1))))
    def source_pipeline():
        x = batched(encode, source).reshape(largest, largest, 2)
        y = batched(advance, x.reshape(-1, 2)).reshape(largest, largest, 2)
        c = batched(encode, training.section(data_map.CONTROLS))
        return x, y, c, batched(advance, c)
    (x, y, c, v), source_time = regions.measure(source_pipeline, args.repeats)
    np.savez_compressed(out/"source-pairs.npz", encoded_sources=x, learned_images=y,
                        encoded_controls=c, control_images=v,
                        physical_a=np.linspace(-1.5, 1.5, largest), physical_r=np.linspace(0., 1.5, largest))
    write(out/"sampling.json", dict(residual_samples=len(section), independent_interior_points=args.samples**2,
          supplemental_checks="axis, radial boundary, and equilibrium controls", source_sizes=sizes))
    faces = np.array([[False, False], [True, False]])
    for size in sizes:
        target = out/f"S{size}"
        target.mkdir()
        stride = (largest-1)//(size-1)
        sources = np.concatenate((x[::stride, ::stride].reshape(-1, 2), c))
        values = np.concatenate((y[::stride, ::stride].reshape(-1, 2), v))
        if not np.all((sources >= training.LOWER) & (sources <= training.UPPER)):
            raise ValueError("Encoded sources leave the fixed latent window. The data-map domain was not padded.")
        function, tree_time = regions.measure(lambda: data_map.box_map(sources, values), args.repeats)
        boxes = analysis.grid(64, training.LOWER, training.UPPER)
        images, query_time = regions.measure(lambda: np.asarray(function.batch(boxes)), args.repeats)
        support_counts, occupancy_seconds = regions.elapsed(lambda: np.array(
            [len(function.map_points(box.tolist())) for box in boxes], dtype=np.int64))
        graph = regions.compute_graph(boxes, images, boundary_faces=faces, map_time=float(np.log(3.)),
            support=support_counts > 0, data_map=function, repeats=args.native_repeats)
        specs = {"euclidean": specification(regions.qc.Euclidean(), graph[3]["boxes"], graph[3]["images"],
                                            predicted, encoded_image)}
        current = dict(metadata, map_time=float(np.log(3.)), primary_metric="euclidean", numerical_dtype="float64",
            grid_shape=[64, 64], physical_domain=[[-1.5, 1.5], [0., 1.5]], source_size=size,
            source_pairs=len(sources), residual_samples=len(section), boundary_faces=faces.tolist(),
            metrics=dict(euclidean="Euclidean distance in the two learned invariant coordinates"),
            data_map=dict(map_empty="outside", domain_padding=False, padding=False),
            encoded_sources_sha256=sha256(sources.tobytes()).hexdigest(),
            learned_images_sha256=sha256(values.tobytes()).hexdigest())
        finish(target, graph, codes, specs, current,
            dict(model_load_seconds=load_seconds, residual_sample_generation_seconds=generation_seconds,
                 physical_flow_seconds=physical_seconds, source_pair_evaluation=source_time,
                 data_index_construction=tree_time, grid_image_queries=query_time,
                 occupancy_query_seconds=occupancy_seconds, residual_evaluation=residual_time), faces,
            dict(physical_section=section, predicted=predicted, encoded_image=encoded_image))


def example3(args):
    from example3.periodic import model as learning, controller as ctl, data
    from common import systems
    import jax
    import jax.numpy as jnp
    root = Path(learning.__file__).parent
    model_path, out, metadata = begin(args, 3, root/f"reference/{args.policy}/seed0/model")
    model, load_seconds = regions.elapsed(lambda: learning.load(model_path))
    if args.data is None:
        (out/"data").mkdir()
        _, held_out_generation_seconds = regions.elapsed(lambda: data.generate_split(
            out/"data", args.policy, "test", args.samples, 2026111323))
        test_path = out/f"data/{args.policy}-test.npz"
    else:
        test_path = args.data/f"{args.policy}-test.npz" if args.data.is_dir() else args.data
        held_out_generation_seconds = 0.
    with np.load(test_path, allow_pickle=False) as saved:
        observations = saved["observations"]
        times = saved["times"]
    if observations.ndim != 3 or observations.shape[2] != 18 or observations.shape[1] < 11:
        raise ValueError("Held-out observations need shape (trajectories, at least 11 times, 18).")
    if times.shape != (observations.shape[1],) or not np.isfinite(observations).all() or not np.isfinite(times).all() or not np.allclose(np.diff(times), .1):
        raise ValueError("Use finite held-out observations at spacing 0.1.")
    starts = observations.shape[1]-10
    def fresh_samples():
        initial = ctl.sample(args.samples, 2026111341, args.policy, eta_radius=0.)
        evolved = ctl.step(initial, duration=1., policy=args.policy)
        return systems.observe("attitude", initial), systems.observe("attitude", evolved)
    (fresh_source, fresh_image), fresh_seconds = regions.elapsed(fresh_samples)
    np.savez_compressed(out/"fresh-observations.npz", source=fresh_source, image=fresh_image)
    physical_source = np.concatenate((observations[:, :starts].reshape(-1, 18), fresh_source))
    physical_image = np.concatenate((observations[:, 10:].reshape(-1, 18), fresh_image))
    encode, _, _, coordinate, inverse, _, latitude = model.functions()
    p = model.prepare(model.params)
    quotient = jax.jit(lambda u: model.quotient(p, u, steps=10))
    def scalar10(z):
        for _ in range(10):
            z = latitude(z)
        return z
    height10 = jax.jit(scalar10)
    x = ((physical_source-model.mean)/model.scale).astype(np.float32)
    y = ((physical_image-model.mean)/model.scale).astype(np.float32)
    def residual_pipeline():
        initial, future = batched(encode, x), batched(encode, y)
        z = batched(height10, initial[:, 2])
        u = batched(coordinate, initial[:, 2])
        return initial, future, z, u, batched(quotient, u), batched(coordinate, future[:, 2])
    (initial, future, predicted_z, codes, predicted_u, image_u), residual_time = regions.measure(residual_pipeline, args.repeats)
    norm_error = max(float(np.max(np.abs(np.linalg.norm(initial, axis=1)-1))),
                     float(np.max(np.abs(np.linalg.norm(future, axis=1)-1))))
    if not np.isfinite(norm_error) or norm_error > 1e-5:
        write(out/"encoder-diagnostics.json", regions.plain(dict(maximum_sampled_norm_error=norm_error,
              available=False, reason="sampled encoder outputs are not on the latent sphere")))
        raise ValueError("The sampled encoder outputs are not on the latent sphere.")
    boundaries = np.linspace(-1., 1., args.cells+1)
    boxes = np.column_stack((boundaries[:-1], boundaries[1:]))
    points = boxes[:, :1]+(boxes[:, 1:]-boxes[:, :1])*np.linspace(0., 1., args.image_samples)
    def image_hulls():
        values = np.asarray(quotient(jnp.asarray(points.reshape(-1), dtype=jnp.float32)), dtype=np.float64).reshape(points.shape)
        return np.column_stack((np.maximum(-1., values.min(axis=1)-2e-6), np.minimum(1., values.max(axis=1)+2e-6)))
    images, image_time = regions.measure(image_hulls, args.repeats)
    faces = np.array([[True, True]])
    graph = regions.compute_graph(boxes, images, boundary_faces=faces, map_time=1.,
                                  aggregation="coning" if args.policy == "coning" else "endpoints",
                                  repeats=args.native_repeats)
    ordered_boxes, ordered_images = graph[3]["boxes"], graph[3]["images"]
    knots, values = np.asarray(model.knots), np.asarray(model.coordinate_values(p))
    specs = {"euclidean": specification(regions.qc.Euclidean(), ordered_boxes, ordered_images,
                                        predicted_u[:, None], image_u[:, None]),
             "sphere_height": specification(regions.qc.SphereHeight(), np.interp(ordered_boxes, values, knots),
                np.interp(ordered_images, values, knots), predicted_z[:, None], future[:, 2:3])}
    metadata.update(map_time=1., primary_metric="sphere_height", policy=args.policy,
        maximum_sampled_encoder_norm_error=norm_error,
        numerical_dtype="float32 learned model, binary64 tolerance geometry", boundary_faces=faces.tolist(),
        image_padding=2e-6, residual_samples=len(codes), held_out_trajectories=len(observations),
        held_out_file=test_path.name, held_out_sha256=digest(test_path),
        held_out_seed=2026111323 if args.data is None else None,
        windows_per_held_out_trajectory=starts, fresh_trajectories=args.samples, fresh_seed=2026111341,
        physical_sample_scope="generated trajectories start on eta=0; caller supplies the scope of an optional dataset",
        metrics=dict(euclidean="distance in the learned scalar coordinate",
                     sphere_height="chordal sphere quotient metric after inverting the learned monotone coordinate"))
    finish(out, graph, codes[:, None], specs, metadata,
        dict(model_load_seconds=load_seconds, held_out_generation_seconds=held_out_generation_seconds,
             fresh_physical_generation_seconds=fresh_seconds, image_construction=image_time,
             residual_evaluation=residual_time), faces,
        dict(predicted=predicted_u, encoded_image=image_u, predicted_sphere_height=predicted_z,
             initial_sphere_codes=initial, next_sphere_codes=future, coordinate_knots=knots, coordinate_values=values))
