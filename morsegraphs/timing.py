"""CMGDB timings for Section 6 and Table tab:timing.

S1 constructs image hulls and their cell covers; S2 computes the Morse graph;
S3 computes native Conley indices. These numerical maps are sampled (sphere)
or floating-point padded (O(3)), without a uniform enclosure certificate.
Use scripts/run_timings.py to run rows in fresh single-threaded processes.
"""

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import CMGDB

THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
LIMIT = 10000
PAD = 1e-12
C = 0.16
RATE = 67.2
TOL = 1e-6
ADAPTIVE_INIT = 12
GEOMETRY = {
    "reduced": {"lower": [-1.5, 0.0], "upper": [1.5, 1.5]},
    "full": {"lower": [-1.5] * 4, "upper": [1.5] * 4},
}
REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA = "morsegraphs/timing/v1"


def peak_rss_bytes():
    """Peak resident set size of this process (ru_maxrss is in bytes on macOS)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def timed(fn, *args, **kwargs):
    """Call fn with the garbage collector off; return (result, wall seconds, cpu seconds)."""
    gc.collect()
    gc.disable()
    try:
        c0, t0 = time.process_time(), time.perf_counter()
        result = fn(*args, **kwargs)
        t1, c1 = time.perf_counter(), time.process_time()
    finally:
        gc.enable()
    return result, t1 - t0, c1 - c0


class TimedMap:
    """Callable for CMGDB that times the wrapped rectangle map (scalar) and batch map, and counts calls."""

    def __init__(self, f=None, batch=None):
        self.f, self.batch_f = f, batch
        self.reset()

    def reset(self):
        self.seconds, self.calls, self.rects = 0.0, 0, 0

    def __call__(self, rect):
        t0 = time.perf_counter()
        out = self.f(rect)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.rects += 1
        return out

    def batch(self, rects):
        t0 = time.perf_counter()
        out = self.batch_f(rects)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.rects += len(rects)
        return out

    def snapshot(self):
        return {"seconds": self.seconds, "calls": self.calls, "rects": self.rects}


def _annotations(mg):
    return {int(v): list(mg.annotations(v)) for v in range(mg.num_vertices())}


def measure_stages(make_model, cb, options=None, cover=True, conley=True):
    """One repeat of S2, S1_cover and S3 for a model built by make_model() with callback cb (a TimedMap).

    make_model must return a fresh model whose map is cb (or cb.batch). options are passed to the native
    calls of a Model. Returns stage times in seconds, callback snapshots and the
    Morse graph and annotations.
    """
    import CMGDB

    opts = dict(options or {})
    out = {}
    cb.reset()
    mg, wall, cpu = timed(CMGDB.ComputeMorseGraphOnly, make_model(), **opts)
    out["S2"] = {"wall_s": wall, "cpu_s": cpu, "callback": cb.snapshot(), "seconds": wall - cb.seconds}
    out["final_cells"] = final_cells(mg)
    out["morse_nodes"] = mg.num_vertices()
    out["morse_edges"] = [list(e) for e in mg.edges()]
    out["morse_set_sizes"] = [len(mg.morse_set(v)) for v in range(mg.num_vertices())]
    del mg  # release the S2 graph and its grid before the next calls, so it does not inflate their peak
    out["peak_rss_after_bytes"] = {"S2": peak_rss_bytes()}
    if cover:
        cb.reset()
        (_, mapg), _, _ = timed(CMGDB.ComputeMorseGraph, make_model(), cache_map_graph=False, **opts)
        cb.reset()
        _, wall, cpu = timed(mapg.build_cache)
        out["S1_cover"] = {"method": "lazy MapGraph.build_cache()", "wall_s": wall, "cpu_s": cpu,
                           "callback": cb.snapshot(), "seconds": wall - cb.seconds,
                           "edges": int(mapg.num_cached_edges())}
        del mapg
        out["peak_rss_after_bytes"]["S1_cover"] = peak_rss_bytes()
    if conley:
        cb.reset()
        try:
            model = make_model()
            def conley_call():
                out["_conley_enter"] = time.perf_counter()
                result = CMGDB.ComputeConleyMorseGraphOnly(model, **opts)
                out["_conley_exit"] = time.perf_counter()
                return result
            mgc, wall, cpu = timed(conley_call)
            out["S3"] = {"wall_s": wall, "cpu_s": cpu, "callback": cb.snapshot(),
                         "seconds": (wall - cb.seconds) - out["S2"]["seconds"]}
            out["annotations"] = _annotations(mgc)
            out["_conley_graph"] = mgc
        except Exception as error:  # Preserve a native failure along with the completed stages.
            out["S3"] = {"available": False, "error": f"{type(error).__name__}: {error}"}
        out["peak_rss_after_bytes"]["S3"] = peak_rss_bytes()
    return out


def h10_float(z):
    """Float time-10 height map, by bisection on the separated equation (float, unvalidated)."""
    z = np.asarray(z, dtype=float)
    w0 = z * z
    out = np.array(w0, copy=True)
    move = (w0 > 0) & (w0 < 1) & (w0 != C)
    w = w0[move]

    def g(v):
        return 21 * np.log(v) + 4 * np.log1p(-v) - 25 * np.log(np.abs(v - C))

    target = g(w) + RATE
    below = w < C  # G increases on (0, c) and decreases on (c, 1); w_10 lies between w_0 and c
    lo = np.where(below, w, C)
    hi = np.where(below, C, w)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        with np.errstate(divide="ignore"):
            value = g(mid)
        go_up = np.where(below, value < target, value > target)
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    out[move] = 0.5 * (lo + hi)
    return np.sign(z) * np.sqrt(out)


def sample_grid(n, points=True):
    """Latitude x longitude grid on S^2 with spacing at most 1/n (half the cell side 2/n) along both.

    Latitudes phi_i = pi (2i - M) / (2M), i = 0..M, M = 2 ceil(pi n / 2) (even, so the equator is sampled);
    longitudes theta_j = 2 pi j / L, L = 4 ceil(2 pi n / 4). The two poles are kept once each.
    Returns the heights sin(phi_i) (the 1D samples) and the 3D points.
    """
    m = 2 * math.ceil(math.pi * n / 2)
    longitudes = 4 * math.ceil(2 * math.pi * n / 4)
    phi = np.pi * (2 * np.arange(m + 1) - m) / (2 * m)
    heights = np.sin(phi)
    heights[0], heights[-1], heights[m // 2] = -1.0, 1.0, 0.0
    sampling = {"latitudes": m + 1, "longitudes": longitudes, "latitude_step": math.pi / m,
                "equator_longitude_step": 2 * math.pi / longitudes}
    if not points:
        return heights, None, sampling
    theta = 2 * np.pi * np.arange(longitudes) / longitudes
    p, t = np.meshgrid(phi[1:-1], theta, indexing="ij")
    ring = np.column_stack([(np.cos(p) * np.cos(t)).ravel(), (np.cos(p) * np.sin(t)).ravel(),
                            np.repeat(heights[1:-1], longitudes)])
    points = np.vstack([[0.0, 0.0, -1.0], ring, [0.0, 0.0, 1.0]])
    return heights, points, sampling


def sphere_map(points):
    """(x, y, z) -> (s' cos th, s' sin th, h_10(z)), s' = sqrt(1 - h^2), th = atan2(y, x); h evaluated per point."""
    h = h10_float(points[:, 2])
    s = np.sqrt(np.maximum(0.0, 1.0 - h * h))
    th = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack([s * np.cos(th), s * np.sin(th), h])


class TableLookup:
    """CMGDB callback on a uniform grid: the image of a cell is a row of the S1 table.

    The row is the integer cell index computed from the rectangle's lower corner; rounding absorbs last-bit
    differences between the tree grid and the chomp cube geometry. A rectangle that is not one grid cell is a
    miss and is evaluated directly with fallback(rect).
    """

    def __init__(self, table, lower, side, n, fallback):
        self.table, self.lower, self.side, self.n, self.fallback = table, np.asarray(lower, float), side, n, fallback
        self.dim = len(lower)
        self.mult = n ** np.arange(self.dim, dtype=np.int64)
        self.misses, self.miss_seconds, self.miss_examples = 0, 0.0, []

    def _miss(self, rect):
        """Direct evaluation of a rectangle that is not a table row; its time is enclosure work (S1_enc_misses)."""
        t0 = time.perf_counter()
        self.misses += 1
        if len(self.miss_examples) < 20:
            self.miss_examples.append([float(v) for v in rect])
        out = [float(v) for v in self.fallback([float(v) for v in rect])]
        self.miss_seconds += time.perf_counter() - t0
        return out

    def scalar(self, rect):
        d, s = self.dim, self.side
        index, ok = 0, True
        for k in range(d):
            a = (rect[k] - self.lower[k]) / s
            b = (rect[d + k] - self.lower[k]) / s
            ia, ib = round(a), round(b)
            if abs(a - ia) > TOL or abs(b - ib) > TOL or ib != ia + 1 or ia < 0 or ib > self.n:
                ok = False
                break
            index += ia * int(self.mult[k])
        if not ok:
            return self._miss(rect)
        return self.table[index].tolist()

    def batch(self, rects):
        r = np.asarray(rects, dtype=float)
        d = self.dim
        a = (r[:, :d] - self.lower) / self.side
        b = (r[:, d:] - self.lower) / self.side
        ia, ib = np.rint(a), np.rint(b)
        ok = ((np.abs(a - ia) <= TOL) & (np.abs(b - ib) <= TOL) & (ib == ia + 1) & (ia >= 0) & (ib <= self.n)).all(1)
        index = (np.where(ok[:, None], ia, 0).astype(np.int64) * self.mult).sum(1)
        out = self.table[index]
        for m in np.nonzero(~ok)[0]:
            out[m] = self._miss(r[m])
        return out


class StageMap(TimedMap):
    """TimedMap that also reports the lookup misses of each stage in its snapshot."""

    def __init__(self, f, batch, lookup=None):
        self.lookup = lookup
        super().__init__(f, batch)

    def reset(self):
        super().reset()
        if self.lookup is not None:
            self.lookup.misses, self.lookup.miss_seconds = 0, 0.0

    def snapshot(self):
        snap = super().snapshot()
        if self.lookup is not None:
            snap["misses"] = self.lookup.misses
            snap["miss_seconds"] = self.lookup.miss_seconds
        return snap


def build_data(level, dim, table=True):
    """Samples, sample images, BoxMapData; and, if table, BoxMapData.compute on every top cell."""
    n = 2**level
    (heights, points, sampling), s0, _ = timed(sample_grid, n, dim == 3)
    x = heights[:, None] if dim == 1 else points
    y, t_img, c_img = timed(lambda: h10_float(heights)[:, None] if dim == 1 else sphere_map(points))
    lower, upper = [-1.0] * dim, [1.0] * dim
    bmd, t_idx, c_idx = timed(CMGDB.BoxMapData, x, y, map_empty="outside", lower_bounds=lower,
                                      upper_bounds=upper)
    parts = {"S0_samples": {"seconds": s0}}
    counts = {"sample_points": int(len(x)), "sample_heights": int(len(heights)), "sampling": sampling,
              "boxmapdata_index": bool(bmd._use_index),
              "boxmapdata_bins_per_dim": int(getattr(bmd, "_bins_per_dim", 0)) or None}
    enc = {"images": t_img, "boxmapdata_index": t_idx}
    cpu = c_img + c_idx
    out = None
    if table:
        def compute_table():
            coords = (-1.0 + (2.0 / n) * np.arange(n + 1)).tolist()
            rows = np.empty((n**dim, 2 * dim))
            compute = bmd.compute
            if dim == 1:
                for i in range(n):
                    rows[i] = compute([coords[i], coords[i + 1]])
            else:
                c = 0
                for kz in range(n):
                    z0, z1 = coords[kz], coords[kz + 1]
                    for jy in range(n):
                        y0, y1 = coords[jy], coords[jy + 1]
                        for ix in range(n):
                            rows[c] = compute([coords[ix], y0, z0, coords[ix + 1], y1, z1])
                            c += 1
            return rows
        out, t_tab, c_tab = timed(compute_table)
        enc["table"] = t_tab
        cpu += c_tab
        empty = out[:, 0] > 1.5  # BoxMapData's 'outside' box is [ub + 1, ub + 2]
        counts.update(unique_enclosures=int(len(out)), cells_with_data=int((~empty).sum()),
                      cells_without_data=int(empty.sum()))
    parts["S1_enc"] = {"seconds": sum(enc.values()), "cpu_s": cpu, "parts": enc}
    return out, parts, counts, {"bmd": bmd, "x": x, "y": y}


def grid_record(side, mode, level):
    n = 2**level
    dim = 1 if side == "reduced" else 3
    depth = level * dim
    init = depth if mode == "uniform" else ADAPTIVE_INIT
    lower, upper = [-1.0] * dim, [1.0] * dim
    return {"model": "CMGDB.Model", "constructor": [depth, depth, init, LIMIT, lower, upper],
            "lower_bounds": lower, "upper_bounds": upper, "periodic": [False] * dim,
            "depths": {"subdiv_min": depth, "subdiv_max": depth, "subdiv_init": init, "subdiv_limit": LIMIT},
            "shape": [n] * dim, "cell_side": [2.0 / n] * dim,
            "cell_side_coordinates": "height z" if dim == 1 else "ambient (x, y, z) of [-1,1]^3",
            "top_cells": n**dim, "initial_cells": 2**init}


def make_model_factory(grid, cb, batch):
    depth_min, depth_max, init, limit, lower, upper = grid["constructor"]

    def make_model():
        model = CMGDB.Model(depth_min, depth_max, init, limit, lower, upper, cb)
        if batch:
            model.set_batch_map(cb.batch)
        return model
    return make_model


def level_geometry(side, n):
    k = int(round(math.log2(n)))
    assert 2 ** k == n and k >= 1, "level n must be a power of two"
    g = GEOMETRY[side]
    d = len(g["lower"])
    depth = 2 * k if side == "reduced" else 4 * k + 3
    shape = tuple(depth // d + (1 if j < depth % d else 0) for j in range(d))
    shape = tuple(2 ** b for b in shape)
    lower = np.array(g["lower"], dtype=float)
    upper = np.array(g["upper"], dtype=float)
    cell = (upper - lower) / np.array(shape, dtype=float)
    return {"dim": d, "depth": depth, "shape": shape, "lower": lower, "upper": upper, "cell": cell,
            "cells": int(np.prod(shape, dtype=np.int64)), "k": k}


def cell_rects(geom, start, stop):
    """Rectangles [lo..., hi...] of the top cells start..stop-1 in C order of geom['shape']."""
    idx = np.stack(np.unravel_index(np.arange(start, stop, dtype=np.int64), geom["shape"]), axis=1)
    lo = geom["lower"] + idx * geom["cell"]
    hi = geom["lower"] + (idx + 1) * geom["cell"]
    return np.concatenate([lo, hi], axis=1)


def g_scalar(a):
    return 3.0 * a / np.sqrt(1.0 + 8.0 * a * a)


def _f(x, s, D):
    return 3.0 * x / np.sqrt(D * (D + 8.0 * x * x + 8.0 * s))


def _sq_range(lo, hi):
    sq_lo, sq_hi = lo * lo, hi * hi
    sq_max = np.maximum(sq_lo, sq_hi)
    sq_min = np.where((lo <= 0.0) & (hi >= 0.0), 0.0, np.minimum(sq_lo, sq_hi))
    return sq_min, sq_max


def _pad(lo, hi):
    return lo - PAD * np.abs(lo), hi + PAD * np.abs(hi)


def full_hull(rects):
    """Monotone coordinate hull of F(a, v) = (3a/sqrt(D), 3v/sqrt(D(D + 8|v|^2))), D = 1 + 8a^2, on boxes in the
    axis order (v1, v2, v3, a). The v_i-component is f(v_i, s, D) = 3 v_i / sqrt(D (D + 8 v_i^2 + 8 s)), increasing in
    v_i, and in s = sum of the other squares and in D decreasing for v_i > 0, increasing for v_i < 0."""
    rects = np.asarray(rects, dtype=float)
    lo, hi = rects[:, :4], rects[:, 4:]
    a_lo, a_hi = lo[:, 3], hi[:, 3]
    asq_min, asq_max = _sq_range(a_lo, a_hi)
    D_min, D_max = 1.0 + 8.0 * asq_min, 1.0 + 8.0 * asq_max
    sq_min, sq_max = _sq_range(lo[:, :3], hi[:, :3])
    out = np.empty_like(rects)
    for i in range(3):
        j, k = [m for m in range(3) if m != i]
        s_min = sq_min[:, j] + sq_min[:, k]
        s_max = sq_max[:, j] + sq_max[:, k]
        x_lo, x_hi = lo[:, i], hi[:, i]
        up = np.where(x_hi > 0.0, _f(x_hi, s_min, D_min), _f(x_hi, s_max, D_max))
        dn = np.where(x_lo > 0.0, _f(x_lo, s_max, D_max), _f(x_lo, s_min, D_min))
        out[:, i], out[:, 4 + i] = _pad(dn, up)
    out[:, 3], out[:, 7] = _pad(g_scalar(a_lo), g_scalar(a_hi))
    return out


def quotient_hull(rects):
    """The same float hull class on the quotient (a, r): r-component f(r, 0, D) with r >= 0."""
    rects = np.asarray(rects, dtype=float)
    a_lo, r_lo, a_hi, r_hi = rects.T
    asq_min, asq_max = _sq_range(a_lo, a_hi)
    D_min, D_max = 1.0 + 8.0 * asq_min, 1.0 + 8.0 * asq_max
    out = np.empty_like(rects)
    out[:, 0], out[:, 2] = _pad(g_scalar(a_lo), g_scalar(a_hi))
    out[:, 1], out[:, 3] = _pad(_f(r_lo, 0.0, D_max), _f(r_hi, 0.0, D_min))
    return out


def full_point(p):
    v, a = p[:, :3], p[:, 3]
    D = 1.0 + 8.0 * a * a
    s = (v * v).sum(axis=1)
    return np.concatenate([3.0 * v / np.sqrt(D * (D + 8.0 * s))[:, None], (3.0 * a / np.sqrt(D))[:, None]], axis=1)


def quotient_point(p):
    a, r = p[:, 0], p[:, 1]
    D = 1.0 + 8.0 * a * a
    return np.stack([3.0 * a / np.sqrt(D), 3.0 * r / np.sqrt(D * (D + 8.0 * r * r))], axis=1)


def build_table(enc, geom, chunk=1 << 20):
    cells, d = geom["cells"], geom["dim"]
    table = np.empty((cells, 2 * d), dtype=float)
    if enc.native is not None:
        box_image = enc.native.box_image
        for start in range(0, cells, chunk):
            stop = min(cells, start + chunk)
            rects = cell_rects(geom, start, stop).tolist()
            table[start:stop] = [box_image(tuple(r)) for r in rects]
    else:
        for start in range(0, cells, chunk):
            stop = min(cells, start + chunk)
            table[start:stop] = enc.batch(cell_rects(geom, start, stop))
    return table


class StampedMap(TimedMap):
    """Base of the two CMGDB callbacks. It keeps TimedMap's contract (self.seconds is all time spent in Python
    callbacks, which measure_stages subtracts from the native stage times) and splits that time into
      eval_seconds      the table lookups (LookupMap) or the enclosure evaluations (DirectMap);
      miss_seconds      the direct evaluations of table misses, a part of eval_seconds (LookupMap only);
      overhead_seconds  bookkeeping: the per-call stamps and the depth census.
    Every call is stamped with its entry and exit times, its rectangle count, the peak RSS after it and its first two
    rectangles; s3_per_set uses the stamps of the Conley call to split S3 per Morse set. snapshot(), which
    measure_stages calls right after each stage, also records the peak RSS at the end of that stage."""

    def __init__(self, geom):
        self.d = geom["dim"]
        self.width = geom["upper"] - geom["lower"]
        self.census = None
        super().__init__(f=self._scalar, batch=self._batch)

    def reset(self):
        super().reset()
        self.eval_seconds = self.miss_seconds = self.overhead_seconds = 0.0
        self.scalar_calls = self.batch_calls = 0
        self.misses = self.inexact = 0
        self.t_in, self.t_out, self.n, self.rss, self.head0, self.head1 = [], [], [], [], [], []
        self.ev, self.ms = [], []  # per call: evaluation seconds, of which direct evaluations of misses
        if self.census is not None:
            self.census = {}  # per stage: measure_stages resets before each stage and snapshots after it

    def snapshot(self):
        out = super().snapshot()
        out.update(eval_seconds=self.eval_seconds, miss_seconds=self.miss_seconds,
                   overhead_seconds=self.overhead_seconds, scalar_calls=self.scalar_calls,
                   batch_calls=self.batch_calls, lookup_misses=self.misses, inexact_hits=self.inexact,
                   peak_rss_bytes_at_end=peak_rss_bytes())
        if self.census is not None:
            out["census_rects_per_depth"] = {str(k): v for k, v in sorted(self.census.items())}
        return out

    def _count(self, rects):
        """Rectangles per subdivision depth (sum over axes of log2(domain width / box width))."""
        w = rects[:, self.d:] - rects[:, :self.d]
        if np.all(w == w[0]):
            depth = int(np.rint(np.log2(self.width / w[0])).sum())
            self.census[depth] = self.census.get(depth, 0) + len(rects)
            return
        depth = np.rint(np.log2(self.width / w)).sum(axis=1).astype(np.int64)
        for dd, c in zip(*np.unique(depth, return_counts=True)):
            self.census[int(dd)] = self.census.get(int(dd), 0) + int(c)

    def _account(self, t0, t1, rects, miss0):
        self.t_in.append(t0)
        self.ev.append(t1 - t0)
        self.ms.append(self.miss_seconds - miss0)
        self.n.append(len(rects))
        self.rss.append(peak_rss_bytes())
        self.head0.append(rects[0].copy())
        self.head1.append(rects[1].copy() if len(rects) > 1 else None)
        if self.census is not None:
            self._count(rects)
        t2 = time.perf_counter()
        self.t_out.append(t2)
        self.seconds += t2 - t0
        self.eval_seconds += t1 - t0
        self.overhead_seconds += t2 - t1
        self.calls += 1
        self.rects += len(rects)

    def __call__(self, rect):
        miss0 = self.miss_seconds
        t0 = time.perf_counter()
        out = self._scalar(rect)
        t1 = time.perf_counter()
        self.scalar_calls += 1
        self._account(t0, t1, np.asarray([rect], dtype=float), miss0)
        return out

    def batch(self, rects):
        miss0 = self.miss_seconds
        t0 = time.perf_counter()
        out = self._batch(rects)
        t1 = time.perf_counter()
        self.batch_calls += 1
        self._account(t0, t1, rects, miss0)
        return out


class LookupMap(StampedMap):
    """CMGDB callback on a uniform grid: lookup into the S1 table by the integer cell index of the lower corner.
    A rectangle whose rounded index is out of range, or whose corners differ from the table cell by more than
    1e-9 cell sides, is a miss and is evaluated directly (timed as miss_seconds). Exact corner equality is counted
    separately (inexact_hits)."""

    def __init__(self, geom, table, enc):
        self.geom, self.table, self.enc = geom, table, enc
        self.lower, self.cell = geom["lower"], geom["cell"]
        self.shape = np.array(geom["shape"], dtype=np.int64)
        self.tol = 1e-9 * self.cell
        self.lower_l, self.cell_l = self.lower.tolist(), self.cell.tolist()
        self.strides = [int(np.prod(geom["shape"][j + 1:], dtype=np.int64)) for j in range(geom["dim"])]
        super().__init__(geom)

    def _direct(self, rects):
        t0 = time.perf_counter()
        out = self.enc.batch(rects)
        self.miss_seconds += time.perf_counter() - t0
        return out

    def _scalar(self, rect):
        d, flat, exact = self.d, 0, True
        for j in range(d):
            i = round((rect[j] - self.lower_l[j]) / self.cell_l[j])
            lo = self.lower_l[j] + i * self.cell_l[j]
            hi = self.lower_l[j] + (i + 1) * self.cell_l[j]
            dev = max(abs(lo - rect[j]), abs(hi - rect[d + j]))
            if not (0 <= i < self.geom["shape"][j]) or dev > self.tol[j]:
                self.misses += 1
                return self._direct(np.asarray([rect], dtype=float))[0].tolist()
            exact = exact and dev == 0.0
            flat += i * self.strides[j]
        self.inexact += 0 if exact else 1
        return self.table[flat].tolist()

    def _batch(self, rects):
        lo, hi = rects[:, :self.d], rects[:, self.d:]
        idx = np.rint((lo - self.lower) / self.cell).astype(np.int64)
        dev = np.maximum(np.abs(self.lower + idx * self.cell - lo), np.abs(self.lower + (idx + 1) * self.cell - hi))
        ok = np.all((idx >= 0) & (idx < self.shape) & (dev <= self.tol), axis=1)
        out = np.empty(rects.shape, dtype=float)
        if ok.all():
            out[:] = self.table[np.ravel_multi_index(idx.T, self.geom["shape"])]
        else:
            out[ok] = self.table[np.ravel_multi_index(idx[ok].T, self.geom["shape"])]
            miss = ~ok
            self.misses += int(miss.sum())
            out[miss] = self._direct(rects[miss])
        self.inexact += int((ok & np.any(dev > 0.0, axis=1)).sum())
        return out


class DirectMap(StampedMap):
    """CMGDB callback that evaluates the enclosure itself (adaptive grids and lazy controls). With census on, it
    counts the rectangles per subdivision depth; that bookkeeping is timed as overhead, inside the callback time,
    so it enters neither the native stage times nor S1_enc."""

    def __init__(self, enc, geom):
        self.enc = enc
        super().__init__(geom)

    def reset(self):
        super().reset()
        self.enc.reset()  # cold enclosure caches at the start of every CMGDB call (measure_stages resets per stage)

    def _scalar(self, rect):
        return self.enc.batch(np.asarray([rect], dtype=float))[0].tolist()

    def _batch(self, rects):
        return self.enc.batch(rects)


def final_cells(mg):
    """Number of cells of the final phase-space grid of a Morse graph. The Python API has no size accessor; on a
    non-atlas grid MorseGraph.phase_space_chart_box raises IndexError exactly when the index is out of range and
    RuntimeError otherwise, so a binary search on IndexError gives the size (CMGDB used as it is)."""
    def exists(i):
        try:
            mg.phase_space_chart_box(i)
        except IndexError:
            return False
        except Exception:  # noqa: BLE001 (RuntimeError: in range, not an atlas)
            return True
        return True
    if not exists(0):
        return 0
    lo, hi = 0, 1
    while exists(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if exists(mid) else (lo, mid)
    return hi


def s3_per_set(cb, mgc, graph_rects, t_enter, t_exit, chunk):
    """Split the ComputeConleyMorseGraphOnly call per Morse set from the callback stamps.

    In that call CMGDB first repeats the Morse graph pass (graph_rects rectangles, as in S2), then runs
    chomp::ConleyIndex for v = 0, 1, ... in order. Each set starts with one batched evaluation of the cells of S_v in
    the order of morse_set(v), in chunks of batch_chunk_size, whose first rectangles are phase_space_box of the first
    cells of S_v (CMGDB/_cmgdb/CMGDB.cpp ComputeMorseGraphCore, chomp/ConleyIndex.h). Set v is the time from the
    first call of its S batch to the first call of the next set's (to the end of the call for the last set), minus
    callback time: its homology, which runs after its last map evaluation, is attributed to it; the preparation of
    S_{v+1} (subset and geometries, linear in |S_{v+1}|) is attributed to set v too. The graph phase runs from the
    entry of the call to the first S batch, minus callback time (it includes the join of the grid)."""
    t_in, t_out, n = np.array(cb.t_in), np.array(cb.t_out), np.array(cb.n, dtype=np.int64)
    if not len(n):
        return None
    cb_time = t_out - t_in
    cum = np.cumsum(n)
    b = int(np.searchsorted(cum, graph_rects))
    out = {"graph_rects": int(graph_rects), "graph_pass_aligned": bool(b < len(cum) and cum[b] == graph_rects)}
    starts, j = [], b + 1
    sizes = []
    for v in range(mgc.num_vertices()):
        cells = mgc.morse_set(v)
        size = len(cells)
        sizes.append(size)
        h0 = np.asarray(mgc.phase_space_box(cells[0]), dtype=float)
        h1 = np.asarray(mgc.phase_space_box(cells[1]), dtype=float) if size > 1 else None
        expect = size if chunk == 0 else min(chunk, size)
        found = None
        for k in np.flatnonzero(n[j:] == expect) + j:
            if np.array_equal(cb.head0[k], h0) and (h1 is None or np.array_equal(cb.head1[k], h1)):
                found = int(k)
                break
        if found is None:
            out.update(matched=False, unmatched_node=v, matched_nodes=len(starts))
            return out
        starts.append(found)
        j = found + 1
    bounds = starts + [len(n)]
    t_bounds = [t_in[k] for k in starts] + [t_exit]
    sets = []
    for v in range(len(starts)):
        k0, k1 = bounds[v], bounds[v + 1]
        span = t_bounds[v + 1] - t_bounds[v]
        cbs = float(cb_time[k0:k1].sum())
        sets.append({"node": v, "cells": sizes[v], "seconds": span - cbs, "callback_seconds": cbs,
                     "calls": int(k1 - k0), "rects": int(n[k0:k1].sum()),
                     "peak_rss_bytes_at_next_set": int(cb.rss[k1]) if k1 < len(n) else None})
    graph = (t_in[starts[0]] - t_enter) - float(cb_time[:starts[0]].sum())
    out.update(matched=True, graph_phase_seconds=graph, sets=sets,
               sets_total_seconds=float(sum(s["seconds"] for s in sets)),
               conley_phase_calls=int(len(n) - starts[0]), conley_phase_rects=int(cum[-1] - cum[starts[0] - 1]),
               peak_rss_bytes_end_of_graph_pass=int(cb.rss[b]) if b < len(n) else None)
    return out


class FloatEnclosure:
    """Monotone coordinate hulls evaluated in binary64 with relative padding."""

    native = None

    def __init__(self, side):
        self.side = side

    def reset(self):
        pass

    def batch(self, rects):
        return full_hull(rects) if self.side == "full" else quotient_hull(rects)


ROWS = {
    "sphere-uniform": dict(example="sphere", side="full", mode="uniform", level=7,
                           repeats=5, warmup=1, inner=1, cells=2**21,
                           paper=[31.0, 1.1, 14.0, 47.0]),
    "sphere-adaptive": dict(example="sphere", side="full", mode="adaptive", level=7,
                            repeats=5, warmup=1, inner=1, cells=17464,
                            paper=[2.9, 0.060, 21.0, 24.0]),
    "sphere-quotient": dict(example="sphere", side="reduced", mode="uniform", level=7,
                            repeats=5, warmup=1, inner=1, cells=2**7,
                            paper=[0.0023, "<1 ms", "<1 ms", 0.0031]),
    "o3-uniform": dict(example="o3", side="full", mode="uniform", level=32,
                       repeats=1, warmup=0, inner=1, cells=2**23,
                       paper=[16.0, 4.9, 991.0, 1011.0]),
    "o3-adaptive": dict(example="o3", side="full", mode="adaptive", level=32,
                        repeats=1, warmup=0, inner=1, cells=109536,
                        paper=[2.3, 0.580, 943.0, 946.0]),
    "o3-quotient": dict(example="o3", side="reduced", mode="uniform", level=32,
                        repeats=5, warmup=1, inner=116, cells=2**10,
                        paper=["<1 ms", "<1 ms", "<1 ms", 0.0018]),
}


def machine_state():
    """Load averages, free memory, and memory available for a new computation."""
    if sys.platform == "darwin":
        import re
        raw = subprocess.check_output(["vm_stat"], text=True)
        page_size = int(re.search(r"page size of (\d+) bytes", raw).group(1))
        free_pages = int(re.search(r"Pages free:\s*(\d+)", raw).group(1))
        inactive_pages = int(re.search(r"Pages inactive:\s*(\d+)", raw).group(1))
        free = page_size * free_pages
        available = page_size * (free_pages + inactive_pages)
    elif sys.platform.startswith("linux"):
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        free = int(fields["MemFree"].split()[0]) * 1024
        available = int(fields["MemAvailable"].split()[0]) * 1024
    else:
        raise RuntimeError("Free-memory measurement is supported on macOS and Linux")
    return {"load_average": list(os.getloadavg()), "free_memory_bytes": free,
            "available_memory_bytes": available,
            "memory_definition": "available memory: free plus inactive pages on macOS; MemAvailable on Linux"}


def require_idle(state=None):
    """Require a one-minute load below 2 and at least 16 GB available memory."""
    state = machine_state() if state is None else state
    loads = state["load_average"]
    recent = loads[0] if loads else float("nan")
    available = state["available_memory_bytes"]
    if (len(loads) != 3 or not all(math.isfinite(x) and x >= 0 for x in loads)
            or recent >= 2 or not available >= 16_000_000_000):
        raise RuntimeError(f"Timing run refused: one-minute load {recent:.2f}; "
                           f"{available / 1e9:.2f} GB available. "
                           "The one-minute load must be below 2 and at least 16 GB must be available.")
    return state


def machine_description():
    """Machine and package versions without account names or installation paths."""
    def query(*args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    import CMGDB._cmgdb as native
    cpu = query("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor()
    total = query("sysctl", "-n", "hw.memsize") if sys.platform == "darwin" else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {"system": platform.system(), "release": platform.release(), "architecture": platform.machine(),
            "cpu": cpu, "logical_cpus": os.cpu_count(), "memory_bytes": int(total) if total else None,
            "python": platform.python_version(), "numpy": np.__version__,
            "cmgdb": importlib.metadata.version("CMGDB"),
            "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "boxmapdata_sha256": hashlib.sha256((Path(CMGDB.__file__).parent / "BoxMapData.py").read_bytes()).hexdigest(),
            "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV}}


def _release_graph(out):
    graph = out.pop("_conley_graph", None)
    out.pop("_conley_enter", None)
    out.pop("_conley_exit", None)
    return graph


def sphere_repeat(config):
    side, mode, level = (config[k] for k in ("side", "mode", "level"))
    grid = grid_record(side, mode, level)
    dim = len(grid["shape"])
    table, parts, counts, extra = build_data(level, dim, table=mode == "uniform")
    lookup = TableLookup(table, grid["lower_bounds"], grid["cell_side"][0], 2**level,
                         extra["bmd"].compute) if mode == "uniform" else None
    cb = StageMap(lookup.scalar, lookup.batch, lookup) if lookup is not None else StageMap(extra["bmd"].compute, extra["bmd"].batch)
    make_model = make_model_factory(grid, cb, batch=True)
    out = measure_stages(make_model, cb, cover=mode == "uniform")
    _release_graph(out)
    stages = {**parts, **{k: out[k] for k in ("S2", "S1_cover", "S3") if k in out}}
    if lookup is not None:
        stages["S1_enc_misses"] = {"seconds": sum(out.get(k, {}).get("callback", {}).get("miss_seconds", 0.0) for k in ("S2", "S3"))}
        stages["S2_graph"] = {"seconds": out["S2"]["seconds"] - out["S1_cover"]["seconds"]}
    else:
        stages["S1_enc"]["seconds"] += out["S2"]["callback"]["seconds"]
        if "callback" in out["S3"]:
            stages["S1_enc_conley_phase"] = {"seconds": out["S3"]["callback"]["seconds"] - out["S2"]["callback"]["seconds"]}
    return {"stages": stages, "counts": counts, "graph": {k: out.get(k) for k in (
        "morse_nodes", "morse_edges", "morse_set_sizes", "annotations", "final_cells")},
        "peak_rss_bytes": peak_rss_bytes(), "load_average": list(os.getloadavg())}


def o3_repeat(config):
    side, mode, level = (config[k] for k in ("side", "mode", "level"))
    geom = level_geometry(side, level)
    enc = FloatEnclosure(side)
    stages = {}
    if mode == "uniform":
        table, wall, cpu = timed(build_table, enc, geom)
        stages["S1_enc"] = {"seconds": wall, "cpu_s": cpu}
        cb = LookupMap(geom, table, enc)
    else:
        cb = DirectMap(enc, geom)
        cb.census = {}
    init = geom["depth"] if mode == "uniform" else 15
    def make_model():
        model = CMGDB.Model(geom["depth"], geom["depth"], init, LIMIT,
                            geom["lower"].tolist(), geom["upper"].tolist(), [False] * geom["dim"], cb)
        model.set_batch_map(cb.batch)
        return model
    opts = {"cache_transition_graph": True, "batch_chunk_size": 512,
            "max_cached_edges": 0, "reserve_edges": 0, "reserve_min_edges": 0}
    out = measure_stages(make_model, cb, options=opts, cover=mode == "uniform")
    stages.update({k: out[k] for k in ("S2", "S1_cover", "S3") if k in out})
    s2, s3 = out["S2"]["callback"], out["S3"].get("callback")
    if s3 is not None:
        boundary = int(np.searchsorted(np.cumsum(cb.n), s2["rects"]))
        if boundary >= len(cb.n) or sum(cb.n[:boundary + 1]) != s2["rects"]:
            raise RuntimeError("Conley-call graph pass does not align with the Morse pass")
        conley_eval = float(np.sum(cb.ev[boundary + 1:]))
        conley_miss = float(np.sum(cb.ms[boundary + 1:]))
        split = s3_per_set(cb, out["_conley_graph"], s2["rects"], out["_conley_enter"], out["_conley_exit"], 512)
        if split and split.get("matched"):
            stages["S3_direct"] = {"seconds": split["sets_total_seconds"]}
        out["S3_sets"] = split
    else:
        conley_eval = conley_miss = None
    if mode == "uniform":
        stages["S1_enc_miss"] = {"seconds": s2["miss_seconds"]}
        stages["S1_enc_conley"] = {"seconds": conley_miss}
        stages["S2_graph"] = {"seconds": out["S2"]["seconds"] - out["S1_cover"]["seconds"]}
    else:
        stages["S1_enc"] = {"seconds": s2["eval_seconds"]}
        stages["S1_enc_conley"] = {"seconds": conley_eval}
    _release_graph(out)
    return {"stages": stages, "graph": {k: out.get(k) for k in (
        "morse_nodes", "morse_edges", "morse_set_sizes", "annotations", "final_cells", "S3_sets")},
        "peak_rss_bytes": peak_rss_bytes(), "load_average": list(os.getloadavg())}


def summarize_repeats(raw, config):
    samples = {}
    for repeat in raw:
        for key, values in repeat["stage_samples"].items():
            samples.setdefault(key, []).append(statistics.mean(values))
    summary = {k: {"median_s": statistics.median(v), "samples_s": v,
                   "spread_over_median": (max(v) - min(v)) / abs(statistics.median(v)) if statistics.median(v) else None}
               for k, v in samples.items()}
    med = {k: v["median_s"] for k, v in summary.items()}
    s1 = sum(med.get(k, 0.0) for k in ("S1_enc", "S1_cover", "S1_enc_misses", "S1_enc_miss",
                                      "S1_enc_conley_phase", "S1_enc_conley"))
    s2 = med.get("S2_graph", med["S2"])
    s3key = "S3_direct" if config["example"] == "o3" and config["side"] == "reduced" else "S3"
    s3 = med.get(s3key)
    # The published table uses a one-sided 1 ms display floor for differenced S3.
    total = None if s3 is None else s1 + s2 + max(0.0, s3)
    return summary, {"map_s": s1, "morse_graph_s": s2, "conley_index_s": s3, "total_s": total,
                     "conley_method": s3key, "morse_contains_cover": config["mode"] == "adaptive"}


def measure_row(row, *, smoke=False, repeats=None, warmup=None):
    """Measure one row. Smoke mode permits only quotient rows and two small uniform grids."""
    wrong = {k: os.environ.get(k) for k, v in THREAD_ENV.items() if os.environ.get(k) != v}
    if wrong:
        raise RuntimeError(f"Set thread variables before Python starts: {THREAD_ENV}; got {wrong}")
    config = dict(ROWS[row])
    if smoke:
        if config["mode"] == "adaptive":
            raise ValueError("Smoke mode excludes the original-space adaptive rows")
        if config["side"] == "full":
            config["level"] = 4  # Sphere: 2^12 cells. O(3): 2^11 cells.
        config.update(repeats=3, warmup=1, inner=1)
    state = machine_state() if smoke else require_idle()
    if repeats is not None:
        config["repeats"] = repeats
    if warmup is not None:
        config["warmup"] = warmup
    if config["repeats"] < 1 or config["warmup"] < 0:
        raise ValueError("Repeats must be positive; warmups must be nonnegative")
    runner = sphere_repeat if config["example"] == "sphere" else o3_repeat
    raw, warm = [], []
    for index in range(config["warmup"] + config["repeats"]):
        warmup_run = index < config["warmup"]
        samples, last = {}, None
        for _ in range(1 if warmup_run else config["inner"]):
            last = runner(config)
            for key, value in last["stages"].items():
                if value.get("seconds") is not None:
                    samples.setdefault(key, []).append(value["seconds"])
        item = {**last, "stage_samples": samples}
        (warm if warmup_run else raw).append(item)
    stages, totals = summarize_repeats(raw, config)
    graph = raw[-1]["graph"]
    findings = []
    if smoke:
        findings.append("Short validation on a shared machine; not a final timing measurement.")
    for key, value in stages.items():
        spread = value["spread_over_median"]
        if spread is not None and spread > 0.10:
            findings.append(f"{key}: timing spread exceeds 10% of the median.")
        if value["median_s"] < 0:
            findings.append(f"{key}: negative difference at the timer floor; retained in raw results.")
    if any(s.get("cpu_s", 1) / s.get("wall_s", 1) < 0.95 for r in raw for s in r["stages"].values() if s.get("wall_s", 0) > 0):
        findings.append("At least one stage has CPU/wall time below 0.95.")
    if config["example"] == "sphere":
        grid = grid_record(config["side"], config["mode"], config["level"])
        options = {"batch_chunk_size": 65536, "cache_transition_graph": None, "max_cached_edges": 0,
                   "reserve_edges": 0, "reserve_min_edges": 16777216}
    else:
        geom = level_geometry(config["side"], config["level"])
        grid = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in geom.items()}
        grid["subdiv_init"] = geom["depth"] if config["mode"] == "uniform" else 15
        options = {"batch_chunk_size": 512, "cache_transition_graph": True, "max_cached_edges": 0,
                   "reserve_edges": 0, "reserve_min_edges": 0}
    if config["side"] == "reduced" and graph["morse_nodes"] != (5 if config["example"] == "sphere" else 4):
        findings.append("Quotient Morse-node count differs from the reference.")
    if config["mode"] == "adaptive":
        findings.append("Morse-graph time includes the cover, which cannot be isolated on the adaptive grid.")
    undefined = [v for v, annotation in (graph.get("annotations") or {}).items()
                 if not annotation or any("undef" in x.lower() for x in annotation)]
    if undefined:
        findings.append(f"Native Conley indices undefined for nodes {undefined}.")
    return {"schema": SCHEMA, "row": row, "status": "complete", "purpose": "validation" if smoke else "benchmark",
            "config": config, "grid": grid, "cmgdb_options": options, "machine": machine_description(),
            "state_before": state, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "stages": stages, "totals": totals, "graph": graph,
            "repeats": raw, "warmup": warm, "findings": findings, "peak_rss_bytes": peak_rss_bytes(),
            "interpretation": "Native indices of the computed multivalued map; physical index transfer is not certified."}


class TimingSession:
    """Replay context for the timing notebook."""

    def __init__(self, mode, directory, records):
        self.mode, self.directory, self.records = mode, directory, records

    def _repr_html_(self):
        complete = sum(r.get("purpose") == "benchmark" for r in self.records.values())
        return (f"<p>Mode: {escape(self.mode)}. {complete} of 6 benchmark rows available. "
                "Validation records are labeled separately. Times are sums of stage medians.</p>")


def notebook_run(mode="replay", directory="data/timings"):
    """Load records, or execute the guarded all-row command before loading them."""
    if mode not in ("replay", "train"):
        raise ValueError('MODE must be "replay" or "train"')
    directory = Path(directory)
    if not directory.is_absolute():
        directory = REPOSITORY / directory
    if mode == "train":
        result = subprocess.run([sys.executable, str(REPOSITORY / "scripts/run_timings.py"), "all",
                                 "--out", str(directory)], capture_output=True, text=True,
                                env={**os.environ, **THREAD_ENV}, cwd=REPOSITORY)
        if result.returncode:
            # The CLI diagnostics name only the measured state and row identifiers.
            if "Timing run refused:" in result.stderr:
                raise RuntimeError(result.stderr.strip())
            raise RuntimeError("Timing command failed; inspect its JSON status records and run it in a terminal.")
    records = {}
    for row in ROWS:
        for candidate in (directory / f"{row}.json", directory / "validation" / f"{row}.json"):
            if candidate.exists():
                record = json.loads(candidate.read_text())
                if record.get("schema") != SCHEMA or record.get("row") != row:
                    raise ValueError(f"Unexpected timing record for {row}")
                if record.get("status") != "complete":
                    continue
                config = record["config"]
                if any(config[key] != ROWS[row][key] for key in ("example", "side", "mode", "level")):
                    continue  # A small original-space grid never stands in for a published row.
                records[row] = record
                break
    return TimingSession(mode, directory, records)


def _html_table(headers, rows, caption):
    from IPython.display import HTML
    heads = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return HTML(f"<table><caption>{escape(caption)}</caption><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>")


def notebook_machines(session):
    rows, seen = [], set()
    for row, record in session.records.items():
        machine = record["machine"]
        identity = json.dumps(machine, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append([machine["cpu"], machine["system"] + " " + machine["release"], machine["logical_cpus"],
                     f'{machine["memory_bytes"] / 1024**3:.1f} GiB' if machine["memory_bytes"] else "unknown",
                     machine["python"], machine["numpy"], machine["cmgdb"], "1 each"])
    return _html_table(["CPU", "System", "CPUs", "RAM", "Python", "NumPy", "CMGDB", "Numerical threads"],
                       rows, "Machines recorded with the measurements")


def format_seconds(value):
    if value is None:
        return "pending"
    if isinstance(value, str):
        return value
    if value < 1e-3:
        return "<1 ms"
    return f"{value * 1000:.2g} ms" if value < 1 else f"{value:,.3g} s"


def format_cells(value):
    return f"2^{value.bit_length() - 1}" if value > 0 and value & (value - 1) == 0 else f"{value:,}"


def notebook_discussion(session):
    rows = []
    for row in ("sphere-quotient", "o3-quotient", "sphere-uniform", "o3-uniform"):
        record = session.records.get(row)
        rows.append([row, format_seconds(ROWS[row]["paper"][-1]),
                     format_seconds(record["totals"]["total_s"]) if record else "pending",
                     record["purpose"] if record else "waiting for idle run"])
    return _html_table(["Computation", "Section 6", "Measured", "Status"], rows, "Times quoted in Section 6")


def notebook_plant_estimate():
    """Display the arithmetic cost projection without evaluating any plant dynamics."""
    path = REPOSITORY / "data/timings/plant-cost-estimate.json"
    record = json.loads(path.read_text())
    rows = []
    for row in record["rows"]:
        seconds = row["projected_map_plus_morse_s"]
        years = row["projected_map_plus_morse_years"]
        duration = (f"{seconds / 86400:.2g} days" if years < 1 else f"{years:.2g} years") + " (est.)"
        face_count = f'{2 * row["intervals_per_axis"] + 1}^{record["intrinsic_dimension"]}'
        rows.append([format_cells(row["intervals_per_axis"]), format_cells(row["cells"]), face_count,
                     f'{row["hull_table_bytes"]:.3g} bytes (est.)', duration])
    return _html_table(["Intervals per axis", "Top-dimensional cells V", "Full grid including faces C",
                       "Image-hull table only", "Projected map + Morse"],
                       rows, "Full 12-dimensional plant: arithmetic projection only, never run")


def notebook_table(session):
    rows = []
    for row, config in ROWS.items():
        record = session.records.get(row)
        dim = (1 if config["side"] == "reduced" else 3) if config["example"] == "sphere" else (2 if config["side"] == "reduced" else 4)
        common = [dim, "SO(2)" if config["example"] == "sphere" else "O(3)", config["mode"]]
        rows.append(common + [format_cells(config["cells"]), "manuscript"]
                    + [format_seconds(v) for v in config["paper"]])
        measured_cells = record["graph"]["final_cells"] if record else config["cells"]
        values = [record["totals"][k] for k in ("map_s", "morse_graph_s", "conley_index_s", "total_s")] if record else [None] * 4
        rows.append(common + [format_cells(measured_cells), record["purpose"] if record else "waiting for idle run"]
                    + [format_seconds(v) for v in values])
    return _html_table(["Dimension", "G", "Grid", "Cells", "Source", "Map", "Morse graph", "Conley index", "Total"],
                       rows, "Table tab:timing: manuscript values and new measurements")
