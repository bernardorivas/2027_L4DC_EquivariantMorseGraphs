"""Attracting-region checks for missing data, window exits, and closed boundaries."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"))
from common import attracting_regions as regions


def interval_grid(count=8):
    edges = np.linspace(-1., 1., count+1)
    return np.column_stack((edges[:-1], edges[1:]))


def test_native_sink_is_not_a_safe_region_when_its_images_leave_the_window():
    boxes = interval_grid()
    images = np.tile([-1.2, -.85], (len(boxes), 1))
    raw, _, graph, arrays, _ = regions.compute_graph(
        boxes, images, boundary_faces=[[False, False]], map_time=1., repeats=0)
    assert len(raw["sink_nodes"]) == 1
    assert sum(raw["basin_cells"].values()) == len(boxes)
    assert arrays["unsafe"].all()
    assert np.all(arrays["basin_node"] < 0)
    assert sum(graph["basin_cells"].values()) == 0


def test_missing_cell_and_every_predecessor_are_removed_from_a_sink_region():
    boxes = interval_grid()
    images = np.full_like(boxes, -.875)
    images[1] = .125
    support = np.ones(len(boxes), dtype=bool)
    support[4] = False
    _, _, graph, arrays, _ = regions.compute_graph(
        boxes, images, support=support, boundary_faces=[[False, False]], map_time=1., repeats=0)
    assert set(np.flatnonzero(arrays["unsafe"])) == {1, 4}
    assert sum(graph["basin_cells"].values()) == 6
    selected = arrays["basin_node"] >= 0
    specs = {"euclidean": dict(metric=regions.qc.Euclidean(), boxes=arrays["boxes"], images=arrays["images"],
              predicted=np.array([[-.875]]), encoded_image=np.array([[-.8]]))}
    rows, _ = regions.compare_regions(graph, arrays, [[-.875]], specs, boundary_faces=[[False, False]])
    assert rows[0]["comparisons"]["euclidean"]["tolerance_lower"] == pytest.approx(.125)
    assert rows[0]["comparisons"]["euclidean"]["passed"]
    assert selected.sum() == 6


def test_boundary_samples_are_included_in_the_closed_region_comparison():
    boxes = interval_grid(4)
    arrays = dict(boxes=boxes, images=np.full_like(boxes, -.75), basin_node=np.array([0, 0, -1, -1]))
    spec = dict(metric=regions.qc.Euclidean(), boxes=boxes, images=arrays["images"],
                predicted=np.array([[-.1], [-.25]]), encoded_image=np.array([[.1], [-.24]]))
    rows, _ = regions.compare_regions(dict(sink_nodes=[0]), arrays, [[0.], [.25]],
                                     {"euclidean": spec}, boundary_faces=[[True, True]])
    result = rows[0]["comparisons"]["euclidean"]
    assert result["count"] == 1
    assert result["max_residual"] == pytest.approx(.2)
    assert result["tolerance_lower"] == pytest.approx(.75)
    assert result["certified"] is False


def test_infinite_whole_domain_tolerance_remains_valid_strict_json():
    boxes = interval_grid()
    images = np.full_like(boxes, .1)
    _, _, graph, arrays, _ = regions.compute_graph(
        boxes, images, boundary_faces=[[True, True]], map_time=1., repeats=0)
    spec = dict(metric=regions.qc.Euclidean(), boxes=arrays["boxes"], images=arrays["images"],
                predicted=np.array([[0.]]), encoded_image=np.array([[.01]]))
    rows, _ = regions.compare_regions(graph, arrays, [[.2]], {"euclidean": spec},
                                     boundary_faces=[[True, True]])
    result = regions.plain(rows[0]["comparisons"]["euclidean"])
    assert result["infinite_tolerance"]
    assert result["tolerance_lower"] is None
    assert result["certified"] is False
    json.dumps(result, allow_nan=False)


def test_missing_pole_structure_does_not_trigger_a_coning_merge():
    boxes = interval_grid()
    images = np.column_stack((boxes[:, 0]/2, boxes[:, 1]/2))
    raw, _, graph, _, _ = regions.compute_graph(boxes, images, boundary_faces=[[True, True]],
        map_time=1., aggregation="coning", repeats=0)
    assert not graph["aggregation"]["applied"]
    assert "pole" in graph["aggregation"]["reason"]
    assert graph["nodes"] == raw["nodes"]
