"""Grids, sampled image hulls, native Morse graphs and merged Morse sets."""
import numpy as np
import pytest

from morsegraphs import grids


def test_cells_are_in_product_order():
    boxes = grids.cells(4, [0., 0.], [1., 2.])
    assert boxes.shape == (16, 4)
    assert boxes[1].tolist() == [0., .5, .25, 1.]  # the last coordinate varies fastest
    assert np.allclose(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1).sum(), 2.)
    with pytest.raises(ValueError):
        grids.cells(3, [0.], [1.])


def test_sampled_hulls_include_the_corners_and_are_cut_at_faces():
    boxes = grids.cells(4, [-1.], [1.])
    hulls = grids.sampled_hulls(lambda p: 2 * p, boxes, 5, 0., [-1.], [1.])
    assert hulls[0, 0] == -1. and hulls[0, 1] == np.nextafter(-1., 0.)  # [-2, -1] cut at the face -1
    assert hulls[1, 0] == -1. and 0. <= hulls[1, 1] < 1e-300  # [-1, 0], rounded outward
    padded = grids.sampled_hulls(lambda p: p, boxes, 2, .1)
    assert np.all(padded[:, 0] < boxes[:, 0] - .09) and np.all(padded[:, 1] > boxes[:, 1] + .09)


def test_graph_of_a_map_with_two_attractors():
    # z -> 0.6 tanh(10 z): a repeller at 0 and two attracting fixed points near -0.6 and 0.6
    boxes = grids.cells(64, [-1.], [1.])
    hulls = grids.sampled_hulls(lambda p: .6 * np.tanh(10 * p), boxes, 9, 1e-12, [-1.], [1.])
    graph = grids.morse_graph(boxes, hulls)
    annotations = sorted(tuple(row["annotations"]) for row in graph.nodes)
    assert annotations == [("0", "x-1"), ("x-1", "0"), ("x-1", "0")]
    repeller = next(row["node"] for row in graph.nodes if row["annotations"] == ["0", "x-1"])
    assert sorted(b for a, b in graph.edges if a == repeller) == sorted(graph.minimal)
    for row in graph.nodes:
        assert np.all(graph.membership[row["cell_ids"]] == row["node"])


def test_merge_recomputes_the_index_of_an_attractor_with_a_trivial_neighbor():
    boxes = grids.cells(8, [-1.], [1.])
    images = np.tile([-.25, -.25], (8, 1))
    images[3] = [-.5, -.25]   # cell 3 = [-.25, 0] maps over cells 2 and 3
    images[2] = [-.375, -.3]  # cell 2 = [-.5, -.25] maps into itself
    graph = grids.morse_graph(boxes, images)
    assert len(graph.nodes) >= 1
    merged = grids.merge(graph, [[row["node"] for row in graph.nodes if 2 in row["cell_ids"] or 3 in row["cell_ids"]]])
    node = next(row for row in merged.nodes if 2 in row["cell_ids"])
    assert node["annotations"][0] == "x-1"


def test_order_convex():
    assert not grids.order_convex([0, 1, 2], [[0, 1], [1, 2]], [0, 2])
    assert grids.order_convex([0, 1, 2], [[0, 1], [1, 2]], [0, 1])
