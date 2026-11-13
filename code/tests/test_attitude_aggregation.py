"""Check structural aggregation, connecting cells, and recomputed indices."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"))
from common.aggregate_interval import from_saved, require_order_convex
from common.workflow import read


def test_coning_aggregate_recomputes_the_index_and_absorbs_connections():
    source = Path(__file__).resolve().parents[1]/"examples/example3/periodic/reference/coning/seed0/cmgdb"
    with np.load(source/"phase.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    native = read(source/"results.json")
    result, phase = from_saved(native, arrays, True)
    assert result["edges"] == [[1, 0], [2, 0]]
    assert [row["annotations"] for row in result["nodes"]] == [["x-1", "0"], ["0", "0"], ["0", "0"]]
    assert result["nodes"][0]["original_nodes"] == [0, 2]
    assert result["nodes"][0]["cell_ids"] == [79, 80]
    assert result["basin_cells"] == {"0": 126}
    assert np.sum(phase["basin_node"] == 0) == 126
    assert result["aggregation"]["index_pairs_checked"]


def test_nonconvex_group_is_rejected():
    with pytest.raises(ValueError, match="order convex"):
        require_order_convex({0, 1, 2}, [[0, 1], [1, 2]], {0, 2})


def test_convex_group_with_an_internal_connection_is_allowed():
    require_order_convex({0, 1, 2}, [[0, 1], [1, 2]], {0, 1})
