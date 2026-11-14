"""Structure of the saved notebook of Section 5.2: one step and one output per code cell."""
import json
from pathlib import Path
import re

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "ex2_o3_flow.ipynb"
LOCAL_PATH = re.compile(r"/(Users|home|private|tmp)/")


@pytest.fixture(scope="module")
def code_cells():
    if not NOTEBOOK.exists():
        pytest.skip("no saved notebook")
    notebook = json.loads(NOTEBOOK.read_text())
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_first_cell_sets_the_mode_and_the_output(code_cells):
    first = "".join(code_cells[0]["source"])
    assert 'MODE = "replay"' in first and "OUT = None" in first
    assert not any("MODE =" in "".join(cell["source"]) for cell in code_cells[1:])


def test_each_code_cell_has_one_output_and_no_local_path(code_cells):
    for cell in code_cells:
        outputs = cell.get("outputs", [])
        assert len(outputs) == 1, "".join(cell["source"])
        assert outputs[0]["output_type"] in ("execute_result", "display_data"), "".join(cell["source"])
        for kind, value in outputs[0]["data"].items():
            assert kind in ("text/plain", "text/html", "image/png")
            if kind.startswith("text/"):
                text = "".join(value) if isinstance(value, list) else value
                assert not LOCAL_PATH.search(text), text[:200]


def test_saved_replay_passed(code_cells):
    outputs = ["".join(cell["outputs"][0]["data"].get("text/plain", "")) for cell in code_cells]
    assert any(text.startswith("The record matches data/ex2_o3_flow/reference/results.json") for text in outputs)
