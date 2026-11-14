"""The attitude rows of the timing table: the coning height map and the row definitions."""
import numpy as np

from morsegraphs import attitude_control, timing


def test_coning_time_one_map_matches_the_integrated_flow():
    z = np.concatenate([np.linspace(-1, 1, 801), 1 - np.geomspace(1e-12, 1e-2, 20), -1 + np.geomspace(1e-12, 1e-2, 20)])
    closed = timing.h1_coning_float(z)
    assert np.max(np.abs(closed - attitude_control.height_flow(z, 1.0, "coning"))) < 1e-11
    assert np.array_equal(timing.h1_coning_float(np.array([-1.0, 0.25, 1.0])), [-1.0, 0.25, 1.0])
    assert np.all(np.diff(timing.h1_coning_float(np.linspace(-1, 1, 20001))) >= 0)


def test_attitude_rows_mirror_the_sphere_rows():
    for side in ("uniform", "quotient"):
        sphere, attitude = timing.ROWS[f"sphere-{side}"], timing.ROWS[f"attitude-{side}"]
        assert attitude["example"] == "attitude" and timing.HEIGHT["attitude"] is timing.h1_coning_float
        assert all(attitude[key] == sphere[key] for key in ("side", "mode", "level", "cells"))
