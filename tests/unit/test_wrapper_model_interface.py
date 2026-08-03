"""WrapperModelInterface.query_grid()/query_grid_at() -- nearest-cell lookup, including at
arbitrary (non-native-grid) lst/lat values, used by lonlat_snapshot_series to sample ROPE onto a
truth-data grid of a different resolution."""

from __future__ import annotations

import numpy as np

from rope_dev_tools.validation.model_interfaces import (
    WrapperModelInterface,
    WrapperRequest,
    WrapperResponse,
)

_GRID = {"n_lst": 4, "n_lat": 3, "lat_min_deg": -80.0, "lat_max_deg": 80.0,
         "alt_min_km": 100.0, "alt_max_km": 500.0, "n_alt": 2}


def _model():
    # density[t, lst, lat, alt] = lst*10 + lat, distinct per cell so lookups are hand-verifiable.
    density = np.zeros((1, 4, 3, 2))
    for lst in range(4):
        for lat in range(3):
            density[0, lst, lat, :] = lst * 10 + lat
    response = WrapperResponse(times=["2024-01-01 00:00:00"], density=density, uncertainty=density)

    def wrapper_fn(request: WrapperRequest) -> WrapperResponse:
        return response

    model = WrapperModelInterface(wrapper_fn, grid=_GRID)
    model.forecast("2024-01-01 00:00:00", "2024-01-01 00:00:00")
    return model


def test_query_grid_returns_native_grid():
    model = _model()
    grid = model.query_grid("2024-01-01 00:00:00", alt_km=100.0)
    assert grid.shape == (4, 3)
    expected = np.array([[lst * 10 + lat for lat in range(3)] for lst in range(4)])
    np.testing.assert_allclose(grid, expected)


def test_query_grid_at_native_points_matches_query_grid():
    model = _model()
    lst_values = np.linspace(0, 24, 4, endpoint=False)
    lat_values = np.linspace(-80.0, 80.0, 3)
    grid_at = model.query_grid_at("2024-01-01 00:00:00", 100.0, lst_values, lat_values)
    grid = model.query_grid("2024-01-01 00:00:00", alt_km=100.0)
    np.testing.assert_allclose(grid_at, grid)


def test_query_grid_at_arbitrary_points_uses_nearest_cell():
    model = _model()
    # lst=7.0h is nearest to native bin 1 (6h, since bins are 0,6,12,18); lat=79 is nearest to bin 2 (80).
    grid_at = model.query_grid_at("2024-01-01 00:00:00", 100.0, [7.0], [79.0])
    assert grid_at.shape == (1, 1)
    assert grid_at[0, 0] == 1 * 10 + 2


def test_query_grid_at_different_output_shape_than_native_grid():
    model = _model()
    lst_values = np.linspace(0, 24, 6, endpoint=False)
    lat_values = np.linspace(-80.0, 80.0, 5)
    grid_at = model.query_grid_at("2024-01-01 00:00:00", 100.0, lst_values, lat_values)
    assert grid_at.shape == (6, 5)
