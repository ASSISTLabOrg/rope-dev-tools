"""time_utils — parsing/formatting helpers, and resolve_start_delta's warm-up/lead-time semantics."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from rope_dev_tools.validation.time_utils import lst_values_for, parse_time, resolve_start_delta


def test_lst_values_for_converts_longitude_and_utc_time():
    # lon=0 at UTC noon -> LST noon; lon=180 at UTC noon -> LST midnight (wrapped).
    lst = lst_values_for(np.array([0.0, 180.0, 350.0]), "2024-01-01 12:00:00")
    np.testing.assert_allclose(lst, [12.0, 0.0, (12.0 + 350.0 / 15.0) % 24.0])


def test_lst_values_for_varies_by_time_for_the_same_longitude():
    # The whole point of the underlying fix: the same longitude maps to a different LST at a
    # different time.
    lst_at_0 = lst_values_for(np.array([0.0]), "2024-01-01 00:00:00")
    lst_at_6 = lst_values_for(np.array([0.0]), "2024-01-01 06:00:00")
    assert lst_at_0[0] != lst_at_6[0]


def test_lst_values_for_accepts_scalar_longitude():
    lst = lst_values_for(0.0, "2024-01-01 12:00:00")
    assert lst == pytest.approx(12.0)


def test_resolve_start_delta_zero_is_unchanged():
    forecast_start, query_start_dt = resolve_start_delta(
        "2023-01-01 00:00:00", "2023-01-02 00:00:00", 0
    )
    assert forecast_start == "2023-01-01 00:00:00"
    assert query_start_dt == datetime(2023, 1, 1, 0)


def test_resolve_start_delta_negative_extends_lead_time_without_widening_query_window():
    forecast_start, query_start_dt = resolve_start_delta(
        "2023-01-01 00:00:00", "2023-01-02 00:00:00", -48
    )
    assert forecast_start == "2022-12-30 00:00:00"
    # the evaluated window stays pinned to the period's own start, not the earlier forecast_start
    assert query_start_dt == datetime(2023, 1, 1, 0)


def test_resolve_start_delta_positive_clamps_query_window():
    forecast_start, query_start_dt = resolve_start_delta(
        "2023-01-01 00:00:00", "2023-01-02 00:00:00", 6
    )
    assert forecast_start == "2023-01-01 06:00:00"
    assert query_start_dt == datetime(2023, 1, 1, 6)


def test_resolve_start_delta_consuming_whole_window_raises():
    with pytest.raises(ValueError, match="leaving no time"):
        resolve_start_delta("2023-01-01 00:00:00", "2023-01-02 00:00:00", 24)


def test_resolve_start_delta_exactly_at_end_raises():
    with pytest.raises(ValueError, match="leaving no time"):
        resolve_start_delta("2023-01-01 00:00:00", "2023-01-01 06:00:00", 6)


def test_resolve_start_delta_one_hour_before_end_does_not_raise():
    forecast_start, query_start_dt = resolve_start_delta(
        "2023-01-01 00:00:00", "2023-01-01 06:00:00", 5
    )
    assert parse_time(forecast_start) == datetime(2023, 1, 1, 5)
    assert query_start_dt == datetime(2023, 1, 1, 5)
