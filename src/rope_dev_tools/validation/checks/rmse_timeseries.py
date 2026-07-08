"""rmse_timeseries — forecasts [start, end] and computes RMSE against a
truth CSV over that window."""

from __future__ import annotations

import numpy as np

from rope_dev_tools.validation.checks import passes_threshold, register_kind
from rope_dev_tools.validation.time_utils import resolve_path
from rope_dev_tools.validation.truth_data import load_truth_csv


@register_kind("rmse_timeseries")
def rmse_timeseries(
    model,
    *,
    id=None,
    start,
    end,
    truth_csv,
    variable="density",
    threshold=None,
    unit=None,
    out_dir=None,
    suite_dir=None,
    **_,
) -> dict:
    model.forecast(start, end)

    truth_path = resolve_path(suite_dir, truth_csv)
    truth = load_truth_csv(truth_path)

    errors = []
    for _, row in truth.iterrows():
        result = model.query(
            row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), row["lst"], row["lat"], row["alt_km"],
        )
        errors.append(result[variable] - row[variable])

    rmse = float(np.sqrt(np.mean(np.square(errors))))
    passed = passes_threshold(rmse, threshold) if threshold else None

    return {"value": rmse, "unit": unit, "passed": passed}
