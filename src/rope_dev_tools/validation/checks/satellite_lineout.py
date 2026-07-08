"""satellite_lineout — forecasts [start, end], produces a trace plot +
RMSE-along-track against a satellite track."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rope_dev_tools.validation.checks import passes_threshold, register_kind
from rope_dev_tools.validation.time_utils import resolve_path
from rope_dev_tools.validation.truth_data import load_truth_csv


@register_kind("satellite_lineout")
def satellite_lineout(
    model,
    *,
    id=None,
    start,
    end,
    satellite_track_csv,
    variable="density",
    threshold=None,
    unit=None,
    out_dir=None,
    suite_dir=None,
    **_,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.forecast(start, end)

    track_path = resolve_path(suite_dir, satellite_track_csv)
    track = load_truth_csv(track_path)

    predicted, truth_values = [], []
    for _, row in track.iterrows():
        result = model.query(
            row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), row["lst"], row["lat"], row["alt_km"],
        )
        predicted.append(result[variable])
        truth_values.append(row[variable])

    predicted = np.asarray(predicted)
    truth_values = np.asarray(truth_values)
    rmse = float(np.sqrt(np.mean(np.square(predicted - truth_values))))
    passed = passes_threshold(rmse, threshold) if threshold else None

    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(truth_values, label="truth")
    ax.plot(predicted, label="predicted")
    ax.set_xlabel("track point")
    ax.set_ylabel(unit or variable)
    ax.set_title(f"{id or 'satellite_lineout'} ({variable})")
    ax.legend()
    plot_name = f"plots/{id or 'satellite_lineout'}.png"
    fig.savefig(out_dir / plot_name)
    plt.close(fig)

    return {"value": rmse, "unit": unit, "passed": passed, "plots": [plot_name]}
