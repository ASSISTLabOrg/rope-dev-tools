"""lonlat_density_plot — LST/lat density map at given altitudes, from model+time_point or a raw density_2d array. Plot-only, no scalar score."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rope_dev_tools.validation.checks import register_kind
from rope_dev_tools.validation.time_utils import add_hours


@register_kind("lonlat_density_plot")
def lonlat_density_plot(
    model=None,
    *,
    id=None,
    time_point=None,
    time_window_hours=None,
    density_2d=None,
    lst=None,
    lat=None,
    altitudes_km=None,
    out_dir=None,
    **_,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    if density_2d is not None:
        grids = [(None, np.asarray(density_2d))]
    else:
        if model is None or time_point is None or time_window_hours is None:
            raise ValueError(
                "lonlat_density_plot needs either density_2d (a raw array), or "
                "model + time_point + time_window_hours"
            )
        start = add_hours(time_point, -time_window_hours)
        model.forecast(start, time_point)
        grids = [(alt_km, model.query_grid(time_point, alt_km)) for alt_km in (altitudes_km or [None])]

    plots = []
    for alt_km, grid in grids:
        fig, ax = plt.subplots()
        im = ax.imshow(grid.T, origin="lower", aspect="auto", extent=[0, 24, -87.5, 87.5])
        ax.set_xlabel("LST (h)")
        ax.set_ylabel("Latitude (deg)")
        title = "density"
        if time_point:
            title += f" @ {time_point}"
        if alt_km is not None:
            title += f", {alt_km} km"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="density")
        suffix = f"_{int(alt_km)}km" if alt_km is not None else ""
        plot_name = f"plots/{id or 'lonlat'}{suffix}.png"
        fig.savefig(out_dir / plot_name)
        plt.close(fig)
        plots.append(plot_name)

    return {"plots": plots}
