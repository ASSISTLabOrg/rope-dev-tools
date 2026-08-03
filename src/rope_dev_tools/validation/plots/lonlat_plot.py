"""lonlat_plot — NxM grid of lon/lat (or LST/lat) density heatmaps."""

from __future__ import annotations

from pathlib import Path


def lonlat_plot(
    panels: list,
    *,
    n_rows: int,
    n_cols: int,
    lat_range: tuple,
    x_range: tuple = (0.0, 24.0),
    xlabel: str = "LST (h)",
    out_path: "Path",
    suptitle: "str | None" = None,
    cmap: str = "viridis",
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    imshow_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
) -> "Path":
    """panels: [{"title", "grid": (n_x, n_lat) array}], row-major, len == n_rows * n_cols. x_range
    defaults to LST hours; pass x_range=(lon_min, lon_max), xlabel="Longitude (deg)" for a
    longitude-native grid instead."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = {"dpi": 150, **(savefig_kwargs or {})}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(4.5 * n_cols, 3.5 * n_rows),
                              constrained_layout=True)
    extent = [x_range[0], x_range[1], lat_range[0], lat_range[1]]
    im = None
    for ax, panel in zip(axes.flat, panels):
        im = ax.imshow(panel["grid"].T, origin="lower", aspect="auto", extent=extent,
                        cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Latitude (deg)", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, label="density")
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("density", fontsize=12)
    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
    return out_path
