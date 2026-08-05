"""doy_lat_plot — single day-of-year (x) vs latitude (y) density heatmap."""

from __future__ import annotations

from pathlib import Path

from rope_dev_tools.validation.plots._common import add_density_colorbar, savefig, use_agg_backend


def doy_lat_plot(
    grid,
    *,
    doy_edges,
    lat_edges,
    out_path: "Path",
    title: "str | None" = None,
    cmap: str = "viridis",
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    imshow_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
) -> "Path":
    """grid: (n_doy_bins, n_lat_bins) array."""
    plt = use_agg_backend()

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    extent = [doy_edges[0], doy_edges[-1], lat_edges[0], lat_edges[-1]]
    im = ax.imshow(grid.T, origin="lower", aspect="auto", extent=extent,
                    cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
    ax.set_xlabel("day of year", fontsize=12)
    ax.set_ylabel("latitude (deg)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    if title:
        ax.set_title(title, fontsize=13)
    add_density_colorbar(fig, im, ax)
    return savefig(fig, out_path, **savefig_kwargs)
