"""doy_lat_plot — single day-of-year (x) vs latitude (y) density heatmap."""

from __future__ import annotations

from pathlib import Path


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    extent = [doy_edges[0], doy_edges[-1], lat_edges[0], lat_edges[-1]]
    im = ax.imshow(grid.T, origin="lower", aspect="auto", extent=extent,
                    cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
    ax.set_xlabel("day of year")
    ax.set_ylabel("latitude (deg)")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="density")
    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
    return out_path
