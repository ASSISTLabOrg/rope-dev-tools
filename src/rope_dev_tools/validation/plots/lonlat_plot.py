"""lonlat_plot — NxM grid of LST/latitude density heatmaps."""

from __future__ import annotations

from pathlib import Path


def lonlat_plot(
    panels: list,
    *,
    n_rows: int,
    n_cols: int,
    lat_range: tuple,
    lst_range: tuple = (0.0, 24.0),
    out_path: "Path",
    suptitle: "str | None" = None,
    cmap: str = "viridis",
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    imshow_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
) -> "Path":
    """panels: [{"title", "grid": (n_lst, n_lat) array}], row-major, len == n_rows * n_cols."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(4.0 * n_cols, 3.0 * n_rows))
    extent = [lst_range[0], lst_range[1], lat_range[0], lat_range[1]]
    im = None
    for ax, panel in zip(axes.flat, panels):
        im = ax.imshow(panel["grid"].T, origin="lower", aspect="auto", extent=extent,
                        cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_title(panel["title"])
        ax.set_xlabel("LST (h)")
        ax.set_ylabel("Latitude (deg)")
    if im is not None:
        fig.colorbar(im, ax=axes, label="density")
    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
    return out_path
