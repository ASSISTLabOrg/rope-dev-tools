"""lonlat_plot — NxM grid of lon/lat (or LST/lat) density heatmaps."""

from __future__ import annotations

from pathlib import Path

from rope_dev_tools.validation.plots._common import add_density_colorbar, savefig, use_agg_backend


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
    """panels: [{"title", "grid": (n_x, n_lat) array, "cmap"/"vmin"/"vmax"/"colorbar_label" (optional per-panel overrides)}], row-major, len == n_rows * n_cols. A panel with its own "cmap" gets its own colorbar instead of sharing the default one."""
    plt = use_agg_backend()

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(4.5 * n_cols, 3.5 * n_rows),
                              constrained_layout=True)
    extent = [x_range[0], x_range[1], lat_range[0], lat_range[1]]
    shared_images, shared_axes = [], []
    for ax, panel in zip(axes.flat, panels):
        p_cmap = panel.get("cmap", cmap)
        p_vmin = panel.get("vmin", vmin)
        p_vmax = panel.get("vmax", vmax)
        im = ax.imshow(panel["grid"].T, origin="lower", aspect="auto", extent=extent,
                        cmap=p_cmap, vmin=p_vmin, vmax=p_vmax, **imshow_kwargs)
        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Latitude (deg)", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        if "cmap" in panel:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if panel.get("colorbar_label"):
                cb.set_label(panel["colorbar_label"], fontsize=10)
        else:
            shared_images.append(im)
            shared_axes.append(ax)
    if shared_images:
        add_density_colorbar(fig, shared_images[0], shared_axes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    return savefig(fig, out_path, **savefig_kwargs)
