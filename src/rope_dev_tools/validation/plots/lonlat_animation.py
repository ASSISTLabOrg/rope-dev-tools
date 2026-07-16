"""lonlat_animation — synced multi-panel LST/latitude heatmap animation, one frame per timestamp."""

from __future__ import annotations

from pathlib import Path


def lonlat_animation(
    panel_frames: list,
    *,
    timestamps: list,
    n_rows: int,
    n_cols: int,
    lat_range: tuple,
    out_path: "Path",
    suptitle: "str | None" = None,
    fps: float = 4.0,
    cmap: str = "viridis",
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    imshow_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
) -> "Path":
    """panel_frames: [{"title", "frames": [(n_lst, n_lat) array, ...]}], one entry per panel, synced by index against timestamps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(timestamps)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(4.0 * n_cols, 3.0 * n_rows))
    extent = [0.0, 24.0, lat_range[0], lat_range[1]]

    images = []
    for ax, panel in zip(axes.flat, panel_frames):
        im = ax.imshow(panel["frames"][0].T, origin="lower", aspect="auto", extent=extent,
                        cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_title(panel["title"])
        ax.set_xlabel("LST (h)")
        ax.set_ylabel("Latitude (deg)")
        images.append(im)
    if suptitle:
        fig.suptitle(suptitle)
    time_text = fig.text(0.5, 0.01, str(timestamps[0]), ha="center")

    def _update(i):
        for im, panel in zip(images, panel_frames):
            im.set_data(panel["frames"][i].T)
        time_text.set_text(str(timestamps[i]))
        return [*images, time_text]

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), **savefig_kwargs)
    plt.close(fig)
    return out_path
