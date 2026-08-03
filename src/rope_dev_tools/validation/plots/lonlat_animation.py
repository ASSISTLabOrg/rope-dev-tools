"""lonlat_animation — synced multi-panel lon/lat (or LST/lat) heatmap animation, one frame per timestamp."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _reference_palette_image(cmap, colors: int = 256):
    """A 1xN image sampling the full colormap gradient (plus explicit white/black for the figure's
    background and text/axes, which aren't part of the colormap itself), quantized to its own
    adaptive palette. Used as a shared reference so every animation frame is quantized against the
    same palette -- otherwise Pillow's GIF writer computes an independent 256-color palette per
    frame (a well-known gotcha), making the same data value render as a visibly different color
    from one frame to the next even though vmin/vmax/cmap never change."""
    from PIL import Image

    n_cmap_colors = colors - 2
    gradient = (cmap(np.linspace(0.0, 1.0, n_cmap_colors))[:, :3] * 255).astype(np.uint8)
    extras = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)  # background white, text black
    combined = np.concatenate([gradient, extras], axis=0)
    return Image.fromarray(combined.reshape(1, colors, 3), mode="RGB").convert(
        "P", palette=Image.ADAPTIVE, colors=colors,
    )


def lonlat_animation(
    panel_frames: list,
    *,
    timestamps: list,
    n_rows: int,
    n_cols: int,
    lat_range: tuple,
    x_range: tuple = (0.0, 24.0),
    xlabel: str = "LST (h)",
    out_path: "Path",
    suptitle: "str | None" = None,
    fps: float = 4.0,
    cmap: str = "viridis",
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    imshow_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
    stats_series: "dict | None" = None,
    stats_ylabel: str = "",
) -> "Path":
    """panel_frames: [{"title", "frames": [(n_x, n_lat) array, ...]}], one entry per panel, synced
    by index against timestamps. x_range defaults to LST hours; pass x_range=(lon_min, lon_max),
    xlabel="Longitude (deg)" for a longitude-native grid instead.

    stats_series, if given: {metric_name: [scalar, ...]}, one value per timestamp -- drawn as an
    extra line-plot panel to the left of the heatmap panels, with each line growing to reveal one
    more point per frame in sync with the heatmaps (rather than showing the whole trace from
    frame 0), matching how the heatmap panels themselves reveal one more timestamp per frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(timestamps)
    has_stats = bool(stats_series)
    total_cols = n_cols + (1 if has_stats else 0)
    fig, axes = plt.subplots(n_rows, total_cols, squeeze=False, figsize=(4.5 * total_cols, 3.5 * n_rows),
                              constrained_layout=True)
    flat_axes = list(axes.flat)
    stats_ax, heatmap_axes = (flat_axes[0], flat_axes[1:]) if has_stats else (None, flat_axes)
    extent = [x_range[0], x_range[1], lat_range[0], lat_range[1]]

    images = []
    for ax, panel in zip(heatmap_axes, panel_frames):
        im = ax.imshow(panel["frames"][0].T, origin="lower", aspect="auto", extent=extent,
                        cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Latitude (deg)", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        images.append(im)
    cbar = fig.colorbar(images[0], ax=heatmap_axes, label="density")
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("density", fontsize=12)

    stats_lines = {}
    if has_stats:
        from datetime import datetime
        dtstamps = [datetime.fromisoformat(timestamps[i]) for i in range(len(timestamps))]
        stats_x = np.array([(dtstamps[i] - dtstamps[0]).total_seconds() / 3600.0 for i in range(len(timestamps))], dtype=int)
        all_y = np.concatenate([np.asarray(v, dtype=float) for v in stats_series.values()])
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
        pad = (y_max - y_min) * 0.05 or (abs(y_min) * 0.05 or 0.01)
        stats_ax.set_xlim(stats_x[0], stats_x[-1])
        stats_ax.set_ylim(y_min - pad, y_max + pad)
        stats_ax.set_title("statistics", fontsize=13)
        stats_ax.set_xlabel("hours", fontsize=12)
        stats_ax.set_ylabel(stats_ylabel, fontsize=12)
        stats_ax.tick_params(axis="both", labelsize=10)
        stats_ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        for name, values in stats_series.items():
            line, = stats_ax.plot([], [], label=name, linewidth=1.8)
            stats_lines[name] = (line, stats_x, np.asarray(values, dtype=float))
        stats_ax.legend(loc="upper right", fontsize=10)

    # The timestamp lives in the suptitle (a constrained_layout-managed artist) rather than a
    # manually-placed fig.text at a fixed figure-fraction position -- the latter doesn't get a
    # reserved margin from constrained_layout and can overlap the bottom row's xlabels.
    title_prefix = f"{suptitle} — " if suptitle else ""
    sup = fig.suptitle(f"{title_prefix}{timestamps[0]}", fontsize=15)

    # constrained_layout recomputes the whole figure's geometry on every draw, and that
    # recomputation isn't perfectly stable frame-to-frame in an animation loop -- freeze it after
    # one pass so every subsequent frame only updates pixel data/text, never re-flows the figure.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    # Every frame is quantized against one shared reference palette (see
    # _reference_palette_image) before being handed to Pillow's GIF writer -- letting Pillow
    # quantize each frame independently (its default behavior for a plain RGB sequence) is what
    # actually caused the reported "colorbar changes over time" symptom: the color-to-value
    # mapping was already correct and constant, but each frame's approximation of it wasn't.
    ref_palette = _reference_palette_image(images[0].cmap)
    w, h = fig.canvas.get_width_height()
    quantized_frames = []
    for i in range(n_frames):
        for im, panel in zip(images, panel_frames):
            im.set_data(panel["frames"][i].T)
        for line, x, y in stats_lines.values():
            line.set_data(x[:i + 1], y[:i + 1])
        sup.set_text(f"{title_prefix}{timestamps[i]}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame_img = Image.fromarray(buf, mode="RGBA").convert("RGB")
        quantized_frames.append(frame_img.quantize(palette=ref_palette, dither=Image.Dither.NONE))
    plt.close(fig)

    duration_ms = int(round(1000.0 / fps))
    quantized_frames[0].save(
        out_path, save_all=True, append_images=quantized_frames[1:],
        duration=duration_ms, loop=0, optimize=False, disposal=2, **savefig_kwargs,
    )
    return out_path
