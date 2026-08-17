"""lonlat_animation — synced multi-panel lon/lat (or LST/lat) heatmap animation, one frame per timestamp."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rope_dev_tools.validation.plots._common import prepare_out_path, use_agg_backend


def _palette_from_frames(rgb_frames, *, colors: int = 256):
    """Builds a shared 256-color palette from sampled rendered frames so GIF quantization is stable."""
    from PIL import Image

    # Sample first, middle, and last frames to capture the full color range.
    indices = sorted({0, len(rgb_frames) // 2, len(rgb_frames) - 1})
    strips = [rgb_frames[i].resize((rgb_frames[i].width, 1), Image.Resampling.BILINEAR) for i in indices]
    combined = Image.new("RGB", (sum(s.width for s in strips), 1))
    x = 0
    for s in strips:
        combined.paste(s, (x, 0))
        x += s.width
    return combined.quantize(colors=colors, dither=Image.Dither.NONE)


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
    stats_ylabel: str = "%",
    stats_uncertainty_series: "dict | None" = None,
) -> "Path":
    """panel_frames: [{"title", "frames": [(n_x, n_lat) array, ...]}], synced by index to timestamps. stats_series (optional): {metric_name: [scalar, ...]}, drawn as a growing line panel alongside the heatmaps. stats_uncertainty_series (optional): {metric_name: [scalar, ...]}, same keys/length as stats_series, shades a +/- band around each metric's line."""
    plt = use_agg_backend()
    from PIL import Image

    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}
    out_path = prepare_out_path(out_path)

    n_frames = len(timestamps)
    has_stats = bool(stats_series)
    total_cols = n_cols + (1 if has_stats else 0)
    fig, axes = plt.subplots(n_rows, total_cols, squeeze=False, figsize=(4.5 * total_cols, 3.5 * n_rows),
                              constrained_layout=True)
    flat_axes = list(axes.flat)
    stats_ax, heatmap_axes = (flat_axes[0], flat_axes[1:]) if has_stats else (None, flat_axes)
    extent = [x_range[0], x_range[1], lat_range[0], lat_range[1]]

    images = []
    shared_axes, separate_cb_panels = [], []
    for ax, panel in zip(heatmap_axes, panel_frames):
        p_cmap = panel.get("cmap", cmap)
        p_vmin = panel.get("vmin", vmin)
        p_vmax = panel.get("vmax", vmax)
        im = ax.imshow(panel["frames"][0].T, origin="lower", aspect="auto", extent=extent,
                        cmap=p_cmap, vmin=p_vmin, vmax=p_vmax, **imshow_kwargs)
        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Latitude (deg)", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        images.append(im)
        if "cmap" in panel:
            separate_cb_panels.append((im, ax, panel.get("colorbar_label")))
        else:
            shared_axes.append(ax)
    if shared_axes:
        cb = fig.colorbar(images[0], ax=shared_axes, fraction=0.046, pad=0.04)
        cb.set_label("density", fontsize=10)
    for im_sep, ax_sep, cb_label in separate_cb_panels:
        cb = fig.colorbar(im_sep, ax=ax_sep, fraction=0.046, pad=0.04)
        if cb_label:
            cb.set_label(cb_label, fontsize=10)

    stats_lines = {}
    stats_uncerts = {}
    stats_bands = {}
    if has_stats:
        from datetime import datetime
        dtstamps = [datetime.fromisoformat(timestamps[i]) for i in range(len(timestamps))]
        stats_x = np.array([(dtstamps[i] - dtstamps[0]).total_seconds() / 3600.0 for i in range(len(timestamps))], dtype=int)
        stats_uncerts = {k: np.asarray(v, dtype=float) for k, v in (stats_uncertainty_series or {}).items()}
        bounds = [np.asarray(v, dtype=float) for v in stats_series.values()]
        bounds += [np.asarray(v, dtype=float) + u for v, u in
                   ((stats_series[k], stats_uncerts[k]) for k in stats_uncerts)]
        bounds += [np.asarray(v, dtype=float) - u for v, u in
                   ((stats_series[k], stats_uncerts[k]) for k in stats_uncerts)]
        all_y = np.concatenate(bounds)
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

    # Timestamp lives in suptitle, not a manually-placed fig.text, so constrained_layout reserves it a margin.
    title_prefix = f"{suptitle} — " if suptitle else ""
    sup = fig.suptitle(f"{title_prefix}{timestamps[0]}", fontsize=15)

    # Freeze the layout after one pass -- constrained_layout isn't stable frame-to-frame otherwise.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    # Render all frames to full-color RGB, then build a shared palette from the actual content.
    w, h = fig.canvas.get_width_height()
    rgb_frames = []
    for i in range(n_frames):
        for im, panel in zip(images, panel_frames):
            im.set_data(panel["frames"][i].T)
        for name, (line, x, y) in stats_lines.items():
            line.set_data(x[:i + 1], y[:i + 1])
            if name in stats_uncerts:
                if name in stats_bands:
                    stats_bands[name].remove()
                u = stats_uncerts[name]
                stats_bands[name] = stats_ax.fill_between(
                    x[:i + 1], y[:i + 1] - u[:i + 1], y[:i + 1] + u[:i + 1],
                    color=line.get_color(), alpha=0.2, linewidth=0,
                )
        sup.set_text(f"{title_prefix}{timestamps[i]}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        rgb_frames.append(Image.fromarray(buf, mode="RGBA").convert("RGB"))
    plt.close(fig)

    ref_palette = _palette_from_frames(rgb_frames)
    quantized_frames = [f.quantize(palette=ref_palette, dither=Image.Dither.NONE) for f in rgb_frames]

    duration_ms = int(round(1000.0 / fps))
    quantized_frames[0].save(
        out_path, save_all=True, append_images=quantized_frames[1:],
        duration=duration_ms, loop=0, optimize=False, disposal=2, **savefig_kwargs,
    )
    return out_path
