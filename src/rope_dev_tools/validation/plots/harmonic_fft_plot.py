"""harmonic_fft_plot — side-by-side FFT-magnitude panels (log y-axis), with vertical reference lines at diurnal-harmonic frequencies."""

from __future__ import annotations

from pathlib import Path

from rope_dev_tools.validation.plots._common import savefig, use_agg_backend


def harmonic_fft_plot(
    panels: list,
    *,
    harmonic_freqs_per_hour: "list[float]",
    out_path: "Path",
    suptitle: "str | None" = None,
    xlabel: str = "frequency (hr$^{-1}$)",
    ylabel: str = "|FFT|",
    figsize_per_panel: tuple = (6.0, 4.5),
    linewidth: float = 1.8,
    plot_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
) -> "Path":
    """panels: [{"title", "series": {label: (freqs, magnitude)}}], side by side, log y-axis, harmonic ref lines. Caller must de-bias (drop DC bin) and low-pass filter first -- log(0) is undefined."""
    plt = use_agg_backend()

    plot_kwargs = {"linewidth": linewidth, **(plot_kwargs or {})}
    savefig_kwargs = savefig_kwargs or {}

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for label, (freqs, magnitude) in panel["series"].items():
            ax.plot(freqs, magnitude, label=label, **plot_kwargs)
        for hf in harmonic_freqs_per_hour:
            ax.axvline(hf, color="gray", linestyle=":", linewidth=1.0, zorder=0)
            ax.text(hf, 0.98, f"{1.0 / hf:.0f}h", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="gray", rotation=90)
        ax.set_yscale("log")
        ax.set_title(panel["title"], fontsize=15)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    return savefig(fig, out_path, **savefig_kwargs)
