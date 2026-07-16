"""line_plot — vertically stacked line-series panels, shared x-axis."""

from __future__ import annotations

from pathlib import Path


def line_plot(
    panels: list,
    *,
    out_path: "Path",
    suptitle: "str | None" = None,
    xlabel: str = "time",
    figsize_per_panel: tuple = (8.0, 2.5),
    plot_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
) -> "Path":
    """panels: [{"title", "ylabel", "series": {label: (x, y)}, "stats_text": str | None}]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_kwargs = plot_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n), squeeze=False,
    )
    for ax, panel in zip(axes[:, 0], panels):
        for label, (x, y) in panel["series"].items():
            ax.plot(x, y, label=label, **plot_kwargs)
        ax.set_title(panel["title"])
        ax.set_ylabel(panel.get("ylabel", ""))
        ax.legend()
        stats_text = panel.get("stats_text")
        if stats_text:
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va="top",
                    fontsize=8, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})
    axes[-1, 0].set_xlabel(xlabel)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
    return out_path
