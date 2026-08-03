"""line_plot — vertically stacked line-series panels, shared x-axis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DOY_THRESHOLD_HOURS = 24.0


def _flatten_x(panels: list):
    for panel in panels:
        for _, (x, _y) in panel["series"].items():
            yield np.asarray(x)


def _needs_doy_formatting(panels: list) -> "tuple[bool, bool]":
    """(use_doy, show_year), decided once across every series in every panel -- so a stacked
    figure doesn't have some panels in DOY and others in full dates. use_doy is True once the
    combined datetime span exceeds a day (full ISO date labels start overlapping well before
    that on longer plots); show_year is True only if that span actually crosses a year boundary,
    since DOY alone is ambiguous there."""
    datetime_x = [x for x in _flatten_x(panels) if np.issubdtype(x.dtype, np.datetime64)]
    if not datetime_x:
        return False, False

    lo = min(x.min() for x in datetime_x)
    hi = max(x.max() for x in datetime_x)
    span_hours = (hi - lo) / np.timedelta64(1, "h")
    if span_hours <= _DOY_THRESHOLD_HOURS:
        return False, False

    return True, bool(lo.astype("datetime64[Y]") != hi.astype("datetime64[Y]"))


def _doy_formatter(show_year: bool):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    def _format(value, _pos=None):
        dt = mdates.num2date(value)
        doy = dt.timetuple().tm_yday
        return f"{dt.year}-{doy:03d}" if show_year else str(doy)

    return mticker.FuncFormatter(_format)


def line_plot(
    panels: list,
    *,
    out_path: "Path",
    suptitle: "str | None" = None,
    xlabel: str = "time",
    figsize_per_panel: tuple = (10.0, 4.0),
    linewidth: float = 1.8,
    plot_kwargs: "dict | None" = None,
    savefig_kwargs: "dict | None" = None,
) -> "Path":
    """panels: [{"title", "ylabel", "series": {label: (x, y)}, "stats_text": str | None}]. Every
    raw point is plotted -- no time-based downsampling (see satellite_orbit_density's
    orbit_averaged for an explicit, opt-in reduction instead). A datetime x-axis spanning more
    than a day switches to day-of-year tick labels (full dates overlap on longer plots), and
    every panel gets a light gridline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_kwargs = {"linewidth": linewidth, **(plot_kwargs or {})}
    savefig_kwargs = {"dpi": 150, **(savefig_kwargs or {})}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_doy, show_year = _needs_doy_formatting(panels)

    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n), squeeze=False,
    )
    for ax, panel in zip(axes[:, 0], panels):
        for label, (x, y) in panel["series"].items():
            ax.plot(x, y, label=label, **plot_kwargs)
        ax.set_title(panel["title"], fontsize=15)
        ax.set_ylabel(panel.get("ylabel", ""), fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=11)
        if use_doy:
            ax.xaxis.set_major_formatter(_doy_formatter(show_year))
        stats_text = panel.get("stats_text")
        if stats_text:
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va="top", fontsize=11,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})
    axes[-1, 0].set_xlabel("day of year" if use_doy else xlabel, fontsize=13)
    # A single panel already has its own title (ax.set_title above) -- a figure-level suptitle on
    # top of that just repeats the same identifying text a second time. Only worth the second line
    # once there's more than one panel for it to apply across.
    if suptitle and n > 1:
        fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
    return out_path
