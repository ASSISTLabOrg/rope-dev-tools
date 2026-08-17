"""line_plot — vertically stacked line-series panels, shared x-axis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rope_dev_tools.validation.plots._common import savefig, use_agg_backend

_DOY_THRESHOLD_HOURS = 24.0


def _flatten_x(panels: list):
    """Yields every panel's every series' x array."""
    for panel in panels:
        for values in panel["series"].values():
            yield np.asarray(values[0])


def _needs_doy_formatting(panels: list) -> "tuple[bool, bool]":
    """(use_doy, show_year), decided once across every panel's series; use_doy once span > a day."""
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
    """Tick formatter: 'YYYY-DDD' if show_year, else plain 'DDD'."""
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
    """panels: [{"title", "ylabel", "series": {label: (x, y) or (x, y, y_uncert)}, "stats_text": str | None}].
    A 3rd, optional series element shades a +/- y_uncert band around that series' own line. No downsampling."""
    plt = use_agg_backend()

    plot_kwargs = {"linewidth": linewidth, **(plot_kwargs or {})}
    savefig_kwargs = savefig_kwargs or {}

    use_doy, show_year = _needs_doy_formatting(panels)

    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n), squeeze=False,
    )
    for ax, panel in zip(axes[:, 0], panels):
        for label, values in panel["series"].items():
            x, y, *rest = values
            line, = ax.plot(x, y, label=label, **plot_kwargs)
            if rest and rest[0] is not None:
                y_uncert = np.asarray(rest[0])
                ax.fill_between(x, np.asarray(y) - y_uncert, np.asarray(y) + y_uncert,
                                 color=line.get_color(), alpha=0.2, linewidth=0)
        ax.set_title(panel["title"], fontsize=15)
        ax.set_ylabel(panel.get("ylabel", ""), fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=11)
        if use_doy:
            ax.xaxis.set_major_formatter(_doy_formatter(show_year))
        ylim = panel.get("ylim")
        if ylim is not None:
            ax.set_ylim(top=ylim)
        stats_text = panel.get("stats_text")
        if stats_text:
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va="top", fontsize=11,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})
    axes[-1, 0].set_xlabel("day of year" if use_doy else xlabel, fontsize=13)
    # Skip suptitle for a single panel -- it already has its own title, this would repeat it.
    if suptitle and n > 1:
        fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    return savefig(fig, out_path, **savefig_kwargs)
