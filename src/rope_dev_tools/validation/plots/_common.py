"""Matplotlib boilerplate shared by plots/*.py -- internal, not part of the public plotting API."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_DPI = 150


def use_agg_backend():
    """Forces the headless Agg backend and returns matplotlib.pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def prepare_out_path(out_path) -> Path:
    """Path(out_path) with its parent directory created."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def savefig(fig, out_path, **savefig_kwargs) -> Path:
    """fig.savefig(out_path), dpi=150 default; closes fig; returns out_path."""
    import matplotlib.pyplot as plt

    out_path = prepare_out_path(out_path)
    fig.savefig(out_path, **{"dpi": _DEFAULT_DPI, **savefig_kwargs})
    plt.close(fig)
    return out_path


def add_density_colorbar(fig, im, axes, *, label: str = "density") -> None:
    """Adds one shared colorbar for im across axes, styled consistently."""
    cbar = fig.colorbar(im, ax=axes, label=label)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label, fontsize=12)
