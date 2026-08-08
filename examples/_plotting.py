"""Shared plotting and command-line helpers for the pockit examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#202020",
}


def configure_matplotlib() -> None:
    """Apply the common, colorblind-friendly style used by the examples."""
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                color=[
                    COLORS["blue"],
                    COLORS["orange"],
                    COLORS["green"],
                    COLORS["vermillion"],
                    COLORS["purple"],
                    COLORS["sky_blue"],
                ]
            ),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "savefig.dpi": 180,
        }
    )


def style_axes(axes: Any) -> None:
    """Add a restrained grid and consistent tick styling to one or more axes."""
    for axis in np.asarray(axes, dtype=object).reshape(-1):
        axis.grid(True, color="#B8B8B8", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=4, width=0.8)


def require_finite(**values: Any) -> None:
    """Raise when a numerical result contains NaN or infinite values."""
    for name, value in values.items():
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} contains non-finite values")


def save_or_show(
    fig: Any, save: str | Path | None = None, show: bool = True
) -> Path | None:
    """Save a figure when requested, show it when requested, and close otherwise."""
    output_path = None
    if save is not None:
        output_path = Path(save).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def parse_plot_arguments(
    description: str, default_filename: str, *, quick: bool = False
) -> argparse.Namespace:
    """Parse the plotting options shared by all standalone examples."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--save",
        nargs="?",
        const=default_filename,
        default=None,
        metavar="PATH",
        help=f"save the figure (default path: {default_filename})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open an interactive figure window",
    )
    if quick:
        parser.add_argument(
            "--quick",
            action="store_true",
            help="use a coarse configuration intended only for a smoke test",
        )
    return parser.parse_args()
