r"""Contain plotting functionalities."""

from __future__ import annotations

__all__ = [
    "plot_null_temporal",
    "plot_cdf",
    "hist_continuous",
]

from flamme.plot.cdf import plot_cdf
from flamme.plot.continuous import hist_continuous
from flamme.plot.null_temp import plot_null_temporal
