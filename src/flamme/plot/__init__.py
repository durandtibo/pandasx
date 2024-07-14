r"""Contain plotting functionalities."""

from __future__ import annotations

__all__ = [
    "hist_continuous",
    "hist_continuous2",
    "bar_discrete",
    "bar_discrete_temporal",
    "plot_cdf",
    "plot_null_temporal",
]

from flamme.plot.cdf import plot_cdf
from flamme.plot.continuous import hist_continuous, hist_continuous2
from flamme.plot.discrete import bar_discrete, bar_discrete_temporal
from flamme.plot.null_temp import plot_null_temporal
