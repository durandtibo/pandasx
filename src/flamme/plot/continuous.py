r"""Contain plotting functions to analyze continuous values."""

from __future__ import annotations

__all__ = ["hist_continuous"]

from typing import TYPE_CHECKING

import numpy as np

from flamme.plot.cdf import plot_cdf
from flamme.plot.utils import (
    auto_yscale_continuous,
    axvline_quantile,
    readable_xticklabels,
)
from flamme.utils.range import find_range

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def hist_continuous(
    ax: Axes,
    array: np.ndarray,
    nbins: int | None = None,
    yscale: str = "linear",
    xmin: float | str | None = None,
    xmax: float | str | None = None,
) -> None:
    r"""Plot the histogram of an array containing continuous values.

    Args:
        ax: The axes of the matplotlib figure to update.
        array: The array with the data.
        nbins: The number of bins to use to plot.
        yscale: The y-axis scale. If ``'auto'``, the
            ``'linear'`` or ``'log'/'symlog'`` scale is chosen based
            on the distribution.
        xmin: The minimum value of the range or its
            associated quantile. ``q0.1`` means the 10% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        xmax: The maximum value of the range or its
            associated quantile. ``q0.9`` means the 90% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from flamme.plot import hist_continuous
    >>> fig, ax = plt.subplots()
    >>> hist_continuous(ax, array=np.arange(101))

    ```
    """
    xmin, xmax = find_range(array, xmin=xmin, xmax=xmax)
    ax.hist(array, bins=nbins, range=(xmin, xmax), color="tab:blue", alpha=0.9)
    readable_xticklabels(ax, max_num_xticks=100)
    if xmin < xmax:
        ax.set_xlim(xmin, xmax)
    ax.set_ylabel("number of occurrences")
    if yscale == "auto":
        yscale = auto_yscale_continuous(array=array, nbins=nbins)
    ax.set_yscale(yscale)
    q05, q95 = np.quantile(array, q=[0.05, 0.95])
    if xmin < q05 < xmax:
        axvline_quantile(ax, quantile=q05, label="q0.05 ", horizontalalignment="right")
    if xmin < q95 < xmax:
        axvline_quantile(ax, quantile=q95, label=" q0.95", horizontalalignment="left")
    plot_cdf(ax, array=array, nbins=nbins, xmin=xmin, xmax=xmax)
