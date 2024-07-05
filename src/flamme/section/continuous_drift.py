r"""Contain the implementation of a section to analyze the temporal
drift of a column with continuous values."""

from __future__ import annotations

__all__ = ["create_temporal_drift_figure"]

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

from flamme.plot import hist_continuous2
from flamme.plot.utils import readable_xticklabels
from flamme.utils.figure import figure2html
from flamme.utils.mapping import sort_by_keys
from flamme.utils.range import find_range

if TYPE_CHECKING:
    import pandas as pd


def create_temporal_drift_figure(
    frame: pd.DataFrame,
    column: str,
    dt_column: str,
    period: str,
    nbins: int | None = None,
    density: bool = False,
    yscale: str = "linear",
    xmin: float | str | None = None,
    xmax: float | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> str:
    r"""Create the HTML code of figures to show the temporal drift.

    Args:
        frame: The DataFrame with the data.
        column: The column name.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        nbins: The number of bins in the histogram.
        density: If True, draw and return a probability density:
            each bin will display the bin's raw count divided by the
            total number of counts and the bin width, so that the area
            under the histogram integrates to 1.
        yscale: The y-axis scale. If ``'auto'``, the
            ``'linear'`` or ``'log'`` scale is chosen based on the
            distribution.
        xmin: The minimum value of the range or its
            associated quantile. ``q0.1`` means the 10% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        xmax: The maximum value of the range or its
            associated quantile. ``q0.9`` means the 90% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        figsize: The figure size in inches. The first
            dimension is the width and the second is the height.

    Returns:
        The HTML code of the figure.
    """
    array = frame[column].dropna().to_numpy()
    if array.size == 0:
        return "<span>&#9888;</span> No figure is generated because the column is empty"

    frame = frame[[column, dt_column]].copy()
    frame[dt_column] = frame[dt_column].dt.to_period(period)
    groups = sort_by_keys(frame.groupby(dt_column).groups)
    groups = {key: frame.loc[index, column].to_numpy() for key, index in groups.items()}

    keys = list(groups.keys())
    nrows = len(keys)
    keys1, keys2 = keys[:-1], keys[1:]
    if len(keys) == 2:
        nrows = 1
    if len(keys) > 2:
        keys1, keys2 = [keys[0], *keys1], [keys[-1], *keys2]

    xmin, xmax = find_range(array, xmin=xmin, xmax=xmax)
    if figsize is not None:
        figsize = (figsize[0], figsize[1] * nrows)
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows)

    for i, (key1, key2) in enumerate(zip(keys1, keys2)):
        ax = axes[i] if nrows > 1 else axes
        hist_continuous2(
            ax=ax,
            array1=groups[key1],
            array2=groups[key2],
            label1=key1,
            label2=key2,
            xmin=xmin,
            xmax=xmax,
            nbins=nbins,
            density=density,
            yscale=yscale,
        )
        ax.set_title(f"{key1} vs {key2}")
        readable_xticklabels(ax, max_num_xticks=100)
    return figure2html(fig, close_fig=True)
