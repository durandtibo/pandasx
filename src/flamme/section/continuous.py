from __future__ import annotations

__all__ = ["ColumnContinuousSection"]

import logging
from collections.abc import Sequence

import numpy as np
from jinja2 import Template
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pandas import Series
from scipy.stats import kurtosis, skew

from flamme.section.base import BaseSection
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)
from flamme.utils.array import nonnan
from flamme.utils.figure import figure2html, readable_xticklabels
from flamme.utils.range import find_range

logger = logging.getLogger(__name__)


class ColumnContinuousSection(BaseSection):
    r"""Implements a section that analyzes a continuous distribution of
    values.

    Args:
        series: Specifies the series/column to analyze.
        column: Specifies the column name.
        nbins: Specifies the number of bins in the histogram.
        yscale: Specifies the y-axis scale. If ``'auto'``, the
            ``'linear'`` or ``'log'`` scale is chosen based on the
            distribution.
        xmin: Specifies the minimum value of the range or its
            associated quantile. ``q0.1`` means the 10% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        xmax: Specifies the maximum value of the range or its
            associated quantile. ``q0.9`` means the 90% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        figsize: Specifies the figure size in inches. The first
            dimension is the width and the second is the height.
    """

    def __init__(
        self,
        series: Series,
        column: str,
        nbins: int | None = None,
        yscale: str = "auto",
        xmin: float | str | None = None,
        xmax: float | str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self._series = series
        self._column = column
        self._nbins = nbins
        self._yscale = yscale
        self._xmin = xmin
        self._xmax = xmax
        self._figsize = figsize

    @property
    def column(self) -> str:
        return self._column

    @property
    def yscale(self) -> str:
        return self._yscale

    @property
    def nbins(self) -> int | None:
        return self._nbins

    @property
    def series(self) -> Series:
        return self._series

    @property
    def xmin(self) -> float | str | None:
        return self._xmin

    @property
    def xmax(self) -> float | str | None:
        return self._xmax

    @property
    def figsize(self) -> tuple[float, float] | None:
        r"""tuple: The individual figure size in pixels. The first
        dimension is the width and the second is the height."""
        return self._figsize

    def get_statistics(self) -> dict[str, float | int]:
        stats = {
            "count": int(self._series.shape[0]),
            "num_nulls": int(self._series.isnull().sum()),
            "nunique": self._series.nunique(dropna=False),
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "q001": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "q999": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
        }
        stats["num_non_nulls"] = stats["count"] - stats["num_nulls"]
        if stats["num_non_nulls"] > 0:
            stats |= (
                self._series.dropna()
                .astype(float)
                .agg(
                    {
                        "mean": "mean",
                        "median": "median",
                        "min": "min",
                        "max": "max",
                        "std": "std",
                        "q001": lambda x: x.quantile(0.001),
                        "q01": lambda x: x.quantile(0.01),
                        "q05": lambda x: x.quantile(0.05),
                        "q10": lambda x: x.quantile(0.1),
                        "q25": lambda x: x.quantile(0.25),
                        "q75": lambda x: x.quantile(0.75),
                        "q90": lambda x: x.quantile(0.9),
                        "q95": lambda x: x.quantile(0.95),
                        "q99": lambda x: x.quantile(0.99),
                        "q999": lambda x: x.quantile(0.999),
                    }
                )
                .to_dict()
            )
            if stats["nunique"] > 1:
                stats["skewness"] = float(
                    skew(self._series.to_numpy(dtype=float), nan_policy="omit")
                )
                stats["kurtosis"] = float(
                    kurtosis(self._series.to_numpy(dtype=float), nan_policy="omit")
                )
        return stats

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        logger.info(f"Rendering the continuous distribution of {self._column}")
        stats = self.get_statistics()
        null_values_pct = (
            f"{100 * stats['num_nulls'] / stats['count']:.2f}" if stats["count"] > 0 else "N/A"
        )
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "column": self._column,
                "table": create_stats_table(stats=stats, column=self._column),
                "total_values": f"{stats['count']:,}",
                "unique_values": f"{stats['nunique']:,}",
                "null_values": f"{stats['num_nulls']:,}",
                "null_values_pct": null_values_pct,
                "histogram_figure": create_histogram_figure(
                    series=self._series,
                    column=self._column,
                    stats=stats,
                    nbins=self._nbins,
                    yscale=self._yscale,
                    xmin=self._xmin,
                    xmax=self._xmax,
                    figsize=self._figsize,
                ),
                "boxplot_figure": create_boxplot_figure(
                    series=self._series,
                    xmin=self._xmin,
                    xmax=self._xmax,
                    figsize=self._figsize,
                ),
            }
        )

    def render_html_toc(
        self, number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
    ) -> str:
        return render_html_toc(number=number, tags=tags, depth=depth, max_depth=max_depth)

    def _create_template(self) -> str:
        return """
<h{{depth}} id="{{id}}">{{section}} {{title}} </h{{depth}}>

{{go_to_top}}

<p style="margin-top: 1rem;">
This section analyzes the discrete distribution of values for column <em>{{column}}</em>.

<ul>
  <li> total values: {{total_values}} </li>
  <li> number of unique values: {{unique_values}} </li>
  <li> number of null values: {{null_values}} / {{total_values}} ({{null_values_pct}}%) </li>
</ul>

{{histogram_figure}}
{{boxplot_figure}}
{{table}}
<p style="margin-top: 1rem;">
"""


def create_boxplot_figure(
    series: Series,
    xmin: float | str | None = None,
    xmax: float | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> str:
    r"""Creates the HTML code of a boxplot figure.

    Args:
        series: Specifies the series/column to analyze.
        xmin: Specifies the minimum value of the range or its
            associated quantile. ``q0.1`` means the 10% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        xmax: Specifies the maximum value of the range or its
            associated quantile. ``q0.9`` means the 90% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        figsize: Specifies the figure size in inches. The first
            dimension is the width and the second is the height.

    Returns:
        str: The HTML code of the figure.
    """
    array = series.dropna().to_numpy()
    if array.size == 0:
        return "<span>&#9888;</span> No figure is generated because the column is empty"
    xmin, xmax = find_range(array, xmin=xmin, xmax=xmax)
    if figsize is not None:
        figsize = (figsize[0], figsize[0] / 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        array,
        notch=True,
        vert=False,
        widths=0.7,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    readable_xticklabels(ax, max_num_xticks=100)
    if xmin < xmax:
        ax.set_xlim(xmin, xmax)
    ax.set_ylabel(" ")
    return figure2html(fig, close_fig=True)


def create_histogram_figure(
    series: Series,
    column: str,
    stats: dict,
    nbins: int | None = None,
    yscale: str = "linear",
    xmin: float | str | None = None,
    xmax: float | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> str:
    r"""Creates the HTML code of a histogram figure.

    Args:
        series: Specifies the series/column to analyze.
        column: Specifies the column name.
        stats: Specifies a dictionary with the statistics.
        nbins: Specifies the number of bins in the histogram.
        yscale: Specifies the y-axis scale. If ``'auto'``, the
            ``'linear'`` or ``'log'`` scale is chosen based on the
            distribution.
        xmin: Specifies the minimum value of the range or its
            associated quantile. ``q0.1`` means the 10% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        xmax: Specifies the maximum value of the range or its
            associated quantile. ``q0.9`` means the 90% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        figsize: Specifies the figure size in inches. The first
            dimension is the width and the second is the height.

    Returns:
        str: The HTML code of the figure.
    """
    array = series.to_numpy()
    if array.size == 0:
        return "<span>&#9888;</span> No figure is generated because the column is empty"
    xmin, xmax = find_range(array, xmin=xmin, xmax=xmax)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        array,
        bins=nbins,
        range=[xmin, xmax],
        color="tab:blue",
    )
    readable_xticklabels(ax, max_num_xticks=100)
    if xmin < xmax:
        ax.set_xlim(xmin, xmax)
    ax.set_title(f"Distribution of values for column {column}")
    ax.set_ylabel("Number of occurrences")
    if yscale == "auto":
        yscale = auto_yscale(array=array, nbins=nbins)
    ax.set_yscale(yscale)
    if stats["q05"] > xmin:
        ax.axvline(stats["q05"], color="black", linestyle="dashed")
        ax.text(
            stats["q05"],
            0.99,
            "q0.05 ",
            transform=ax.get_xaxis_transform(),
            color="black",
            horizontalalignment="right",
            verticalalignment="top",
        )
    if stats["q95"] < xmax:
        ax.axvline(stats["q95"], color="black", linestyle="dashed")
        ax.text(
            stats["q95"],
            0.99,
            " q0.95",
            transform=ax.get_xaxis_transform(),
            color="black",
            horizontalalignment="left",
            verticalalignment="top",
        )
    ax.legend(
        [Line2D([0], [0], linestyle="none", mfc="black", mec="none", marker="")] * 3,
        [
            f'std={stats["std"]:.2f}',
            f'skewness={stats["skewness"]:.2f}',
            f'kurtosis={stats["kurtosis"]:.2f}',
        ],
    )
    return figure2html(fig, close_fig=True)


def create_stats_table(stats: dict, column: str) -> str:
    r"""Creates the HTML code of the table with statistics.

    Args:
        stats: Specifies a dictionary with the statistics.
        column: Specifies the column name.

    Returns:
        The HTML code of the table.
    """
    return Template(
        """
<details>
    <summary>Statistics</summary>

    <p>The following table shows some statistics about the distribution for column {{column}}.

    <table class="table table-hover table-responsive w-auto" >
        <thead class="thead table-group-divider">
            <tr><th>stat</th><th>value</th></tr>
        </thead>
        <tbody class="tbody table-group-divider">
            <tr><th>count</th><td {{num_style}}>{{count}}</td></tr>
            <tr><th>mean</th><td {{num_style}}>{{mean}}</td></tr>
            <tr><th>std</th><td {{num_style}}>{{std}}</td></tr>
            <tr><th>skewness</th><td {{num_style}}>{{skewness}}</td></tr>
            <tr><th>kurtosis</th><td {{num_style}}>{{kurtosis}}</td></tr>
            <tr><th>min</th><td {{num_style}}>{{min}}</td></tr>
            <tr><th>quantile 0.1%</th><td {{num_style}}>{{q01}}</td></tr>
            <tr><th>quantile 1%</th><td {{num_style}}>{{q01}}</td></tr>
            <tr><th>quantile 5%</th><td {{num_style}}>{{q05}}</td></tr>
            <tr><th>quantile 10%</th><td {{num_style}}>{{q10}}</td></tr>
            <tr><th>quantile 25%</th><td {{num_style}}>{{q25}}</td></tr>
            <tr><th>median</th><td {{num_style}}>{{median}}</td></tr>
            <tr><th>quantile 75%</th><td {{num_style}}>{{q75}}</td></tr>
            <tr><th>quantile 90%</th><td {{num_style}}>{{q90}}</td></tr>
            <tr><th>quantile 95%</th><td {{num_style}}>{{q95}}</td></tr>
            <tr><th>quantile 99%</th><td {{num_style}}>{{q99}}</td></tr>
            <tr><th>quantile 99.9%</th><td {{num_style}}>{{q99}}</td></tr>
            <tr><th>max</th><td {{num_style}}>{{max}}</td></tr>
            <tr class="table-group-divider"></tr>
        </tbody>
    </table>
</details>
"""
    ).render(
        {
            "column": column,
            "num_style": 'style="text-align: right;"',
            "count": f"{stats['count']:,}",
            "mean": f"{stats['mean']:,.4f}",
            "std": f"{stats['std']:,.4f}",
            "skewness": f"{stats['skewness']:,.4f}",
            "kurtosis": f"{stats['kurtosis']:,.4f}",
            "min": f"{stats['min']:,.4f}",
            "q001": f"{stats['q001']:,.4f}",
            "q01": f"{stats['q01']:,.4f}",
            "q05": f"{stats['q05']:,.4f}",
            "q10": f"{stats['q10']:,.4f}",
            "q25": f"{stats['q25']:,.4f}",
            "median": f"{stats['median']:,.4f}",
            "q75": f"{stats['q75']:,.4f}",
            "q90": f"{stats['q90']:,.4f}",
            "q95": f"{stats['q95']:,.4f}",
            "q99": f"{stats['q99']:,.4f}",
            "q999": f"{stats['q999']:,.4f}",
            "max": f"{stats['max']:,.4f}",
        }
    )


def auto_yscale(array: np.ndarray, nbins: int | None) -> str:
    r"""Finds a good scale for y-axis based on the data.

    Args:
        array: Specifies the data to use to find the scale.
        nbins: Specifies the number of bins in the histogram.

    Returns:
        The scale for y-axis
    """
    if nbins is None:
        nbins = 100
    array = nonnan(array)
    counts = np.histogram(array, bins=nbins)[0]
    min_nonzero_count = min([c for c in counts if c > 0])
    if (np.max(counts) / max(min_nonzero_count, 1)) < 50:
        return "linear"
    if np.nanmin(array) <= 0.0:
        return "symlog"
    return "log"
