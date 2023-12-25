from __future__ import annotations

__all__ = ["ColumnContinuousSection"]

import logging
from collections.abc import Sequence

from jinja2 import Template
from matplotlib import pyplot as plt
from pandas import Series

from flamme.section.base import BaseSection
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)
from flamme.utils.figure import figure2html
from flamme.utils.range import find_range

logger = logging.getLogger(__name__)


class ColumnContinuousSection(BaseSection):
    r"""Implements a section that analyzes a continuous distribution of
    values.

    Args:
        series (``pandas.Series``): Specifies the series/column to
            analyze.
        column (str): Specifies the column name.
        nbins (int or None, optional): Specifies the number of bins in
            the histogram. Default: ``None``
        yscale (str, optional): Specifies the y-axis scale.
            Default: ``linear``
        xmin (float or str or None, optional): Specifies the minimum
            value of the range or its associated quantile.
            ``q0.1`` means the 10% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``None``
        xmax (float or str or None, optional): Specifies the maximum
            value of the range or its associated quantile.
            ``q0.9`` means the 90% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``None``
        figsize (``tuple`` or ``None``, optional): Specifies the figure
            size in inches. The first dimension is the width and the
            second is the height. Default: ``None``
    """

    def __init__(
        self,
        series: Series,
        column: str,
        nbins: int | None = None,
        yscale: str = "linear",
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

    def get_statistics(self) -> dict:
        stats = {
            "count": int(self._series.shape[0]),
            "num_nulls": int(self._series.isnull().sum()),
            "nunique": self._series.nunique(dropna=False),
        }
        stats["num_non_nulls"] = stats["count"] - stats["num_nulls"]
        if stats["num_non_nulls"] > 0:
            stats |= (
                self._series.dropna()
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
        else:
            stats |= {
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
            }
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
This section analyzes the discrete distribution of values for column {{column}}.

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
        row (``pandas.Series``): Specifies the series of data.
        xmin (float or str or None, optional): Specifies the minimum
            value of the range or its associated quantile.
            ``q0.1`` means the 10% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``None``
        xmax (float or str or None, optional): Specifies the maximum
            value of the range or its associated quantile.
            ``q0.9`` means the 90% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``None``
        figsize (``tuple`` or ``None``, optional): Specifies the figure
            size in inches. The first dimension is the width and the
            second is the height. Default: ``None``

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
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(" ")
    return figure2html(fig, close_fig=True)


def create_histogram_figure(
    series: Series,
    column: str,
    stats: dict | None = None,
    nbins: int | None = None,
    yscale: str = "linear",
    xmin: float | str | None = None,
    xmax: float | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> str:
    r"""Creates the HTML code of a histogram figure.

    Args:
        row (``pandas.Series``): Specifies the series of data.
        column (str): Specifies the column name.
        nbins (int or None, optional): Specifies the number of bins in
            the histogram. Default: ``None``
        yscale (str, optional): Specifies the y-axis scale.
            Default: ``linear``
        xmin (float or str or None, optional): Specifies the minimum
            value of the range or its associated quantile.
            ``q0.1`` means the 10% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``None``
        xmax (float or str or None, optional): Specifies the maximum
            value of the range or its associated quantile.
            ``q0.9`` means the 90% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``None``
        figsize (``tuple`` or ``None``, optional): Specifies the figure
            size in inches. The first dimension is the width and the
            second is the height. Default: ``None``

    Returns:
        str: The HTML code of the figure.
    """
    stats = stats or {}
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
    ax.set_title(f"Distribution of values for column {column}")
    ax.set_ylabel("Number of occurrences")
    ax.set_yscale(yscale)
    ax.set_xlim(xmin, xmax)
    if "q05" in stats and stats["q05"] > xmin:
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
    if "q95" in stats and stats["q95"] < xmax:
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
    return figure2html(fig, close_fig=True)


def create_stats_table(stats: dict, column: str) -> str:
    r"""Creates the HTML code of the table with statistics.

    Args:
        stats (dict): Specifies a dictionary with the statistics.
        column (str): Specifies the column name.

    Returns:
        str: The HTML code of the table.
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
            "median": f"{stats['median']:,.4f}",
            "min": f"{stats['min']:,.4f}",
            "max": f"{stats['max']:,.4f}",
            "std": f"{stats['std']:,.4f}",
            "q001": f"{stats['q001']:,.4f}",
            "q01": f"{stats['q01']:,.4f}",
            "q05": f"{stats['q05']:,.4f}",
            "q10": f"{stats['q10']:,.4f}",
            "q25": f"{stats['q25']:,.4f}",
            "q75": f"{stats['q75']:,.4f}",
            "q90": f"{stats['q90']:,.4f}",
            "q95": f"{stats['q95']:,.4f}",
            "q99": f"{stats['q99']:,.4f}",
            "q999": f"{stats['q999']:,.4f}",
        }
    )
