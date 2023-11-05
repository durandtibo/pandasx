from __future__ import annotations

__all__ = ["TemporalContinuousDistributionSection"]

from collections.abc import Sequence

import plotly
import plotly.express as px
from jinja2 import Template
from pandas import DataFrame

from flamme.section.base import BaseSection
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)


class TemporalContinuousDistributionSection(BaseSection):
    r"""Implements a section that analyzes the temporal distribution of a
    column with continuous values.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.
        column (str): Specifies the column of the DataFrame to analyze.
        dt_column (str): Specifies the datetime column used to analyze
            the temporal distribution.
        period (str): Specifies the temporal period e.g. monthly or
            daily.
    """

    def __init__(self, df: DataFrame, column: str, dt_column: str, period: str) -> None:
        self._df = df
        self._column = column
        self._dt_column = dt_column
        self._period = period

    def get_statistics(self) -> dict:
        return {}

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "column": self._column,
                "dt_column": self._dt_column,
                "period": self._period,
                "figure": create_temporal_figure(
                    df=self._df,
                    column=self._column,
                    dt_column=self._dt_column,
                    period=self._period,
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
This section analyzes the temporal distribution of column {{column}} by using the column {{dt_column}}.

{{figure}}
"""


def create_temporal_figure(df: DataFrame, column: str, dt_column: str, period: str) -> str:
    r"""Creates a HTML representation of a figure with the temporal null
    value distribution.

    Args:
    ----
        df (``DataFrame``): Specifies the DataFrame to analyze.
        column (str): Specifies the column to analyze.
        dt_column (str): Specifies the datetime column used to analyze
            the temporal distribution.
        period (str): Specifies the temporal period e.g. monthly or
            daily.

    Returns:
    -------
        str: The HTML representation of the figure.
    """
    if df.shape[0] == 0:
        return ""
    df = df[[column, dt_column]].copy()
    dt_col = "__datetime__"
    df[dt_col] = df[dt_column].dt.to_period(period).astype(str)

    fig = px.box(
        df,
        x=dt_col,
        y=column,
        title=f"Distribution of values for column {column}",
        labels={column: "value", dt_col: "time"},
        points="outliers",
    )
    return plotly.io.to_html(fig, full_html=False)
