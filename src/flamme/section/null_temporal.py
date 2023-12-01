from __future__ import annotations

__all__ = ["ColumnTemporalNullValueSection"]

import logging
from collections.abc import Sequence

import plotly
import plotly.graph_objects as go
from jinja2 import Template
from pandas import DataFrame
from plotly.subplots import make_subplots

from flamme.section.base import BaseSection
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)

logger = logging.getLogger(__name__)


class ColumnTemporalNullValueSection(BaseSection):
    r"""Implements a section to analyze the temporal distribution of null
    values for a given column.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.
        column (str): Specifies the column to analyze.
        dt_column (str): Specifies the datetime column used to analyze
            the temporal distribution.
        period (str): Specifies the temporal period e.g. monthly or
            daily.
        figsize (``tuple`` or list , optional): Specifies the figure
            size in pixels. The first dimension is the width and the
            second is the height. Default: ``(None, None)``
    """

    def __init__(
        self,
        df: DataFrame,
        column: str,
        dt_column: str,
        period: str,
        figsize: tuple[int | None, int | None] | list[int | None] = (None, None),
    ) -> None:
        if column not in df:
            raise ValueError(
                f"Column {column} is not in the DataFrame (columns:{sorted(df.columns)})"
            )
        if dt_column not in df:
            raise ValueError(
                f"Datetime column {dt_column} is not in the DataFrame (columns:{sorted(df.columns)})"
            )

        self._df = df
        self._column = column
        self._dt_column = dt_column
        self._period = period
        self._figsize = figsize

    @property
    def df(self) -> DataFrame:
        r"""``pandas.DataFrame``: The DataFrame to analyze."""
        return self._df

    @property
    def column(self) -> str:
        r"""str: The column to analyze."""
        return self._column

    @property
    def dt_column(self) -> str:
        r"""str: The datetime column."""
        return self._dt_column

    @property
    def period(self) -> str:
        r"""str: The temporal period used to analyze the data."""
        return self._period

    @property
    def figsize(self) -> tuple[int, int]:
        r"""tuple: The individual figure size in pixels. The first
        dimension is the width and the second is the height."""
        return self._figsize

    def get_statistics(self) -> dict:
        return {}

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        logger.info(
            f"Rendering the temporal distribution of null values for column {self._column} "
            f"| datetime column: {self._dt_column} | period: {self._period}"
        )
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "column": self._column,
                "dt_column": self._dt_column,
                "figure": create_temporal_null_figure(
                    df=self._df,
                    column=self._column,
                    dt_column=self._dt_column,
                    period=self._period,
                    figsize=self._figsize,
                ),
                "table": "",
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
This section analyzes the temporal distribution of null values in column <em>{{column}}</em>.
The column <em>{{dt_column}}</em> is used as temporal column.

{{figure}}
"""


def create_temporal_null_figure(
    df: DataFrame,
    column: str,
    dt_column: str,
    period: str,
    figsize: tuple[int | None, int | None] | list[int | None] = (None, None),
) -> str:
    r"""Creates a HTML representation of a figure with the temporal null
    value distribution.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.
        column (str): Specifies the column to analyze.
        dt_column (str): Specifies the datetime column used to analyze
            the temporal distribution.
        period (str): Specifies the temporal period e.g. monthly or
            daily.
        figsize (``tuple`` or list , optional): Specifies the figure
            size in pixels. The first dimension is the width and the
            second is the height. Default: ``(None, None)``

    Returns:
    -------
        str: The HTML representation of the figure.
    """
    if df.shape[0] == 0:
        return ""
    df = df[[column, dt_column]].copy()
    dt_col = "__datetime__"
    df[dt_col] = df[dt_column].dt.to_period(period)

    null_col = f"__{column}_isnull__"
    df.loc[:, null_col] = df.loc[:, column].isnull()

    df_sum = df.groupby(dt_col)[null_col].sum().sort_index()
    df_count = df.groupby(dt_col)[null_col].count().sort_index()
    labels = [str(dt) for dt in df_sum.index]

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df_count.to_numpy(),
            marker=dict(color="rgba(0, 191, 255, 0.9)"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df_sum.to_numpy(),
            marker=dict(color="rgba(255, 191, 0, 0.9)"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=df_sum.to_numpy() / df_count.to_numpy(),
            marker=dict(color="rgba(0, 71, 171, 0.9)"),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text=(
            '<span style="color:RGB(255, 191, 0)">null</span>/'
            '<span style="color:RGB(0, 191, 255)">total</span>'
        ),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text='<span style="color:RGB(0, 71, 171)">percentage</span>',
        secondary_y=True,
    )
    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        showlegend=False,
        barmode="overlay",
    )
    return plotly.io.to_html(fig, full_html=False)
