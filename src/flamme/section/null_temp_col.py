from __future__ import annotations

__all__ = ["ColumnTemporalNullValueSection"]

import logging
from collections.abc import Sequence

import numpy as np
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
                "table": create_temporal_null_table(
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
This section analyzes the temporal distribution of null values in column <em>{{column}}</em>.
The column <em>{{dt_column}}</em> is used as temporal column.

{{figure}}

{{table}}
<p style="margin-top: 1rem;">
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

    num_nulls, total, labels = prepare_data(
        df=df, column=column, dt_column=dt_column, period=period
    )

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=labels,
            y=total,
            marker=dict(color="rgba(0, 191, 255, 0.9)"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=num_nulls,
            marker=dict(color="rgba(255, 191, 0, 0.9)"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=num_nulls / total,
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


def create_temporal_null_table(df: DataFrame, column: str, dt_column: str, period: str) -> str:
    r"""Creates a HTML representation of a table with the temporal
    distribution of null values.

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
        str: The HTML representation of the table.
    """
    if df.shape[0] == 0:
        return ""
    num_nulls, totals, labels = prepare_data(
        df=df, column=column, dt_column=dt_column, period=period
    )
    rows = []
    for label, num_null, total in zip(labels, num_nulls, totals):
        rows.append(create_temporal_null_table_row(label=label, num_nulls=num_null, total=total))
    return Template(
        """
<details>
    <summary>Statistics per period</summary>

    <p>The following table shows some statistics for each period of column {{column}}.

    <table class="table table-hover table-responsive w-auto" >
        <thead class="thead table-group-divider">
            <tr>
                <th>period</th>
                <th>number of null values</th>
                <th>number of non-null values</th>
                <th>total number of values</th>
                <th>percentage of null values</th>
                <th>percentage of non-null values</th>
            </tr>
        </thead>
        <tbody class="tbody table-group-divider">
            {{rows}}
            <tr class="table-group-divider"></tr>
        </tbody>
    </table>
</details>
"""
    ).render({"rows": "\n".join(rows), "column": column, "period": period})


def create_temporal_null_table_row(label: str, num_nulls: int, total: int) -> str:
    r"""Creates the HTML code of a new table row.

    Args:
    ----
        row ("pd.core.frame.Pandas"): Specifies a DataFrame row.

    Returns:
    -------
        str: The HTML code of a row.
    """
    num_non_nulls = total - num_nulls
    return Template(
        """<tr>
    <th>{{label}}</th>
    <td {{num_style}}>{{num_nulls}}</td>
    <td {{num_style}}>{{num_non_nulls}}</td>
    <td {{num_style}}>{{total}}</td>
    <td {{num_style}}>{{num_nulls_pct}}</td>
    <td {{num_style}}>{{num_non_nulls_pct}}</td>
</tr>"""
    ).render(
        {
            "num_style": 'style="text-align: right;"',
            "label": label,
            "num_nulls": f"{num_nulls:,}",
            "num_non_nulls": f"{num_non_nulls:,}",
            "total": f"{total:,}",
            "num_nulls_pct": f"{100 * num_nulls / total:.2f}%",
            "num_non_nulls_pct": f"{100 * num_non_nulls / total:.2f}%",
        }
    )


def prepare_data(
    df: DataFrame,
    column: str,
    dt_column: str,
    period: str,
) -> tuple[np.ndarray, np.ndarray, list]:
    r"""Prepares the data to create the figure and table.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.
        column (str): Specifies the column to analyze.
        dt_column (str): Specifies the datetime column used to analyze
            the temporal distribution.
        period (str): Specifies the temporal period e.g. monthly or
            daily.

    Returns:
    -------
        tuple: A tuple with 3 values. The first value is a numpy NDArray
            that contains the number of null values per period. The
            second value is a numpy NDArray that contains the total
            number of values. The third value is a list that contains
            the label of each period.

    Example usage:

    .. code-block:: pycon

        >>> import pandas as pd
        >>> from flamme.section.null_temporal import prepare_data
        >>> num_nulls, total, labels = prepare_data(
        ...     df=pd.DataFrame(
        ...         {
        ...             "col": np.array([np.nan, 1, 0, 1]),
        ...             "datetime": pd.to_datetime(
        ...                 ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
        ...             ),
        ...         }
        ...     ),
        ...     column="col",
        ...     dt_column="datetime",
        ...     period="M",
        ... )
        >>> num_nulls
        array([1, 0, 0, 0])
        >>> total
        array([1, 1, 1, 1])
        >>> labels
        ['2020-01', '2020-02', '2020-03', '2020-04']
    """
    df = df[[column, dt_column]].copy()
    dt_col = "__datetime__"
    df[dt_col] = df[dt_column].dt.to_period(period)

    null_col = f"__{column}_isnull__"
    df.loc[:, null_col] = df.loc[:, column].isnull()

    df_num_nulls = df.groupby(dt_col)[null_col].sum().sort_index()
    df_total = df.groupby(dt_col)[null_col].count().sort_index()
    labels = [str(dt) for dt in df_num_nulls.index]
    return df_num_nulls.to_numpy().astype(int), df_total.to_numpy().astype(int), labels
