r"""Contain the implementation of sections to analyze the number null
values for each column."""

from __future__ import annotations

__all__ = ["ColumnTemporalNullValueSection"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent
from jinja2 import Template
from matplotlib import pyplot as plt
from tqdm import tqdm

from flamme.section.base import BaseSection
from flamme.section.null_temp import plot_temporal_null_total
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)
from flamme.utils.figure import figure2html, readable_xticklabels

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


class ColumnTemporalNullValueSection(BaseSection):
    r"""Implement a section to analyze the temporal distribution of null
    values for all columns.

    Args:
        frame: The DataFrame to analyze.
        columns: The list of columns to analyze. A plot is generated
            for each column.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        ncols: The number of columns.
        figsize: The figure size in inches. The first dimension
            is the width and the second is the height.

    Example usage:

    ```pycon
    >>> import pandas as pd
    >>> import numpy as np
    >>> from flamme.section import ColumnTemporalNullValueSection
    >>> dataframe = pd.DataFrame(
    ...     {
    ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
    ...         "int": np.array([np.nan, 1, 0, 1]),
    ...         "str": np.array(["A", "B", None, np.nan]),
    ...         "datetime": pd.to_datetime(
    ...             ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
    ...         ),
    ...     }
    ... )
    >>> section = ColumnTemporalNullValueSection(
    ...     frame=dataframe, columns=["float", "int", "str"], dt_column="datetime", period="M"
    ... )
    >>> section
    ColumnTemporalNullValueSection(
      (columns): ('float', 'int', 'str')
      (dt_column): datetime
      (period): M
      (ncols): 2
      (figsize): (7, 5)
    )
    >>> section.get_statistics()
    {}

    ```
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str],
        dt_column: str,
        period: str,
        ncols: int = 2,
        figsize: tuple[float, float] = (7, 5),
    ) -> None:
        self._frame = frame
        self._columns = tuple(columns)
        self._dt_column = dt_column
        self._period = period
        self._ncols = min(ncols, len(self._columns))
        self._figsize = figsize

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "columns": self._columns,
                    "dt_column": self._dt_column,
                    "period": self._period,
                    "ncols": self._ncols,
                    "figsize": self._figsize,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def frame(self) -> pd.DataFrame:
        r"""The DataFrame to analyze."""
        return self._frame

    @property
    def columns(self) -> tuple[str, ...]:
        r"""The columns to analyze."""
        return self._columns

    @property
    def dt_column(self) -> str:
        r"""The datetime column."""
        return self._dt_column

    @property
    def period(self) -> str:
        r"""The temporal period used to analyze the data."""
        return self._period

    @property
    def ncols(self) -> int:
        r"""The number of columns to show the figures."""
        return self._ncols

    @property
    def figsize(self) -> tuple[float, float] | None:
        r"""The individual figure size in pixels.

        The first dimension is the width and the second is the height.
        """
        return self._figsize

    def get_statistics(self) -> dict:
        return {}

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        logger.info(
            f"Rendering the temporal null value distribution of the following columns: "
            f"{self._columns}\ndatetime column: {self._dt_column} | period: {self._period}"
        )
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "column": self._dt_column,
                "figure": create_temporal_null_figure(
                    frame=self._frame,
                    columns=self._columns,
                    dt_column=self._dt_column,
                    period=self._period,
                    ncols=self._ncols,
                    figsize=self._figsize,
                ),
                "table": create_table_section(
                    frame=self._frame,
                    columns=self._columns,
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
This section analyzes the temporal distribution of null values.
The column <em>{{column}}</em> is used as the temporal column.

{{figure}}

<p style="margin-top: 1rem;">

{{table}}

<p style="margin-top: 1rem;">
"""


def create_temporal_null_figure(
    frame: pd.DataFrame,
    columns: Sequence[str],
    dt_column: str,
    period: str,
    ncols: int = 2,
    figsize: tuple[float, float] = (7, 5),
) -> str:
    r"""Create a HTML representation of a figure with the temporal null
    value distribution.

    Args:
        frame: The DataFrame to analyze.
        columns: The list of columns to analyze. A plot is generated
            for each column.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        ncols: The number of columns.
        figsize: The figure size in inches. The first dimension
            is the width and the second is the height.

    Returns:
        The HTML representation of the figure.
    """
    if frame.shape[0] == 0:
        return ""
    figures = create_temporal_null_figures(
        frame=frame, columns=columns, dt_column=dt_column, period=period, figsize=figsize
    )
    figures = add_column_to_figure(columns=columns, figures=figures)
    return Template(
        """
    <div class="container-fluid text-center">
      <div class="row align-items-start">
        {{columns}}
      </div>
    </div>
    """
    ).render({"columns": "\n".join(split_figures_by_column(figures=figures, ncols=ncols))})


def create_temporal_null_figures(
    frame: pd.DataFrame,
    columns: Sequence[str],
    dt_column: str,
    period: str,
    figsize: tuple[float, float] = (7, 5),
) -> list[str]:
    r"""Create a HTML representation of each figure with the temporal
    null value distribution.

    Args:
        frame: The DataFrame to analyze.
        columns: The list of columns to analyze. A plot is generated
            for each column.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        figsize: The figure size in inches. The first dimension
            is the width and the second is the height.

    Returns:
        The HTML representations of the figures.
    """
    if frame.shape[0] == 0:
        return []

    figures = []
    for column in tqdm(columns, desc="generating figures"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"column: {column}")

        num_nulls, total, labels = prepare_data(
            frame=frame, column=column, dt_column=dt_column, period=period
        )
        plot_temporal_null_total(ax=ax, labels=labels, num_nulls=num_nulls, total=total)
        readable_xticklabels(ax, max_num_xticks=50)
        figures.append(figure2html(fig, close_fig=True))

    return figures


def add_column_to_figure(columns: Sequence[str], figures: Sequence[str]) -> list[str]:
    r"""Add the column name to the HTML representation of the figure.

    Args:
        columns: The column names.
        figures: The HTML representations of each figure.

    Returns:
        The updated HTML representations of each figure.

    Raises:
        RuntimeError: if the number of column names is different from
            the number of figures.
    """
    if len(columns) != len(figures):
        msg = (
            f"The number of column names is different from the number of figures: "
            f"{len(columns):,} vs{len(figures):,}"
        )
        raise RuntimeError(msg)
    outputs = []
    for col, figure in zip(columns, figures):
        outputs.append(f'<div style="text-align:center">{col}\n{figure}</div>')
    return outputs


def split_figures_by_column(figures: Sequence[str], ncols: int) -> list[str]:
    r"""Split the figures into multiple columns.

    Args:
        figures: The HTML representations of each figure.
        ncols: The number of columns.

    Returns:
        The columns.
    """
    cols = []
    for i in range(ncols):
        figs = str_indent("\n<hr>\n".join(figures[i::ncols]))
        cols.append(f'<div class="col">\n  {figs}\n</div>')
    return cols


def prepare_data(
    frame: pd.DataFrame,
    column: str,
    dt_column: str,
    period: str,
) -> tuple[np.ndarray, np.ndarray, list]:
    r"""Prepare the data to create the figure and table.

    Args:
        frame: The DataFrame to analyze.
        column: The column to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or
            daily.

    Returns:
        A tuple with 3 values. The first value is a numpy NDArray
            that contains the number of null values per period. The
            second value is a numpy NDArray that contains the total
            number of values. The third value is a list that contains
            the label of each period.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.section.null_temp_col import prepare_data
    >>> num_nulls, total, labels = prepare_data(
    ...     frame=pd.DataFrame(
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

    ```
    """
    dataframe = frame[[column, dt_column]].copy()
    dt_col = "__datetime__"
    dataframe[dt_col] = dataframe[dt_column].dt.to_period(period)

    null_col = f"__{column}_isna__"
    dataframe.loc[:, null_col] = dataframe.loc[:, column].isna()

    num_nulls = dataframe.groupby(dt_col)[null_col].sum().sort_index()
    total = dataframe.groupby(dt_col)[null_col].count().sort_index()
    labels = [str(dt) for dt in num_nulls.index]
    return num_nulls.to_numpy().astype(int), total.to_numpy().astype(int), labels


def create_table_section(
    frame: pd.DataFrame, columns: Sequence[str], dt_column: str, period: str
) -> str:
    r"""Create a HTML representation of a table with the temporal
    distribution of null values.

    Args:
        frame: The DataFrame to analyze.
        columns: The list of columns to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or
            daily.

    Returns:
        The HTML representation of the table.
    """
    if frame.shape[0] == 0:
        return ""
    tables = []
    for column in columns:
        table = create_temporal_null_table(
            frame=frame, column=column, dt_column=dt_column, period=period
        )
        tables.append(f'<p style="margin-top: 1rem;">\n\n{table}\n')
    return Template(
        """
<details>
    <summary>[show statistics per temporal period]</summary>

    <p style="margin-top: 1rem;">
    The following table shows some statistics for each period of column {{column}}.

    {{tables}}
</details>
"""
    ).render({"tables": "\n".join(tables)})


def create_temporal_null_table(
    frame: pd.DataFrame, column: str, dt_column: str, period: str
) -> str:
    r"""Create a HTML representation of a table with the temporal
    distribution of null values.

    Args:
        frame: The DataFrame to analyze.
        column: The column to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or
            daily.

    Returns:
        The HTML representation of the table.
    """
    if frame.shape[0] == 0:
        return ""
    num_nulls, totals, labels = prepare_data(
        frame=frame, column=column, dt_column=dt_column, period=period
    )
    rows = []
    for label, num_null, total in zip(labels, num_nulls, totals):
        rows.append(create_temporal_null_table_row(label=label, num_nulls=num_null, total=total))
    return Template(
        """
<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th colspan="6" style="text-align: center">column: {{column}}</th>
        </tr>
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
"""
    ).render({"rows": "\n".join(rows), "column": column, "period": period})


def create_temporal_null_table_row(label: str, num_nulls: int, total: int) -> str:
    r"""Create the HTML code of a new table row.

    Args:
        label: The label of the row.
        num_nulls: The number of null values.
        total: The total number of values.

    Returns:
        The HTML code of a row.
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
