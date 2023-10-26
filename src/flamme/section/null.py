from __future__ import annotations

__all__ = ["NullValueSection"]

from collections.abc import Sequence

import numpy as np
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


class NullValueSection(BaseSection):
    r"""Implements a section that analyzes the number of null values.

    Args:
    ----
        columns (``Sequence``): Specifies the column names.
        null_count (``numpy.ndarray``): Specifies the number of null
            values for each column.
        total_count (``numpy.ndarray``): Specifies the total number
            of values for each column.
    """

    def __init__(
        self, columns: Sequence[str], null_count: np.ndarray, total_count: np.ndarray
    ) -> None:
        self._columns = tuple(columns)
        self._null_count = null_count.flatten().astype(int)
        self._total_count = total_count.flatten().astype(int)

        if len(self._columns) != self._null_count.shape[0]:
            raise RuntimeError(
                f"columns ({len(self._columns):,}) and null_count ({self._null_count.shape[0]:,}) "
                "do not match"
            )
        if len(self._columns) != self._total_count.shape[0]:
            raise RuntimeError(
                f"columns ({len(self._columns):,}) and total_count "
                f"({self._total_count.shape[0]:,}) do not match"
            )

    def get_statistics(self) -> dict:
        return {
            "columns": self._columns,
            "null_count": tuple(self._null_count.tolist()),
            "total_count": tuple(self._total_count.tolist()),
        }

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "table_alpha": self._create_table(sort_by="column"),
                "table_sort": self._create_table(sort_by="null"),
                "bar_figure": self._create_bar_figure(),
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
This section analyzes the number/proportion of null values for each column.
The color indicates the proportion of missing values.
Dark blues indicates more missing values than light blues.

{{bar_figure}}

<div class="container-fluid">
    <div class="row align-items-start">
        <div class="col align-self-center">
            <p><b>Columns sorted by alphabetical order</b></p>

            {{table_alpha}}

        </div>
        <div class="col">
            <p><b>Columns sorted by ascending order of missing values</b></p>

            {{table_sort}}

        </div>
    </div>
</div>
"""

    def _create_bar_figure(self) -> str:
        df = self._get_dataframe().sort_values(by="null")
        fig = px.bar(
            df,
            x="column",
            y="null",
            title="number of null values per column",
            labels={"column": "column", "null": "number of null values"},
            text_auto=True,
            template="seaborn",
        )
        return plotly.io.to_html(fig, full_html=False)

    def _create_table(self, sort_by: str) -> str:
        df = self._get_dataframe().sort_values(by=sort_by)
        rows = "\n".join(
            [
                create_table_row(column=column, null_count=null_count, total_count=total_count)
                for column, null_count, total_count in zip(
                    df["column"].to_numpy(), df["null"].to_numpy(), df["total"].to_numpy()
                )
            ]
        )
        return Template(
            """
<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th>column</th>
            <th>null pct</th>
            <th>null count</th>
            <th>total count</th>
        </tr>
    </thead>
    <tbody class="tbody table-group-divider">
        {{rows}}
        <tr class="table-group-divider"></tr>
    </tbody>
</table>
"""
        ).render({"rows": rows})

    def _get_dataframe(self) -> DataFrame:
        return DataFrame(
            {"column": self._columns, "null": self._null_count, "total": self._total_count}
        )


def create_table_row(column: str, null_count: int, total_count: int) -> str:
    r"""Creates the HTML code of a new table row.

    Args:
    ----
        column (str): Specifies the column name.
        null_count (int): Specifies the number of null values.
        total_count (int): Specifies the total number of rows.

    Returns:
    -------
        str: The HTML code of a row.
    """
    pct = null_count / total_count
    return Template(
        """<tr>
    <th style="background-color: rgba(64, 161, 255, {{null_pct}})">{{column}}</th>
    <td {{num_style}}>{{null_pct}}</td>
    <td {{num_style}}>{{null_count}}</td>
    <td {{num_style}}>{{total_count}}</td>
</tr>"""
    ).render(
        {
            "num_style": f'style="text-align: right; background-color: rgba(64, 161, 255, {pct})"',
            "column": column,
            "null_count": f"{null_count:,}",
            "null_pct": f"{pct:.4f}",
            "total_count": f"{total_count:,}",
        }
    )
