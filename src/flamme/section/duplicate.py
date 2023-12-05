from __future__ import annotations

__all__ = ["DuplicatedRowSection"]

import logging
from collections.abc import Sequence

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

logger = logging.getLogger(__name__)


class DuplicatedRowSection(BaseSection):
    r"""Implements a section to analyze the number of duplicated rows.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.
        columns (``Sequence`` or ``None``): Specifies the columns used
            to compute the duplicated rows. ``None`` means all the
            columns. Default: ``None``
    """

    def __init__(self, df: DataFrame, columns: Sequence[str] | None = None) -> None:
        self._df = df
        self._columns = columns if columns is None else tuple(columns)

    @property
    def df(self) -> DataFrame:
        r"""``pandas.DataFrame``: The DataFrame to analyze."""
        return self._df

    @property
    def columns(self) -> tuple[str, ...] | None:
        r"""Tuple or ``None``: The columns used to compute the
        duplicated rows."""
        return self._columns

    def get_statistics(self) -> dict:
        df_no_duplicate = self._df.drop_duplicates(subset=self._columns)
        return {"num_rows": self._df.shape[0], "num_unique_rows": df_no_duplicate.shape[0]}

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        logger.info(f"Rendering the duplicated rows section using the columns: {self._columns}")
        stats = self.get_statistics()
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "columns": self._columns,
                "table": create_duplicate_table(
                    num_rows=stats["num_rows"], num_unique_rows=stats["num_unique_rows"]
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
This section shows the number of duplicated rows using the following columns <em>{{columns}}</em>.

{{table}}
<p style="margin-top: 1rem;">
"""


def create_duplicate_table(num_rows: int, num_unique_rows: int) -> str:
    r"""Creates a table with information about duplicated rows.

    Args:
    ----
        num_rows (int): Specifies the number of rows.
        num_unique_rows (int): Specifies the number of unique rows.

    Returns:
    -------
        str: The HTML representation of the table.
    """
    num_duplicated_rows = num_rows - num_unique_rows
    pct_unique_rows = num_unique_rows / num_rows if num_rows else float("nan")
    pct_duplicated_rows = 1.0 - pct_unique_rows
    return Template(
        """
<table class="table table-hover table-responsive w-auto" >
<thead class="thead table-group-divider">
    <tr>
        <th>number of rows</th>
        <th>number of unique rows</th>
        <th>number of duplicated rows</th>
    </tr>
</thead>
<tbody class="tbody table-group-divider">
    <tr>
        <td {{num_style}}>{{num_rows}}</td>
        <td {{num_style}}>{{num_unique_rows}} ({{pct_unique_rows}}%)</td>
        <td {{num_style}}>{{num_duplicated_rows}} ({{pct_duplicated_rows}}%)</td>
    </tr>
    <tr class="table-group-divider"></tr>
</tbody>
</table>
"""
    ).render(
        {
            "num_style": 'style="text-align: right;"',
            "num_rows": f"{num_rows:,}",
            "num_unique_rows": f"{num_unique_rows:,}",
            "num_duplicated_rows": f"{num_duplicated_rows:,}",
            "pct_unique_rows": f"{100 * pct_unique_rows:.2f}",
            "pct_duplicated_rows": f"{100 * pct_duplicated_rows:.2f}",
        }
    )
