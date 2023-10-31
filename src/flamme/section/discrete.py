from __future__ import annotations

__all__ = ["DiscreteDistributionSection"]

from collections import Counter
from collections.abc import Sequence

from jinja2 import Template

from flamme.section.base import BaseSection
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)


class DiscreteDistributionSection(BaseSection):
    r"""Implements a section that analyzes a discrete distribution of
    values.

    Args:
    ----
        counter (``Counter``): Specifies the counter that represents
            the discrete distribution.
        column (str, optional): Specifies the column name.
            Default: ``'N/A'``
        max_rows (int, optional): Specifies the maximum number of rows
            to show in the table. Default: ``20``
    """

    def __init__(self, counter: Counter, column: str = "N/A", max_rows: int = 20) -> None:
        self._counter = counter
        self._column = column
        self._max_rows = int(max_rows)

    def get_statistics(self) -> dict:
        most_common = [(col, count) for col, count in self._counter.most_common() if count > 0]
        return {
            "most_common": most_common,
            "nunique": len(most_common),
            "total": sum(self._counter.values()),
        }

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        stats = self.get_statistics()
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "column": self._column,
                "total_values": f"{stats['total']:,}",
                "unique_values": f"{stats['nunique']:,}",
                "table": self._create_table(),
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
  <li> unique values: {{unique_values}} </li>
</ul>

{{table}}
"""

    def _create_table(self) -> str:
        rows = "\n".join(
            [
                create_table_row(column=col, count=count)
                for col, count in self._counter.most_common(self._max_rows)
            ]
        )
        return Template(
            """<p style="margin-top: 1rem;">
<b>Most common values in column {{column}}</b>

<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th>column</th>
            <th>count</th>
        </tr>
    </thead>
    <tbody class="tbody table-group-divider">
        {{rows}}
        <tr class="table-group-divider"></tr>
    </tbody>
</table>
"""
        ).render({"rows": rows, "column": self._column})


def create_table_row(column: str, count: int) -> str:
    r"""Creates the HTML code of a new table row.

    Args:
    ----
        column (str): Specifies the column name.
        count (int): Specifies the count for the column.

    Returns:
    -------
        str: The HTML code of a row.
    """
    return Template("""<tr><th>{{column}}</th><td {{num_style}}>{{count}}</td></tr>""").render(
        {"num_style": 'style="text-align: right;"', "column": column, "count": f"{count:,}"}
    )
