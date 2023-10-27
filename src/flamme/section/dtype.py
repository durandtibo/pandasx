from __future__ import annotations

__all__ = ["ColumnDtypeSection", "ColumnTypeSection"]

import copy
from collections.abc import Sequence

from jinja2 import Template
from pandas import Series

from flamme.section.base import BaseSection
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)


class ColumnDtypeSection(BaseSection):
    r"""Implements a section that analyzes the data type of each column.

    Args:
    ----
        dtypes (``Series``): Specifies the data type for each column.
    """

    def __init__(self, dtypes: Series) -> None:
        self._dtypes = dtypes

    def get_statistics(self) -> dict:
        return self._dtypes.to_dict()

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "table": self._dtypes.to_frame(name="dtype").to_html(),
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
This section analyzes the data type of each column.

{{table}}
"""


class ColumnTypeSection(BaseSection):
    r"""Implements a section that analyzes the data type of each column.

    Args:
    ----
        types (``dict``): Specifies the types of the values in each
            column. A column can contain multiple types. The keys are
            the column names.
    """

    def __init__(self, types: dict[str, set]) -> None:
        self._types = types

    def get_statistics(self) -> dict:
        return copy.deepcopy(self._types)

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
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
This section analyzes the type of the values in each column.

{{table}}
"""

    def _create_table(self) -> str:
        rows = "\n".join(
            [create_table_row(column=column, types=types) for column, types in self._types.items()]
        )
        return Template(
            """
<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th>column</th>
            <th>type</th>
        </tr>
    </thead>
    <tbody class="tbody table-group-divider">
        {{rows}}
        <tr class="table-group-divider"></tr>
    </tbody>
</table>
"""
        ).render({"rows": rows})


def create_table_row(column: str, types: set) -> str:
    r"""Creates the HTML code of a new table row.

    Args:
    ----
        column (str): Specifies the column name.
        types (set): Specifies the types in th column.

    Returns:
    -------
        str: The HTML code of a row.
    """
    return Template(
        """<tr>
    <th>{{column}}</th>
    <td >{{null_pct}}</td>
</tr>"""
    ).render({"column": column, "types": ", ".join(sorted([str(t) for t in types]))})
