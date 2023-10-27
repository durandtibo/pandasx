from __future__ import annotations

__all__ = ["ColumnDtypeSection"]

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
