from __future__ import annotations

__all__ = ["ContinuousDistributionSection"]

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


class ContinuousDistributionSection(BaseSection):
    r"""Implements a section that analyzes a continuous distribution of
    values.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.
        column (str): Specifies the column of the DataFrame to analyze.
    """

    def __init__(self, df: DataFrame, column: str = "N/A") -> None:
        self._df = df
        self._column = column

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

{{figure}}
{{table}}
<p style="margin-top: 1rem;">
"""
