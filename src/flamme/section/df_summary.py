from __future__ import annotations

__all__ = ["DataFrameSummarySection"]

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
from flamme.utils.dtype import series_column_types

logger = logging.getLogger(__name__)


class DataFrameSummarySection(BaseSection):
    r"""Implement a section that returns a summary of a DataFrame.

    Args:
        df: Specifies the DataFrame to analyze.
    """

    def __init__(self, df: DataFrame) -> None:
        self._df = df

    @property
    def df(self) -> DataFrame:
        r"""The DataFrame to analyze."""
        return self._df

    def get_columns(self) -> tuple[str, ...]:
        return tuple(self._df.columns)

    def get_null_count(self) -> tuple[int, ...]:
        return tuple(self._df.isna().sum().to_frame("__count__")["__count__"].tolist())

    def get_nunique(self) -> tuple[int, ...]:
        return tuple(self._df.nunique(dropna=False).tolist())

    def get_column_types(self) -> tuple[set, ...]:
        return tuple(series_column_types(self._df[col]) for col in self.df)

    def get_statistics(self) -> dict:
        return {
            "columns": self.get_columns(),
            "null_count": self.get_null_count(),
            "nunique": self.get_nunique(),
            "column_types": self.get_column_types(),
        }

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        logger.info("Rendering the DataFrame summary section...")
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
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
This section shows a short summary of each column.

<ul>
  <li> <b>column</b>: are the column names</li>
  <li> <b>null</b>: are the number (and percentage) of null values in the column </li>
  <li> <b>unique</b>: are the number (and percentage) of unique values in the column </li>
  <li> <b>types</b>: are the real object types for the objects in the column </li>
</ul>

{{table}}
"""
