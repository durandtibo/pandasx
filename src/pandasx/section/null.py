from __future__ import annotations

__all__ = ["NullValueSection"]

from collections.abc import Sequence

import numpy as np

from pandasx.section.base import BaseSection


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
        self._columns = columns
        self._null_count = null_count.flatten()
        self._total_count = total_count.flatten()

        if len(self._columns) != self._null_count.shape[0]:
            raise RuntimeError(
                f"columns ({len(self._columns):,}) and null_count ({self._null_count.shape[0]:,}) "
                "do not match"
            )
        if len(self._columns) != self._total_count.shape[0]:
            raise RuntimeError(
                f"columns ({len(self._columns):,}) and total_count ({self._total_count.shape[0]:,}) "
                "do not match"
            )

    def get_statistics(self) -> dict:
        return {}

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        return ""

    def render_html_toc(
        self, number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
    ) -> str:
        return ""
