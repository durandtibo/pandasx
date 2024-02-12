r"""Implement an analyzer that generates a markdown section."""

from __future__ import annotations

__all__ = ["MarkdownAnalyzer"]

from typing import TYPE_CHECKING

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import MarkdownSection

if TYPE_CHECKING:
    from pandas import DataFrame


class MarkdownAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer that adds a mardown string to the report.

    Args:
        desc: Specifies the markdown description.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.analyzer import MarkdownAnalyzer
    >>> analyzer = MarkdownAnalyzer(desc="hello cats!")
    >>> analyzer
    MarkdownAnalyzer()
    >>> df = pd.DataFrame({})
    >>> section = analyzer.analyze(df)

    ```
    """

    def __init__(self, desc: str) -> None:
        self._desc = str(desc)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, df: DataFrame) -> MarkdownSection:
        return MarkdownSection(desc=self._desc)
