r"""Implement an analyzer that generates a section to analyze the number
of null values."""

from __future__ import annotations

__all__ = ["NullValueAnalyzer"]

import logging

import numpy as np
import pandas as pd
import polars as pl

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import NullValueSection

logger = logging.getLogger(__name__)


class NullValueAnalyzer(BaseAnalyzer):
    r"""Implement a null value analyzer.

    Args:
        figsize: The figure size in inches. The first
            dimension is the width and the second is the height.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from flamme.analyzer import NullValueAnalyzer
    >>> analyzer = NullValueAnalyzer()
    >>> analyzer
    NullValueAnalyzer(figsize=None)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "float": [1.2, 4.2, None, 2.2],
    ...         "int": [None, 1, 0, 1],
    ...         "str": ["A", "B", None, None],
    ...     },
    ...     schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
    ... )
    >>> section = analyzer.analyze(frame)
    >>> section
    NullValueSection(
      (columns): ('float', 'int', 'str')
      (null_count): array([1, 1, 2])
      (total_count): array([4, 4, 4])
      (figsize): None
    )

    ```
    """

    def __init__(self, figsize: tuple[float, float] | None = None) -> None:
        self._figsize = figsize

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(figsize={self._figsize})"

    def analyze(self, frame: pd.DataFrame | pl.DataFrame) -> NullValueSection:
        logger.info("Analyzing the null value distribution of all columns...")
        if isinstance(frame, pd.DataFrame):  # TODO (tibo): remove later  # noqa: TD003
            frame = pl.from_pandas(frame)
        nrows, ncols = frame.shape
        return NullValueSection(
            columns=list(frame.columns),
            null_count=(
                frame.null_count().to_numpy().astype(int)
                if ncols > 0
                else np.zeros(ncols, dtype=int)
            ),
            total_count=np.full((ncols,), nrows),
            figsize=self._figsize,
        )
