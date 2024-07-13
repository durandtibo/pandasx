r"""Implement discrete values analyzers."""

from __future__ import annotations

__all__ = ["ColumnTemporalDiscreteAnalyzer"]

import logging
from typing import TYPE_CHECKING

import polars as pl

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import ColumnTemporalDiscreteSection, EmptySection

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ColumnTemporalDiscreteAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer to show the temporal distribution of
    discrete values.

    Args:
        column: The column to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or
            daily.
        figsize: The figure size in inches. The first dimension
            is the width and the second is the height.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.analyzer import ColumnTemporalDiscreteAnalyzer
    >>> analyzer = ColumnTemporalDiscreteAnalyzer(
    ...     column="str", dt_column="datetime", period="M"
    ... )
    >>> analyzer
    ColumnTemporalDiscreteAnalyzer(column=str, dt_column=datetime, period=M, figsize=None)
    >>> frame = pd.DataFrame(
    ...     {
    ...         "int": np.array([np.nan, 1, 0, 1]),
    ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
    ...         "str": np.array(["A", "B", None, np.nan]),
    ...         "datetime": pd.to_datetime(
    ...             ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
    ...         ),
    ...     }
    ... )
    >>> section = analyzer.analyze(frame)
    >>> section
    ColumnTemporalDiscreteSection(
      (column): str
      (dt_column): datetime
      (period): M
      (figsize): None
    )

    ```
    """

    def __init__(
        self,
        column: str,
        dt_column: str,
        period: str,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self._column = column
        self._dt_column = dt_column
        self._period = period
        self._figsize = figsize

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(column={self._column}, "
            f"dt_column={self._dt_column}, period={self._period}, figsize={self._figsize})"
        )

    def analyze(
        self, frame: pd.DataFrame | pl.DataFrame
    ) -> ColumnTemporalDiscreteSection | EmptySection:
        logger.info(
            f"Analyzing the temporal discrete distribution of {self._column} | "
            f"datetime column: {self._dt_column} | period: {self._period}"
        )
        for column in [self._column, self._dt_column]:
            if column not in frame:
                logger.info(
                    "Skipping temporal discrete distribution analysis because the column "
                    f"({column}) is not in the DataFrame"
                )
                return EmptySection()
        if self._column == self._dt_column:
            logger.info(
                "Skipping temporal discrete distribution analysis because the datetime column "
                f"({self._column}) is the column to analyze"
            )
            return EmptySection()
        if isinstance(frame, pl.DataFrame):  # TODO (tibo): remove later # noqa: TD003
            frame = frame.to_pandas()
        return ColumnTemporalDiscreteSection(
            column=self._column,
            frame=frame,
            dt_column=self._dt_column,
            period=self._period,
            figsize=self._figsize,
        )
