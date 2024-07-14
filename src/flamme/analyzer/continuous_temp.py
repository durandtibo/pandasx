r"""Implement continuous values analyzers."""

from __future__ import annotations

__all__ = ["ColumnTemporalContinuousAnalyzer"]

import logging
from typing import TYPE_CHECKING

import polars as pl

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import ColumnTemporalContinuousSection, EmptySection

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ColumnTemporalContinuousAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer to show the temporal distribution of
    continuous values.

    Args:
        column: The column to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        yscale: The y-axis scale. If ``'auto'``, the
            ``'linear'`` or ``'log'/'symlog'`` scale is chosen based
            on the distribution.
        figsize: The figure size in inches. The first
            dimension is the width and the second is the height.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.analyzer import TemporalNullValueAnalyzer
    >>> analyzer = ColumnTemporalContinuousAnalyzer(
    ...     column="float", dt_column="datetime", period="M"
    ... )
    >>> analyzer
    ColumnTemporalContinuousAnalyzer(column=float, dt_column=datetime, period=M, yscale=auto, figsize=None)
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

    ```
    """

    def __init__(
        self,
        column: str,
        dt_column: str,
        period: str,
        yscale: str = "auto",
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self._column = column
        self._dt_column = dt_column
        self._period = period
        self._yscale = yscale
        self._figsize = figsize

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(column={self._column}, "
            f"dt_column={self._dt_column}, period={self._period}, "
            f"yscale={self._yscale}, figsize={self._figsize})"
        )

    def analyze(
        self, frame: pd.DataFrame | pl.DataFrame
    ) -> ColumnTemporalContinuousSection | EmptySection:
        logger.info(
            f"Analyzing the temporal continuous distribution of {self._column} | "
            f"datetime column: {self._dt_column} | period: {self._period}"
        )
        for column in [self._column, self._dt_column]:
            if column not in frame:
                logger.info(
                    "Skipping temporal continuous distribution analysis because the column "
                    f"({column}) is not in the DataFrame"
                )
                return EmptySection()
        if self._column == self._dt_column:
            logger.info(
                "Skipping temporal continuous distribution analysis because the datetime column "
                f"({self._column}) is the column to analyze"
            )
            return EmptySection()
        if isinstance(frame, pl.DataFrame):  # TODO (tibo): remove later # noqa: TD003
            frame = frame.to_pandas()
        return ColumnTemporalContinuousSection(
            column=self._column,
            frame=frame,
            dt_column=self._dt_column,
            period=self._period,
            yscale=self._yscale,
            figsize=self._figsize,
        )
