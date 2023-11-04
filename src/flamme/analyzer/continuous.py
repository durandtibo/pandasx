from __future__ import annotations

__all__ = ["TemporalContinuousDistributionAnalyzer"]

import logging

from pandas import DataFrame

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import EmptySection, TemporalContinuousDistributionSection

logger = logging.getLogger(__name__)


class TemporalContinuousDistributionAnalyzer(BaseAnalyzer):
    r"""Implements an analyzer to show the temporal distribution of
    continuous values.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.analyzer import TemporalNullValueAnalyzer
        >>> analyzer = TemporalContinuousDistributionAnalyzer(
        ...     column="float", dt_column="datetime", period="M"
        ... )
        >>> analyzer
        TemporalContinuousDistributionAnalyzer(column=float, dt_column=datetime, period=M)
        >>> df = pd.DataFrame(
        ...     {
        ...         "int": np.array([np.nan, 1, 0, 1]),
        ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
        ...         "str": np.array(["A", "B", None, np.nan]),
        ...         "datetime": pd.to_datetime(
        ...             ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
        ...         ),
        ...     }
        ... )
        >>> section = analyzer.analyze(df)
    """

    def __init__(self, column: str, dt_column: str, period: str) -> None:
        self._column = column
        self._dt_column = dt_column
        self._period = period

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(column={self._column}, "
            f"dt_column={self._dt_column}, period={self._period})"
        )

    def analyze(self, df: DataFrame) -> TemporalContinuousDistributionSection | EmptySection:
        if self._column not in df:
            logger.info(
                "Skipping temporal continuous distribution analysis because the column "
                f"({self._column}) is not in the DataFrame: {sorted(df.columns)}"
            )
            return EmptySection()
        if self._dt_column not in df:
            logger.info(
                "Skipping temporal continuous distribution analysis because the datetime column "
                f"({self._dt_column}) is not in the DataFrame: {sorted(df.columns)}"
            )
            return EmptySection()
        return TemporalContinuousDistributionSection(
            column=self._column, df=df, dt_column=self._dt_column, period=self._period
        )
