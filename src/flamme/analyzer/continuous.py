from __future__ import annotations

__all__ = ["ColumnContinuousAnalyzer", "ColumnTemporalContinuousAnalyzer"]

import logging

from pandas import DataFrame

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import (
    ColumnContinuousSection,
    ColumnTemporalContinuousSection,
    EmptySection,
)

logger = logging.getLogger(__name__)


class ColumnContinuousAnalyzer(BaseAnalyzer):
    r"""Implements an analyzer to show the temporal distribution of
    continuous values.

    Args:
    ----
        column (str): Specifies the column to analyze.
        nbins (int or None, optional): Specifies the number of bins in
            the histogram. Default: ``None``
        yscale (bool, optional): Specifies the y-axis scale.
            Default: ``linear``
        xmin (float or str or None, optional): Specifies the minimum
            value of the range or its associated quantile.
            ``q0.1`` means the 10% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``q0``
        xmax (float or str or None, optional): Specifies the maximum
            value of the range or its associated quantile.
            ``q0.9`` means the 90% quantile. ``0`` is the minimum
            value and ``1`` is the maximum value. Default: ``q1``
        figsize (``tuple`` , optional): Specifies the figure size in
            inches. The first dimension is the width and the second is
            the height. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.analyzer import ColumnContinuousAnalyzer
        >>> analyzer = ColumnContinuousAnalyzer(column="float")
        >>> analyzer
        ColumnContinuousAnalyzer(column=float, nbins=None, yscale=linear, xmin=q0, xmax=q1, figsize=None)
        >>> df = pd.DataFrame(
        ...     {
        ...         "int": np.array([np.nan, 1, 0, 1]),
        ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
        ...         "str": np.array(["A", "B", None, np.nan]),
        ...     }
        ... )
        >>> section = analyzer.analyze(df)
    """

    def __init__(
        self,
        column: str,
        nbins: int | None = None,
        yscale: str = "linear",
        xmin: float | str | None = "q0",
        xmax: float | str | None = "q1",
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self._column = column
        self._nbins = nbins
        self._yscale = yscale
        self._xmin = xmin
        self._xmax = xmax
        self._figsize = figsize

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(column={self._column}, nbins={self._nbins}, "
            f"yscale={self._yscale}, xmin={self._xmin}, xmax={self._xmax}, figsize={self._figsize})"
        )

    def analyze(self, df: DataFrame) -> ColumnContinuousSection | EmptySection:
        logger.info(f"Analyzing the continuous distribution of {self._column}")
        if self._column not in df:
            logger.info(
                "Skipping temporal continuous distribution analysis because the column "
                f"({self._column}) is not in the DataFrame: {sorted(df.columns)}"
            )
            return EmptySection()
        return ColumnContinuousSection(
            column=self._column,
            series=df[self._column],
            nbins=self._nbins,
            yscale=self._yscale,
            xmin=self._xmin,
            xmax=self._xmax,
            figsize=self._figsize,
        )


class ColumnTemporalContinuousAnalyzer(BaseAnalyzer):
    r"""Implements an analyzer to show the temporal distribution of
    continuous values.

    Args:
    ----
        column (str): Specifies the column to analyze.
        dt_column (str): Specifies the datetime column used to analyze
            the temporal distribution.
        period (str): Specifies the temporal period e.g. monthly or
            daily.
        yscale (bool, optional): Specifies the y-axis scale.
            Default: ``linear``
        figsize (``tuple`` , optional): Specifies the figure size in
            inches. The first dimension is the width and the second is
            the height. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.analyzer import TemporalNullValueAnalyzer
        >>> analyzer = ColumnTemporalContinuousAnalyzer(
        ...     column="float", dt_column="datetime", period="M"
        ... )
        >>> analyzer
        ColumnTemporalContinuousAnalyzer(column=float, dt_column=datetime, period=M, yscale=linear, figsize=None)
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

    def __init__(
        self,
        column: str,
        dt_column: str,
        period: str,
        yscale: str = "linear",
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

    def analyze(self, df: DataFrame) -> ColumnTemporalContinuousSection | EmptySection:
        logger.info(
            f"Analyzing the temporal continuous distribution of {self._column} | "
            f"datetime column: {self._dt_column} | period: {self._period}"
        )
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
        return ColumnTemporalContinuousSection(
            column=self._column,
            df=df,
            dt_column=self._dt_column,
            period=self._period,
            yscale=self._yscale,
            figsize=self._figsize,
        )
