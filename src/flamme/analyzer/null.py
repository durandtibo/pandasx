from __future__ import annotations

__all__ = ["MonthlyNullValueAnalyzer", "NullValueAnalyzer"]

import logging

import numpy as np
from pandas import DataFrame

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import EmptySection
from flamme.section.null import MonthlyNullValueSection, NullValueSection

logger = logging.getLogger(__name__)


class NullValueAnalyzer(BaseAnalyzer):
    r"""Implements a null value analyzer.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.analyzer import NullValueAnalyzer
        >>> analyzer = NullValueAnalyzer()
        >>> analyzer
        NullValueAnalyzer()
        >>> df = pd.DataFrame(
        ...     {
        ...         "int": np.array([np.nan, 1, 0, 1]),
        ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
        ...         "str": np.array(["A", "B", None, np.nan]),
        ...     }
        ... )
        >>> section = analyzer.analyze(df)
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, df: DataFrame) -> NullValueSection:
        return NullValueSection(
            columns=list(df.columns),
            null_count=df.isnull().sum().to_frame("count")["count"].to_numpy(),
            total_count=np.full((df.shape[1],), df.shape[0]),
        )


class MonthlyNullValueAnalyzer(BaseAnalyzer):
    r"""Implements an analyzer to show the monthly distribution of null
    values.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.analyzer import MonthlyNullValueAnalyzer
        >>> analyzer = MonthlyNullValueAnalyzer()
        >>> analyzer
        MonthlyNullValueAnalyzer()
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

    def __init__(self, dt_column: str) -> None:
        self._dt_column = dt_column

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(dt_column={self._dt_column})"

    def analyze(self, df: DataFrame) -> MonthlyNullValueSection | EmptySection:
        if self._dt_column not in df:
            logger.info(
                "Skipping monthly null value analysis because the datetime column "
                f"({self._dt_column}) is not in the DataFrame: {sorted(df.columns)}"
            )
            return EmptySection()
        return MonthlyNullValueSection(df=df, dt_column=self._dt_column)
