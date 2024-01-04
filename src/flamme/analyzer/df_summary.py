from __future__ import annotations

__all__ = ["DataFrameSummaryAnalyzer"]

import logging

from pandas import DataFrame

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import DataFrameSummarySection

logger = logging.getLogger(__name__)


class DataFrameSummaryAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer to show a short summary of the DataFrame.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.analyzer import DataFrameSummaryAnalyzer
    >>> analyzer = DataFrameSummaryAnalyzer()
    >>> analyzer
    DataFrameSummaryAnalyzer()
    >>> df = pd.DataFrame(
    ...     {
    ...         "col1": np.array([0, 1, 0, 1]),
    ...         "col2": np.array([1, 0, 1, 0]),
    ...         "col3": np.array([1, 1, 1, 1]),
    ...     }
    ... )
    >>> section = analyzer.analyze(df)

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, df: DataFrame) -> DataFrameSummarySection:
        logger.info("Analyzing the DataFrame...")
        return DataFrameSummarySection(df=df)
