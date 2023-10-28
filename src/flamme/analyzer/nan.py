from __future__ import annotations

__all__ = ["NanValueAnalyzer"]

import numpy as np
from pandas import DataFrame

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import NanValueSection


class NanValueAnalyzer(BaseAnalyzer):
    r"""Implements a NaN value analyzer.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.analyzer import NanValueAnalyzer
        >>> analyzer = NanValueAnalyzer()
        >>> analyzer
        NanValueAnalyzer()
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

    def analyze(self, df: DataFrame) -> NanValueSection:
        return NanValueSection(
            columns=list(df.columns),
            nan_count=df.isna().sum().to_frame("count")["count"].to_numpy(),
            total_count=np.full((df.shape[1],), df.shape[0]),
        )
