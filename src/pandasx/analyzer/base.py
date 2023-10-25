from __future__ import annotations

__all__ = ["BaseAnalyzer"]

import logging
from abc import ABC

from objectory import AbstractFactory
from pandas import DataFrame

from pandasx.section import BaseSection

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to analyze a DataFrame.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from pandasx.analyzer import NullValueAnalyzer
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
        >>> analyzer.analyze(df)
    """

    def analyze(self, df: DataFrame) -> BaseSection:
        r"""Ingests a DataFrame.

        Returns:
            ``BaseSection``: The section report.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> import pandas as pd
            >>> from pandasx.analyzer import NullValueAnalyzer
            >>> analyzer = NullValueAnalyzer()
            >>> df = pd.DataFrame(
            ...     {
            ...         "int": np.array([np.nan, 1, 0, 1]),
            ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
            ...         "str": np.array(["A", "B", None, np.nan]),
            ...     }
            ... )
            >>> analyzer.analyze(df)
        """
