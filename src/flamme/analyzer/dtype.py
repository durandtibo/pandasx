r"""Implement an analyzer that generates a section to analyze the data
types of each column."""

from __future__ import annotations

__all__ = ["DataTypeAnalyzer"]

import logging


from flamme.analyzer.base import BaseAnalyzer
from flamme.section import DataTypeSection
from flamme.utils.dtype2 import frame_types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd

logger = logging.getLogger(__name__)


class DataTypeAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer to find all the value types in each column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from flamme.analyzer import DataTypeAnalyzer
    >>> analyzer = DataTypeAnalyzer()
    >>> analyzer
    DataTypeAnalyzer()
    >>> frame = pl.DataFrame(
    ...     {
    ...         "int": [None, 1, 0, 1],
    ...         "float": [1.2, 4.2, float("nan"), 2.2],
    ...         "str": ["A", "B", None, None],
    ...     },
    ...     schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
    ... )
    >>> section = analyzer.analyze(frame)
    >>> section
    DataTypeSection(
      (dtypes): {'int': Int64, 'float': Float64, 'str': String}
      (types): {'int': {<class 'int'>, <class 'NoneType'>}, 'float': {<class 'float'>}, 'str': {<class 'NoneType'>, <class 'str'>}}
    )

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, frame: pl.DataFrame | pd.DataFrame) -> DataTypeSection:
        logger.info("Analyzing the data types...")
        return DataTypeSection(dtypes=dict(frame.schema), types=frame_types(frame))
