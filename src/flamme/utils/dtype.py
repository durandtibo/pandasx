from __future__ import annotations

__all__ = ["df_column_types", "series_column_types", "read_dtypes_from_schema"]

import logging
from pathlib import Path

import pyarrow.parquet as pq
from pandas import DataFrame, Series

from flamme.utils.path import sanitize_path

logger = logging.getLogger(__name__)


def df_column_types(df: DataFrame) -> dict[str, set]:
    r"""Computes the value types per column.

    Args:
        df: Specifies the DataFrame to analyze.

    Returns:
        A dictionary with the value types for each column.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.utils.dtype import df_column_types
    >>> df = pd.DataFrame(
    ...     {
    ...         "int": np.array([np.nan, 1, 0, 1]),
    ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
    ...     }
    ... )
    >>> coltypes = df_column_types(df)
    >>> coltypes
    {'int': {<class 'float'>}, 'float': {<class 'float'>}}

    ```
    """
    types = {}
    for col in df:
        types[col] = series_column_types(df[col])
    return types


def series_column_types(series: Series) -> set[type]:
    r"""Computes the value types in a ``pandas.Series``.

    Args:
        series: Specifies the DataFrame to analyze.

    Returns:
        A dictionary with the value types for each column.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.utils.dtype import series_column_types
    >>> coltypes = series_column_types(pd.Series([1.2, 4.2, np.nan, 2.2]))
    >>> coltypes
    {<class 'float'>}

    ```
    """
    return {type(x) for x in series.tolist()}


def read_dtypes_from_schema(path: Path | str) -> dict:
    r"""Read the column data types from the schema.

    Args:
        path: Specifies the path to the schema.

    Returns:
        The mapping of column names and data types.
    """
    path = sanitize_path(path)
    logger.info(f"Reading schema from {path}")
    schema = pq.read_schema(path)
    return {name: dtype for name, dtype in zip(schema.names, schema.types)}
