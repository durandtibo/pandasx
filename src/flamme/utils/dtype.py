from __future__ import annotations

__all__ = ["df_column_types", "series_column_types"]

from pandas import DataFrame, Series


def df_column_types(df: DataFrame) -> dict[str, set]:
    r"""Computes the value types per column.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.

    Returns:
    -------
        dict: A dictionary with the value types for each column.

    Example usage:

    .. code-block:: pycon

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
    """
    types = {}
    for col in df:
        types[col] = series_column_types(df[col])
    return types


def series_column_types(series: Series) -> set[type]:
    r"""Computes the value types in a ``pandas.Series``.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.

    Returns:
    -------
        dict: A dictionary with the value types for each column.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.utils.dtype import series_column_types
        >>> coltypes = series_column_types(pd.Series([1.2, 4.2, np.nan, 2.2]))
        >>> coltypes
        {<class 'float'>}
    """
    return {type(x) for x in series.tolist()}
