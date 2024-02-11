r"""Contain ``pandas.DataFrame`` transformers to transform columns with
decimal values."""

from __future__ import annotations

__all__ = ["DecimalToNumericDataFrameTransformer"]


from typing import TYPE_CHECKING, Any

import pandas as pd
from tqdm import tqdm

from flamme.transformer.df.base import BaseDataFrameTransformer
from flamme.utils.filtering import find_columns_decimal

if TYPE_CHECKING:
    from pandas import DataFrame


class DecimalToNumericDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a ``pandas.DataFrame`` to convert all the columns with
    ``Decimal`` objects to floats.

    Args:
        **kwargs: Specifies the keyword arguments for
            ``pandas.to_numeric``.

    Example usage:

    .. code-block:: pycon

        >>> from decimal import Decimal
        >>> import pandas as pd
        >>> from flamme.transformer.df import DecimalToNumeric
        >>> transformer = DecimalToNumericDataFrameTransformer()
        >>> transformer
        DecimalToNumericDataFrameTransformer()
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, Decimal(5)],
        ...         "col2": [Decimal(1), Decimal(2), Decimal(3), Decimal(4), Decimal(5)],
        ...         "col3": ["1", "2", "3", "4", "5"],
        ...         "col4": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.dtypes
        col1    object
        col2    object
        col3    object
        col4    object
        dtype: object
        >>> df = transformer.transform(df)
        >>> df.dtypes
        col1     int64
        col2   float64
        col3    object
        col4    object
        dtype: object
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join([f"{key}={value}" for key, value in self._kwargs.items()])
        return f"{self.__class__.__qualname__}({args})"

    def transform(self, df: DataFrame) -> DataFrame:
        columns = find_columns_decimal(df)
        for col in tqdm(columns, desc="Converting Decimal to numeric type"):
            df[col] = pd.to_numeric(df[col], **self._kwargs)
        return df
