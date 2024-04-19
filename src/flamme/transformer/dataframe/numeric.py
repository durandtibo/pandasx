r"""Contain ``pandas.DataFrame`` transformers to transform columns with
numeric values."""

from __future__ import annotations

__all__ = ["ToNumericDataFrameTransformer"]

from typing import TYPE_CHECKING, Any

import pandas as pd
from tqdm import tqdm

from flamme.transformer.dataframe.base import BaseDataFrameTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


class ToNumericDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to convert some columns to numeric type.

    Args:
        columns: The columns to convert.
        **kwargs: The keyword arguments for
            ``pandas.to_numeric``.

    Example usage:

    .. code-block:: pycon

        >>> import pandas as pd
        >>> from flamme.transformer.dataframe import ToNumeric
        >>> transformer = ToNumeric(columns=["col1", "col3"])
        >>> transformer
        ToNumericDataFrameTransformer(columns=('col1', 'col3'))
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ["1", "2", "3", "4", "5"],
        ...         "col3": ["1", "2", "3", "4", "5"],
        ...         "col4": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.dtypes
        col1     int64
        col2    object
        col3    object
        col4    object
        dtype: object
        >>> df = transformer.transform(df)
        >>> df.dtypes
        col1     int64
        col2    object
        col3     int64
        col4    object
        dtype: object
    """

    def __init__(self, columns: Sequence[str], **kwargs: Any) -> None:
        self._columns = tuple(columns)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join([f"{key}={value}" for key, value in self._kwargs.items()])
        if args:
            args = ", " + args
        return f"{self.__class__.__qualname__}(columns={self._columns}{args})"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in tqdm(self._columns, desc="Converting to numeric type"):
            df[col] = pd.to_numeric(df[col], **self._kwargs)
        return df
