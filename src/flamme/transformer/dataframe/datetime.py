r"""Contain ``pandas.DataFrame`` transformers to transform columns with
datetime values."""

from __future__ import annotations

__all__ = ["ToDatetimeDataFrameTransformer"]

from typing import TYPE_CHECKING, Any

import pandas as pd
from tqdm import tqdm

from flamme.transformer.dataframe.base import BaseDataFrameTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


class ToDatetimeDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to convert some columns to numeric type.

    Args:
        columns: The columns to convert.
        **kwargs: The keyword arguments for
            ``pandas.to_datetime``.

    Example usage:

    .. code-block:: pycon

        >>> import pandas as pd
        >>> from flamme.transformer.dataframe import ToDatetime
        >>> transformer = ToDatetime(columns=["col1"])
        >>> transformer
        ToDatetimeDataFrameTransformer(columns=('col1',))
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
        ...         "col2": [1, 2, 3, 4, 5],
        ...         "col3": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.dtypes
        col1    object
        col2     int64
        col3    object
        dtype: object
        >>> df = transformer.transform(df)
        >>> df.dtypes
        col1    datetime64[ns]
        col2             int64
        col3            object
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
            df[col] = pd.to_datetime(df[col], **self._kwargs)
        return df
