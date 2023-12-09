from __future__ import annotations

__all__ = ["ColumnDataFrameTransformer"]

from collections.abc import Mapping

from coola.utils import str_indent, str_mapping
from pandas import DataFrame

from flamme.transformer.df.base import BaseDataFrameTransformer
from flamme.transformer.series.base import BaseSeriesTransformer


class ColumnDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implements a ``pandas.DataFrame`` transformer that applies
    ``pandas.Series`` transformers on some columns.

    Args:
    ----
        columns (``Mapping``): Specifies the ``pandas.Series``
            transformers.

    Example usage:

    .. code-block:: pycon

        >>> import pandas as pd
        >>> from flamme.transformer.df import Column
        >>> from flamme.transformer.series import ToNumeric, ToDatetime
        >>> transformer = Column({'col2': ToNumeric(), 'col3': ToDatetime()})
        >>> transformer
        ColumnDataFrameTransformer(columns=['col1', 'col2'])
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": [1, 2, 3, 4, 5],
        ...         "col2": ['1', '2', '3', '4', '5'],
        ...         "col3": [" a", "b ", " c ", "  d  ", "e"],
        ...     }
        ... )
        >>> df = transformer.transform(df)
        >>> df
                 col1  col2
        0    2020-1-1     1
        1    2020-1-2     2
        2   2020-1-31     3
        3  2020-12-31     4
        4        None     5
    """

    def __init__(self, columns: Mapping[str, BaseSeriesTransformer | dict]) -> None:
        self._columns = columns

    def __repr__(self) -> str:
        args = ""
        if self._columns:
            args = f"\n  {str_indent(str_mapping(self._columns))}\n"
        return f"{self.__class__.__qualname__}({args})"

    def transform(self, df: DataFrame) -> DataFrame:
        for col, transformer in self._columns.items():
            if col not in df:
                raise RuntimeError(
                    f"Column {col} is not in the DataFrame (columns:{sorted(df.columns)})"
                )
            df[col] = transformer.transform(df[col])
        return df
