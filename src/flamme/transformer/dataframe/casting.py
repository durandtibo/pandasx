r"""Contain ``polars.DataFrame`` transformers to convert some columns to
a new data type."""

from __future__ import annotations

__all__ = ["CastDataFrameTransformer", "ToTimeDataFrameTransformer"]

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

from flamme.transformer.dataframe.base import BaseDataFrameTransformer
from flamme.utils.imports import is_tqdm_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_tqdm_available():
    from tqdm import tqdm
else:  # pragma: no cover
    from flamme.utils.noop import tqdm

logger = logging.getLogger(__name__)


class CastDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to convert some columns to a new data
    type.

    Args:
        columns: The columns to convert.
        dtype: The target data type.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.
        **kwargs: The keyword arguments for ``cast``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from flamme.transformer.dataframe import Cast
    >>> transformer = Cast(columns=["col1", "col3"], dtype=pl.Int32)
    >>> transformer
    CastDataFrameTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["1", "2", "3", "4", "5"],
    ...         "col4": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> frame
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 ┆ col4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i32  ┆ str  ┆ i32  ┆ str  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┴──────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str],
        dtype: type[pl.DataType],
        ignore_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        self._columns = tuple(columns)
        self._dtype = dtype
        self._ignore_missing = bool(ignore_missing)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join([f"{key}={value}" for key, value in self._kwargs.items()])
        if args:
            args = ", " + args
        return (
            f"{self.__class__.__qualname__}(columns={self._columns}, dtype={self._dtype}, "
            f"ignore_missing={self._ignore_missing}{args})"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for col in tqdm(self._columns, desc=f"converting to {self._dtype}"):
            if col not in frame:
                if self._ignore_missing:
                    logger.warning(
                        f"skipping transformation for column {col} because the column is missing"
                    )
                else:
                    msg = f"column {col} is not in the DataFrame (columns:{sorted(frame.columns)})"
                    raise RuntimeError(msg)
            else:
                logger.info(f"transforming column `{col}`...")
                frame = frame.with_columns(
                    frame.select(pl.col(col).cast(self._dtype, **self._kwargs))
                )
        return frame


class ToTimeDataFrameTransformer(BaseDataFrameTransformer):
    r"""Implement a transformer to convert some columns to a
    ``polars.Time`` type.

    Args:
        columns: The columns to convert.
        format: Format to use for conversion. Refer to the
            [chrono crate documentation](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)
            for the full specification. Example: ``"%H:%M:%S"``.
            If set to ``None`` (default), the format is inferred from
            the data.
        ignore_missing: If ``False``, an exception is raised if a
            column is missing, otherwise just a warning message is
            shown.
        **kwargs: The keyword arguments for ``cast``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from flamme.transformer.dataframe import ToTime
    >>> transformer = ToTime(columns=["col1"], format="%H:%M:%S")
    >>> transformer
    ToTimeDataFrameTransformer(columns=('col1',), format=%H:%M:%S, ignore_missing=False)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["01:01:01", "02:02:02", "12:00:01", "18:18:18", "23:59:59"],
    ...     }
    ... )
    >>> frame
    shape: (5, 3)
    ┌──────────┬──────┬──────────┐
    │ col1     ┆ col2 ┆ col3     │
    │ ---      ┆ ---  ┆ ---      │
    │ str      ┆ str  ┆ str      │
    ╞══════════╪══════╪══════════╡
    │ 01:01:01 ┆ 1    ┆ 01:01:01 │
    │ 02:02:02 ┆ 2    ┆ 02:02:02 │
    │ 12:00:01 ┆ 3    ┆ 12:00:01 │
    │ 18:18:18 ┆ 4    ┆ 18:18:18 │
    │ 23:59:59 ┆ 5    ┆ 23:59:59 │
    └──────────┴──────┴──────────┘
    >>> out = transformer.transform(frame)
    >>> out
    shape: (5, 3)
    ┌──────────┬──────┬──────────┐
    │ col1     ┆ col2 ┆ col3     │
    │ ---      ┆ ---  ┆ ---      │
    │ time     ┆ str  ┆ str      │
    ╞══════════╪══════╪══════════╡
    │ 01:01:01 ┆ 1    ┆ 01:01:01 │
    │ 02:02:02 ┆ 2    ┆ 02:02:02 │
    │ 12:00:01 ┆ 3    ┆ 12:00:01 │
    │ 18:18:18 ┆ 4    ┆ 18:18:18 │
    │ 23:59:59 ┆ 5    ┆ 23:59:59 │
    └──────────┴──────┴──────────┘

    ```
    """

    def __init__(
        self,
        columns: Sequence[str],
        format: str | None = None,  # noqa: A002
        ignore_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        self._columns = tuple(columns)
        self._format = format
        self._ignore_missing = bool(ignore_missing)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join([f"{key}={value}" for key, value in self._kwargs.items()])
        if args:
            args = ", " + args
        return (
            f"{self.__class__.__qualname__}(columns={self._columns}, format={self._format}, "
            f"ignore_missing={self._ignore_missing}{args})"
        )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        for col in tqdm(self._columns, desc=f"converting to time ({self._format})"):
            if col not in frame:
                if self._ignore_missing:
                    logger.warning(
                        f"skipping transformation for column {col} because the column is missing"
                    )
                else:
                    msg = f"column {col} is not in the DataFrame (columns:{sorted(frame.columns)})"
                    raise RuntimeError(msg)
            else:
                logger.info(f"transforming column `{col}`...")
                frame = frame.with_columns(
                    frame.select(pl.col(col).str.to_time(self._format, **self._kwargs))
                )
        return frame
