r"""Contain utility functions to analyze data with null values."""

from __future__ import annotations

__all__ = ["compute_col_null", "compute_temporal_null"]


from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.selectors as cs
from grizz.utils.period import period_to_strftime_format

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_col_null(frame: pl.DataFrame) -> pl.DataFrame:
    r"""Return the number and percentage of null values per column.

    Args:
        frame: The DataFrame to analyze.

    Returns:
        A DataFrame with the number and percentage of null values per
            column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from flamme.utils.null import compute_col_null
    >>> frame = compute_col_null(
    ...     pl.DataFrame(
    ...         {
    ...             "int": [None, 1, 0, 1],
    ...             "float": [1.2, 4.2, None, 2.2],
    ...             "str": ["A", "B", None, None],
    ...         },
    ...         schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
    ...     )
    ... )
    >>> frame
    shape: (3, 4)
    ┌────────┬──────┬───────┬──────────┐
    │ column ┆ null ┆ total ┆ null_pct │
    │ ---    ┆ ---  ┆ ---   ┆ ---      │
    │ str    ┆ i64  ┆ i64   ┆ f64      │
    ╞════════╪══════╪═══════╪══════════╡
    │ int    ┆ 1    ┆ 4     ┆ 0.25     │
    │ float  ┆ 1    ┆ 4     ┆ 0.25     │
    │ str    ┆ 2    ┆ 4     ┆ 0.5      │
    └────────┴──────┴───────┴──────────┘

    ```
    """
    if frame.shape[0] > 0:
        null_count = frame.null_count().to_numpy()[0].astype(np.int64)
    else:
        null_count = np.zeros((frame.shape[1],), dtype=np.int64)
    total_count = np.full((frame.shape[1],), frame.shape[0], dtype=np.int64)
    with np.errstate(invalid="ignore"):
        null_pct = null_count.astype(np.float64) / total_count.astype(np.float64)
    return pl.DataFrame(
        {
            "column": list(frame.columns),
            "null": null_count,
            "total": total_count,
            "null_pct": null_pct,
        },
        schema={"column": pl.String, "null": pl.Int64, "total": pl.Int64, "null_pct": pl.Float64},
    )


def compute_temporal_null(
    frame: pl.DataFrame,
    columns: Sequence[str],
    dt_column: str,
    period: str,
) -> tuple[np.ndarray, np.ndarray, list]:
    r"""Compute the number of null values per temporal segments.

    Args:
        frame: The DataFrame to analyze.
        columns: The list of columns to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.

    Returns:
        A tuple with 3 values. The first value is a numpy NDArray
            that contains the number of null values per period. The
            second value is a numpy NDArray that contains the total
            number of values. The third value is a list that contains
            the label of each period.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from flamme.utils.null import compute_temporal_null
    >>> nulls, totals, labels = compute_temporal_null(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col1": [None, float("nan"), 0.0, 1.0],
    ...             "col2": [None, 1, 0, None],
    ...             "datetime": [
    ...                 datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
    ...             ],
    ...         },
    ...         schema={
    ...             "col1": pl.Float64,
    ...             "col2": pl.Int64,
    ...             "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...         },
    ...     ),
    ...     columns=["col1", "col2"],
    ...     dt_column="datetime",
    ...     period="1mo",
    ... )
    >>> nulls
    array([2, 0, 0, 1])
    >>> totals
    array([2, 2, 2, 2])
    >>> labels
    ['2020-01', '2020-02', '2020-03', '2020-04']

    ```
    """
    columns = list(columns)
    frame_na = frame.select([*columns, dt_column]).with_columns(
        cs.by_name(columns).is_null().cast(pl.Int64)
    )
    frame_group = frame_na.sort(dt_column).group_by_dynamic(dt_column, every=period)
    format_dt = period_to_strftime_format(period)
    labels = [name[0].strftime(format_dt) for name, _ in frame_group]

    nulls = np.zeros(len(labels), dtype=np.int64)
    totals = np.zeros(len(labels), dtype=np.int64)
    if columns:
        nulls += (
            frame_group.agg(cs.by_name(columns).sum()).drop(dt_column).sum_horizontal().to_numpy()
        )
        totals += (
            frame_group.agg(cs.by_name(columns).count()).drop(dt_column).sum_horizontal().to_numpy()
        )
    return nulls, totals, labels
