r"""Contain utility functions for counting."""

from __future__ import annotations

__all__ = ["row_temporal_count"]

import numpy as np
import polars as pl
from grizz.utils.period import period_to_strftime_format


def row_temporal_count(
    frame: pl.DataFrame,
    dt_column: str,
    period: str,
) -> tuple[np.ndarray, list[str]]:
    r"""Prepare the data to create the figure and table.

    Args:
        frame: The DataFrame to analyze.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.

    Returns:
        A tuple with the counts and the temporal window labels.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from flamme.utils.count import row_temporal_count
    >>> counts, labels = row_temporal_count(
    ...     frame=pl.DataFrame(
    ...         {
    ...             "col1": [None, float("nan"), 0.0, 1.0, 4.2, 42.0],
    ...             "col2": [None, 1, 0, None, 2, 3],
    ...             "datetime": [
    ...                 datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
    ...                 datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
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
    ...     dt_column="datetime",
    ...     period="1mo",
    ... )
    >>> counts
    array([3, 1, 1, 1])
    >>> labels
    ['2020-01', '2020-02', '2020-03', '2020-04']

    ```
    """
    if frame.shape[0] == 0:
        return np.array([], dtype=np.int64), []

    groups = (
        frame.select(pl.col(dt_column).alias("datetime"), pl.lit(1).alias("count"))
        .sort("datetime")
        .group_by_dynamic("datetime", every=period)
    )
    format_dt = period_to_strftime_format(period)
    labels = [name[0].strftime(format_dt) for name, _ in groups]
    counts = groups.agg(pl.col("count").sum())["count"].to_numpy().astype(np.int64)
    return counts, labels
