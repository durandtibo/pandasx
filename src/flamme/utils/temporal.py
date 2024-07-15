r"""Contain utility functions to do temporal transformations."""

from __future__ import annotations

__all__ = ["to_temporal_frames"]

from typing import TYPE_CHECKING

from grizz.utils.period import period_to_strftime_format

if TYPE_CHECKING:
    import polars as pl


def to_temporal_frames(
    frame: pl.DataFrame,
    dt_column: str,
    period: str,
) -> tuple[list[pl.DataFrame], list[str]]:
    r"""Return a list of temporal DataFrames and the associated time
    steps.

    Args:
        frame: The DataFrame to analyze.
        dt_column: The datetime column used to create the temporal
            DataFrames.
        period: The temporal period e.g. monthly or daily.

    Returns:
        A tuple with the counts and the temporal steps.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from flamme.utils.temporal import to_temporal_frames
    >>> frames, steps = to_temporal_frames(
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
    >>> frames
    [shape: (3, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ null ┆ null ┆ 2020-01-03 00:00:00 UTC │
    │ NaN  ┆ 1    ┆ 2020-01-04 00:00:00 UTC │
    │ 0.0  ┆ 0    ┆ 2020-01-05 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘, shape: (1, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ 1.0  ┆ null ┆ 2020-02-03 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘, shape: (1, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ 4.2  ┆ 2    ┆ 2020-03-03 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘, shape: (1, 3)
    ┌──────┬──────┬─────────────────────────┐
    │ col1 ┆ col2 ┆ datetime                │
    │ ---  ┆ ---  ┆ ---                     │
    │ f64  ┆ i64  ┆ datetime[μs, UTC]       │
    ╞══════╪══════╪═════════════════════════╡
    │ 42.0 ┆ 3    ┆ 2020-04-03 00:00:00 UTC │
    └──────┴──────┴─────────────────────────┘]
    >>> steps
    ['2020-01', '2020-02', '2020-03', '2020-04']

    ```
    """
    if frame.shape[0] == 0:
        return [], []

    groups = frame.sort(dt_column).group_by_dynamic(dt_column, every=period)
    format_dt = period_to_strftime_format(period)
    steps = [name[0].strftime(format_dt) for name, _ in groups]
    frames = [frame for _, frame in groups]
    return frames, steps
