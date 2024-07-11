r"""Contain utility functions to analyze data with null values."""

from __future__ import annotations

__all__ = ["compute_col_null"]

import numpy as np
import polars as pl


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
