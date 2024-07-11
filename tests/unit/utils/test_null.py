from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from flamme.utils.null import compute_col_null

######################################
#     Tests for compute_col_null     #
######################################


def test_compute_col_null() -> None:
    assert_frame_equal(
        compute_col_null(
            pl.DataFrame(
                {
                    "int": [None, 1, 0, 1],
                    "float": [1.2, 4.2, None, 2.2],
                    "str": ["A", "B", None, None],
                },
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            )
        ),
        pl.DataFrame(
            {
                "column": ["int", "float", "str"],
                "null": [1, 1, 2],
                "total": [4, 4, 4],
                "null_pct": [0.25, 0.25, 0.5],
            },
            schema={
                "column": pl.String,
                "null": pl.Int64,
                "total": pl.Int64,
                "null_pct": pl.Float64,
            },
        ),
    )


def test_compute_col_null_empty_row() -> None:
    assert_frame_equal(
        compute_col_null(
            pl.DataFrame(
                {"int": [], "float": [], "str": []},
                schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
            )
        ),
        pl.DataFrame(
            {
                "column": ["int", "float", "str"],
                "null": [0, 0, 0],
                "total": [0, 0, 0],
                "null_pct": [np.nan, np.nan, np.nan],
            },
            schema={
                "column": pl.String,
                "null": pl.Int64,
                "total": pl.Int64,
                "null_pct": pl.Float64,
            },
        ),
    )


def test_compute_col_null_empty() -> None:
    assert_frame_equal(
        compute_col_null(pl.DataFrame({})),
        pl.DataFrame(
            {
                "column": [],
                "null": [],
                "total": [],
                "null_pct": [],
            },
            schema={
                "column": pl.String,
                "null": pl.Int64,
                "total": pl.Int64,
                "null_pct": pl.Float64,
            },
        ),
    )
