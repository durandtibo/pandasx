from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from flamme.utils.null import compute_col_null, compute_temporal_null

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


##########################################
#    Tests for compute_temporal_null     #
##########################################


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [None, float("nan"), 0.0, 1.0],
            "col2": [None, 1, 0, None],
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col1": pl.Float64,
            "col2": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


@pytest.fixture()
def dataframe_empty() -> pl.DataFrame:
    return pl.DataFrame(
        {"col1": [], "col2": [], "datetime": []},
        schema={
            "col1": pl.Float64,
            "col2": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


def test_compute_temporal_null(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null(
            frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="1mo"
        ),
        (
            np.array([2, 0, 0, 1], dtype=np.int64),
            np.array([2, 2, 2, 2], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_compute_temporal_null_subset(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null(
            frame=dataframe, columns=["col1"], dt_column="datetime", period="1mo"
        ),
        (
            np.array([1, 0, 0, 0], dtype=np.int64),
            np.array([1, 1, 1, 1], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_compute_temporal_null_empty_columns(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null(frame=dataframe, columns=[], dt_column="datetime", period="1mo"),
        (
            np.array([0, 0, 0, 0], dtype=np.int64),
            np.array([0, 0, 0, 0], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_compute_temporal_null_empty(dataframe_empty: pl.DataFrame) -> None:
    assert objects_are_equal(
        compute_temporal_null(
            frame=dataframe_empty,
            columns=["col1", "col2"],
            dt_column="datetime",
            period="1mo",
        ),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), []),
    )
