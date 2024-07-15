from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal

from flamme.utils.temporal import to_temporal_frames


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [1, 2, 3, 4, 5, 6],
            "datetime": [
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col1": pl.Float64,
            "col2": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


########################################
#     Tests for to_temporal_frames     #
########################################


def test_to_temporal_frames_monthly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_temporal_frames(dataframe, dt_column="datetime", period="1mo"),
        (
            [
                pl.DataFrame(
                    {
                        "col1": [1.0, 2.0, 3.0],
                        "col2": [2, 3, 4],
                        "datetime": [
                            datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                            datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                            datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                        ],
                    },
                    schema={
                        "col1": pl.Float64,
                        "col2": pl.Int64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
                pl.DataFrame(
                    {
                        "col1": [4.0],
                        "col2": [5],
                        "datetime": [datetime(year=2020, month=2, day=3, tzinfo=timezone.utc)],
                    },
                    schema={
                        "col1": pl.Float64,
                        "col2": pl.Int64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
                pl.DataFrame(
                    {
                        "col1": [5.0],
                        "col2": [6],
                        "datetime": [datetime(year=2020, month=3, day=3, tzinfo=timezone.utc)],
                    },
                    schema={
                        "col1": pl.Float64,
                        "col2": pl.Int64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
                pl.DataFrame(
                    {
                        "col1": [0.0],
                        "col2": [1],
                        "datetime": [datetime(year=2020, month=4, day=3, tzinfo=timezone.utc)],
                    },
                    schema={
                        "col1": pl.Float64,
                        "col2": pl.Int64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
            ],
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_to_temporal_frames_yearly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        to_temporal_frames(dataframe, dt_column="datetime", period="1y"),
        (
            [
                pl.DataFrame(
                    {
                        "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
                        "col2": [2, 3, 4, 5, 6, 1],
                        "datetime": [
                            datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                            datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
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
            ],
            ["2020"],
        ),
    )


def test_to_temporal_frames_empty() -> None:
    assert objects_are_equal(
        to_temporal_frames(pl.DataFrame({}), dt_column="datetime", period="1mo"), ([], [])
    )
