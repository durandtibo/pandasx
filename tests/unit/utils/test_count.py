from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from flamme.utils.count import row_temporal_count


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ]
        },
        schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


#######################################
#    Tests for row_temporal_count     #
#######################################


def test_row_temporal_count_month(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        row_temporal_count(
            frame=dataframe,
            dt_column="datetime",
            period="1mo",
        ),
        (
            np.array([3, 1, 1, 1], dtype=np.int64),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_row_temporal_count_biweekly(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        row_temporal_count(
            frame=dataframe,
            dt_column="datetime",
            period="2w",
        ),
        (np.array([3, 1, 1, 1], dtype=np.int64), ["2019 52", "2020 04", "2020 08", "2020 12"]),
    )


def test_row_temporal_count_empty() -> None:
    assert objects_are_equal(
        row_temporal_count(
            frame=pl.DataFrame(
                {"datetime": []},
                schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
            ),
            dt_column="datetime",
            period="1mo",
        ),
        (np.array([], dtype=np.int64), []),
    )
