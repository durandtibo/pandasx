from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
from polars.testing import assert_series_equal

from flamme.utils.data import datetime_range

####################################
#     Tests for datetime_range     #
####################################


def test_datetime_range_day() -> None:
    assert_series_equal(
        datetime_range(
            start=datetime(
                year=2017, month=2, day=3, hour=4, minute=5, second=6, tzinfo=timezone.utc
            ),
            periods=5,
            interval="1d",
            eager=True,
        ).alias("datetime"),
        pl.Series(
            name="datetime",
            values=[
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=4, hour=4, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=5, hour=4, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=6, hour=4, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=7, hour=4, minute=5, second=6, tzinfo=timezone.utc
                ),
            ],
        ),
    )


def test_datetime_range_hour() -> None:
    assert_series_equal(
        datetime_range(
            start=datetime(
                year=2017, month=2, day=3, hour=4, minute=5, second=6, tzinfo=timezone.utc
            ),
            periods=4,
            interval="1h",
            eager=True,
        ).alias("datetime"),
        pl.Series(
            name="datetime",
            values=[
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=5, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=6, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=7, minute=5, second=6, tzinfo=timezone.utc
                ),
            ],
        ),
    )


def test_datetime_range_minute() -> None:
    assert_series_equal(
        datetime_range(
            start=datetime(
                year=2017, month=2, day=3, hour=4, minute=5, second=6, tzinfo=timezone.utc
            ),
            periods=6,
            interval="1m",
            eager=True,
        ).alias("datetime"),
        pl.Series(
            name="datetime",
            values=[
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=5, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=6, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=7, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=8, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=9, second=6, tzinfo=timezone.utc
                ),
                datetime(
                    year=2017, month=2, day=3, hour=4, minute=10, second=6, tzinfo=timezone.utc
                ),
            ],
        ),
    )
