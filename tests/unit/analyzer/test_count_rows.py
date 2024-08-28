from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from flamme.analyzer import TemporalRowCountAnalyzer
from flamme.section import EmptySection, TemporalRowCountSection


@pytest.fixture
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


##############################################
#     Tests for TemporalRowCountAnalyzer     #
##############################################


def test_temporal_row_count_analyzer_str() -> None:
    assert str(TemporalRowCountAnalyzer(dt_column="datetime", period="1mo")).startswith(
        "TemporalRowCountAnalyzer("
    )


def test_temporal_row_count_analyzer_frame(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="1mo").analyze(dataframe)
    assert_frame_equal(section.frame, dataframe)


@pytest.mark.parametrize("dt_column", ["datetime", "date"])
def test_temporal_row_count_analyzer_dt_column(dt_column: str, dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountAnalyzer(dt_column=dt_column, period="1mo").analyze(
        dataframe.with_columns(pl.col("datetime").alias("date"))
    )
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["1mo", "1d"])
def test_temporal_row_count_analyzer_period(dataframe: pl.DataFrame, period: str) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period=period).analyze(dataframe)
    assert section.period == period


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_temporal_row_count_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="1mo", figsize=figsize).analyze(
        dataframe
    )
    assert section.figsize == figsize


def test_temporal_row_count_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="1mo").analyze(dataframe)
    assert section.figsize is None


def test_temporal_row_count_analyzer_get_statistics(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="1mo").analyze(dataframe)
    assert isinstance(section, TemporalRowCountSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_row_count_analyzer_get_statistics_empty_rows() -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="1mo").analyze(
        pl.DataFrame({"datetime": []})
    )
    assert isinstance(section, TemporalRowCountSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_row_count_analyzer_get_statistics_missing_dt_column() -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="1mo").analyze(
        pl.DataFrame({"col": []})
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
