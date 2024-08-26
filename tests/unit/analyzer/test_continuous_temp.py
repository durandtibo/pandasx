from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnTemporalContinuousAnalyzer
from flamme.section import ColumnTemporalContinuousSection, EmptySection

######################################################
#     Tests for ColumnTemporalContinuousAnalyzer     #
######################################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
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
            "col": pl.Float64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


def test_column_temporal_continuous_analyzer_str() -> None:
    assert str(
        ColumnTemporalContinuousAnalyzer(column="col", dt_column="datetime", period="1mo")
    ).startswith("ColumnTemporalContinuousAnalyzer(")


def test_column_temporal_continuous_analyzer_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.column == "col"


def test_column_temporal_continuous_analyzer_dt_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.dt_column == "datetime"


def test_column_temporal_continuous_analyzer_period(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.period == "1mo"


def test_column_temporal_continuous_analyzer_yscale_default(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_temporal_continuous_analyzer_yscale(dataframe: pl.DataFrame, yscale: str) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo", yscale=yscale
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.yscale == yscale


def test_column_temporal_continuous_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col",
        dt_column="datetime",
        period="1mo",
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_continuous_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize == figsize


def test_column_temporal_continuous_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_empty() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(pl.DataFrame({"col": [], "datetime": []}))
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_missing_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(pl.DataFrame({"datetime": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_missing_dt_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(pl.DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_same_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(column="col", dt_column="col", period="1mo").analyze(
        pl.DataFrame(
            {
                "col": [
                    datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                    datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                ],
            },
            schema={"col": pl.Datetime(time_unit="us", time_zone="UTC")},
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
