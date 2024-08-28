from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from flamme.analyzer import TemporalNullValueAnalyzer
from flamme.section import EmptySection, TemporalNullValueSection


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [1.2, 4.2, None, 2.2],
            "int": [None, 1, None, 1],
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "float": pl.Float64,
            "int": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


###############################################
#     Tests for TemporalNullValueAnalyzer     #
###############################################


def test_temporal_null_value_analyzer_str() -> None:
    assert str(TemporalNullValueAnalyzer(dt_column="datetime", period="M")).startswith(
        "TemporalNullValueAnalyzer("
    )


def test_temporal_null_value_analyzer_frame(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert_frame_equal(section.frame, dataframe)


def test_temporal_null_value_analyzer_columns(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueAnalyzer(
        dt_column="datetime", period="M", columns=("float",)
    ).analyze(dataframe)
    assert section.columns == ("float",)


def test_temporal_null_value_analyzer_columns_none(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert section.columns == ("float", "int")


@pytest.mark.parametrize("dt_column", ["datetime", "date"])
def test_temporal_null_value_analyzer_dt_column(dataframe: pl.DataFrame, dt_column: str) -> None:
    dataframe = dataframe.with_columns(pl.col("datetime").alias("date"))
    section = TemporalNullValueAnalyzer(dt_column=dt_column, period="M").analyze(dataframe)
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["M", "D"])
def test_temporal_null_value_analyzer_period(dataframe: pl.DataFrame, period: str) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period=period).analyze(dataframe)
    assert section.period == period


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_temporal_null_value_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M", figsize=figsize).analyze(
        dataframe
    )
    assert section.figsize == figsize


def test_temporal_null_value_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert section.figsize is None


def test_temporal_null_value_analyzer_get_statistics(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert isinstance(section, TemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_null_value_analyzer_get_statistics_empty_rows() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        pl.DataFrame(
            {"col": [], "datetime": []},
            schema={
                "col": pl.Float64,
                "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        )
    )
    assert isinstance(section, TemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_null_value_analyzer_get_statistics_missing_dt_column() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        pl.DataFrame({"col": []})
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
