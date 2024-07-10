from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal
from pandas.testing import assert_frame_equal

from flamme.analyzer import ColumnTemporalNullValueAnalyzer
from flamme.section import ColumnTemporalNullValueSection, EmptySection


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [1.2, 4.2, None, 2.2],
            "int": [None, 1, 0, 1],
            "str": ["A", "B", None, None],
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
            "str": pl.String,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


#####################################################
#     Tests for ColumnTemporalNullValueAnalyzer     #
#####################################################


def test_column_temporal_null_value_analyzer_str() -> None:
    assert str(ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M")).startswith(
        "ColumnTemporalNullValueAnalyzer("
    )


def test_column_temporal_null_value_analyzer_frame(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert_frame_equal(section.frame, dataframe.to_pandas())


def test_column_temporal_null_value_analyzer_columns_none(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert section.columns == ("float", "int", "str")
    assert section.ncols == 2


def test_column_temporal_null_value_analyzer_columns_1(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", columns=["float"]
    ).analyze(dataframe)
    assert section.columns == ("float",)
    assert section.ncols == 1


def test_column_temporal_null_value_analyzer_columns(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", columns=["float", "int", "str"]
    ).analyze(dataframe)
    assert section.columns == ("float", "int", "str")
    assert section.ncols == 2


@pytest.mark.parametrize("dt_column", ["datetime", "str"])
def test_column_temporal_null_value_analyzer_dt_column(
    dataframe: pl.DataFrame, dt_column: str
) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column=dt_column, period="M").analyze(dataframe)
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["M", "D"])
def test_column_temporal_null_value_analyzer_period(dataframe: pl.DataFrame, period: str) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period=period).analyze(
        dataframe
    )
    assert section.period == period


@pytest.mark.parametrize("ncols", [1, 2])
def test_column_temporal_null_value_analyzer_ncols(dataframe: pl.DataFrame, ncols: int) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", ncols=ncols
    ).analyze(dataframe)
    assert section.ncols == ncols


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_null_value_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[int, int]
) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert section.figsize == figsize


def test_column_temporal_null_value_analyzer_get_statistics(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert isinstance(section, ColumnTemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_null_value_analyzer_get_statistics_empty() -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        pl.DataFrame(
            {"float": [], "int": [], "str": [], "datetime": []},
            schema={
                "float": pl.Float64,
                "int": pl.Int64,
                "str": pl.String,
                "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
            },
        )
    )
    assert isinstance(section, ColumnTemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_null_value_analyzer_get_statistics_missing_empty_column() -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        pl.DataFrame({})
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
