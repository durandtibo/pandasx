from __future__ import annotations

import pandas as pd
import pytest
from coola import objects_are_equal
from pandas import DataFrame
from pandas._testing import assert_frame_equal
from pytest import mark

from flamme.analyzer import TemporalRowCountAnalyzer
from flamme.section import EmptySection, TemporalRowCountSection


@pytest.fixture
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-02-03",
                    "2020-03-03",
                    "2020-04-03",
                ]
            ),
        }
    )


##############################################
#     Tests for TemporalRowCountAnalyzer     #
##############################################


def test_temporal_row_count_analyzer_str() -> None:
    assert str(TemporalRowCountAnalyzer(dt_column="datetime", period="M")).startswith(
        "TemporalRowCountAnalyzer("
    )


def test_temporal_row_count_analyzer_df(dataframe: DataFrame) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert_frame_equal(section.df, dataframe)


@mark.parametrize("dt_column", ("datetime", "date"))
def test_temporal_row_count_analyzer_dt_column(dt_column: str) -> None:
    section = TemporalRowCountAnalyzer(dt_column=dt_column, period="M").analyze(
        DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
                "date": pd.to_datetime(["2021-01-03", "2021-02-03", "2021-03-03", "2021-04-03"]),
            }
        )
    )
    assert section.dt_column == dt_column


@mark.parametrize("period", ("M", "D"))
def test_temporal_row_count_analyzer_period(dataframe: DataFrame, period: str) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period=period).analyze(dataframe)
    assert section.period == period


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_temporal_row_count_analyzer_figsize(
    dataframe: DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="M", figsize=figsize).analyze(
        dataframe
    )
    assert section.figsize == figsize


def test_temporal_row_count_analyzer_figsize_default(dataframe: DataFrame) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert section.figsize is None


def test_temporal_row_count_analyzer_get_statistics(dataframe: DataFrame) -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert isinstance(section, TemporalRowCountSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_row_count_analyzer_get_statistics_empty_rows() -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="M").analyze(
        DataFrame({"datetime": []})
    )
    assert isinstance(section, TemporalRowCountSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_row_count_analyzer_get_statistics_missing_dt_column() -> None:
    section = TemporalRowCountAnalyzer(dt_column="datetime", period="M").analyze(
        DataFrame({"col": []})
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
