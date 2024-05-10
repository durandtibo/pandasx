from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from flamme.analyzer import TemporalNullValueAnalyzer
from flamme.section import EmptySection, TemporalNullValueSection


@pytest.fixture()
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col1": np.array([1.2, 4.2, np.nan, 2.2]),
            "col2": np.array([np.nan, 1, np.nan, 1]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


###############################################
#     Tests for TemporalNullValueAnalyzer     #
###############################################


def test_temporal_null_value_analyzer_str() -> None:
    assert str(TemporalNullValueAnalyzer(dt_column="datetime", period="M")).startswith(
        "TemporalNullValueAnalyzer("
    )


def test_temporal_null_value_analyzer_frame(dataframe: DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert_frame_equal(section.frame, dataframe)


def test_temporal_null_value_analyzer_columns(dataframe: DataFrame) -> None:
    section = TemporalNullValueAnalyzer(
        dt_column="datetime", period="M", columns=("col1",)
    ).analyze(dataframe)
    assert section.columns == ("col1",)


def test_temporal_null_value_analyzer_columns_none(dataframe: DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert section.columns == ("col1", "col2")


@pytest.mark.parametrize("dt_column", ["datetime", "date"])
def test_temporal_null_value_analyzer_dt_column(dataframe: pd.DataFrame, dt_column: str) -> None:
    dataframe["date"] = pd.to_datetime(["2021-01-03", "2021-02-03", "2021-03-03", "2021-04-03"])
    section = TemporalNullValueAnalyzer(dt_column=dt_column, period="M").analyze(dataframe)
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["M", "D"])
def test_temporal_null_value_analyzer_period(dataframe: DataFrame, period: str) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period=period).analyze(dataframe)
    assert section.period == period


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_temporal_null_value_analyzer_figsize(
    dataframe: DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M", figsize=figsize).analyze(
        dataframe
    )
    assert section.figsize == figsize


def test_temporal_null_value_analyzer_figsize_default(dataframe: DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert section.figsize is None


def test_temporal_null_value_analyzer_get_statistics(dataframe: DataFrame) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert isinstance(section, TemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_null_value_analyzer_get_statistics_empty_rows() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        DataFrame({"col": [], "datetime": []})
    )
    assert isinstance(section, TemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_null_value_analyzer_get_statistics_missing_dt_column() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        DataFrame({"col": []})
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
