from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal
from pandas.testing import assert_frame_equal

from flamme.analyzer import ColumnTemporalNullValueAnalyzer
from flamme.section import ColumnTemporalNullValueSection, EmptySection


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "float": np.array([1.2, 4.2, np.nan, 2.2]),
            "int": np.array([np.nan, 1, 0, 1]),
            "str": np.array(["A", "B", None, np.nan]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


#####################################################
#     Tests for ColumnTemporalNullValueAnalyzer     #
#####################################################


def test_column_temporal_null_value_analyzer_str() -> None:
    assert str(ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M")).startswith(
        "ColumnTemporalNullValueAnalyzer("
    )


def test_column_temporal_null_value_analyzer_frame(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert_frame_equal(section.frame, dataframe)


def test_column_temporal_null_value_analyzer_columns_none(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert section.columns == ("float", "int", "str")
    assert section.ncols == 2


def test_column_temporal_null_value_analyzer_columns_1(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", columns=["float"]
    ).analyze(dataframe)
    assert section.columns == ("float",)
    assert section.ncols == 1


def test_column_temporal_null_value_analyzer_columns(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", columns=["float", "int", "str"]
    ).analyze(dataframe)
    assert section.columns == ("float", "int", "str")
    assert section.ncols == 2


@pytest.mark.parametrize("dt_column", ["datetime", "str"])
def test_column_temporal_null_value_analyzer_dt_column(
    dataframe: pd.DataFrame, dt_column: str
) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column=dt_column, period="M").analyze(dataframe)
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["M", "D"])
def test_column_temporal_null_value_analyzer_period(dataframe: pd.DataFrame, period: str) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period=period).analyze(
        dataframe
    )
    assert section.period == period


@pytest.mark.parametrize("ncols", [1, 2])
def test_column_temporal_null_value_analyzer_ncols(dataframe: pd.DataFrame, ncols: int) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", ncols=ncols
    ).analyze(dataframe)
    assert section.ncols == ncols


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_null_value_analyzer_figsize(
    dataframe: pd.DataFrame, figsize: tuple[int, int]
) -> None:
    section = ColumnTemporalNullValueAnalyzer(
        dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert section.figsize == figsize


def test_column_temporal_null_value_analyzer_get_statistics(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(dataframe)
    assert isinstance(section, ColumnTemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_null_value_analyzer_get_statistics_empty() -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        pd.DataFrame({"float": [], "int": [], "str": [], "datetime": []})
    )
    assert isinstance(section, ColumnTemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_null_value_analyzer_get_statistics_missing_empty_column() -> None:
    section = ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        pd.DataFrame({})
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
