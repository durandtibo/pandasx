from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from flamme.analyzer import NullValueAnalyzer, TemporalNullValueAnalyzer
from flamme.section import EmptySection, NullValueSection, TemporalNullValueSection

#######################################
#     Tests for NullValueAnalyzer     #
#######################################


def test_null_value_analyzer_str() -> None:
    assert str(NullValueAnalyzer()).startswith("NullValueAnalyzer(")


def test_null_value_analyzer_figsize() -> None:
    section = NullValueAnalyzer(figsize=(200, 100)).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, NullValueSection)
    assert section.figsize == (200, 100)


def test_null_value_analyzer_get_statistics() -> None:
    section = NullValueAnalyzer().analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, NullValueSection)
    assert section.columns == ("float", "int", "str")
    assert objects_are_equal(section.null_count, np.array([1, 1, 2]))
    assert objects_are_equal(section.total_count, np.array([4, 4, 4]))
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (1, 1, 2),
            "total_count": (4, 4, 4),
        },
    )


def test_null_value_analyzer_get_statistics_empty() -> None:
    section = NullValueAnalyzer().analyze(
        DataFrame({"float": np.array([]), "int": np.array([]), "str": np.array([])})
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_null_value_analyzer_get_statistics_empty_no_column() -> None:
    section = NullValueAnalyzer().analyze(DataFrame({}))
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"columns": (), "null_count": (), "total_count": ()},
    )


###############################################
#     Tests for TemporalNullValueAnalyzer     #
###############################################


def test_monthly_null_value_analyzer_str() -> None:
    assert str(TemporalNullValueAnalyzer(dt_column="datetime", period="M")).startswith(
        "TemporalNullValueAnalyzer("
    )


def test_monthly_null_value_analyzer_df() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        )
    )
    assert_frame_equal(
        section.df,
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
    )


@pytest.mark.parametrize("dt_column", ["datetime", "str"])
def test_monthly_null_value_analyzer_dt_column(dt_column: str) -> None:
    section = TemporalNullValueAnalyzer(dt_column=dt_column, period="M").analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        )
    )
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["M", "D"])
def test_monthly_null_value_analyzer_period(period: str) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period=period).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        )
    )
    assert section.period == period


@pytest.mark.parametrize("ncols", [1, 2])
def test_monthly_null_value_analyzer_ncols(ncols: int) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M", ncols=ncols).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        )
    )
    assert section.ncols == ncols


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_monthly_null_value_analyzer_figsize(figsize: tuple[int, int]) -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M", figsize=figsize).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        )
    )
    assert section.figsize == figsize


def test_monthly_null_value_analyzer_get_statistics() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        )
    )
    assert isinstance(section, TemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_monthly_null_value_analyzer_get_statistics_empty() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(
        DataFrame({"float": [], "int": [], "str": [], "datetime": []})
    )
    assert isinstance(section, TemporalNullValueSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_monthly_null_value_analyzer_get_statistics_missing_empty_column() -> None:
    section = TemporalNullValueAnalyzer(dt_column="datetime", period="M").analyze(DataFrame({}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
