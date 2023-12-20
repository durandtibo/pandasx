from __future__ import annotations

import math

import numpy as np
import pandas as pd
from coola import objects_are_allclose, objects_are_equal
from pandas import DataFrame, Series
from pandas.testing import assert_series_equal
from pytest import fixture, mark

from flamme.analyzer import ColumnContinuousAnalyzer, ColumnTemporalContinuousAnalyzer
from flamme.section import (
    ColumnContinuousSection,
    ColumnTemporalContinuousSection,
    EmptySection,
)
from tests.unit.section.test_continous import STATS_KEYS

##############################################
#     Tests for ColumnContinuousAnalyzer     #
##############################################


def test_column_continuous_analyzer_str() -> None:
    assert str(ColumnContinuousAnalyzer(column="col")).startswith("ColumnContinuousAnalyzer(")


def test_column_continuous_analyzer_series() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert_series_equal(section.series, Series([np.nan] + list(range(101)) + [np.nan], name="col"))


def test_column_continuous_analyzer_column() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.column == "col"


def test_column_continuous_analyzer_nbins_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.nbins is None


@mark.parametrize("nbins", (1, 2, 4))
def test_column_continuous_analyzer_nbins(nbins: int) -> None:
    section = ColumnContinuousAnalyzer(column="col", nbins=nbins).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.nbins == nbins


def test_column_continuous_analyzer_log_y_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert not section.log_y


@mark.parametrize("log_y", (True, False))
def test_column_continuous_analyzer_log_y(log_y: bool) -> None:
    section = ColumnContinuousAnalyzer(column="col", log_y=log_y).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.log_y == log_y


def test_column_continuous_analyzer_xmin_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmin == "q0"


@mark.parametrize("xmin", (1.0, "q0.1", None))
def test_column_continuous_analyzer_xmin(xmin: float | str | None) -> None:
    section = ColumnContinuousAnalyzer(column="col", xmin=xmin).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmin == xmin


def test_column_continuous_analyzer_xmax_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmax == "q1"


@mark.parametrize("xmax", (1.0, "q0.1", None))
def test_column_continuous_analyzer_xmax(xmax: float | str | None) -> None:
    section = ColumnContinuousAnalyzer(column="col", xmax=xmax).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmax == xmax


def test_column_continuous_analyzer_figsize_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.figsize is None


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_column_continuous_analyzer_figsize(figsize: tuple[float, float]) -> None:
    section = ColumnContinuousAnalyzer(column="col", figsize=figsize).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.figsize == figsize


def test_column_continuous_analyzer_get_statistics() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "count": 103,
            "num_nulls": 2,
            "num_non_nulls": 101,
            "nunique": 102,
            "mean": 50.0,
            "median": 50.0,
            "min": 0.0,
            "max": 100.0,
            "std": 29.300170647967224,
            "q01": 1.0,
            "q05": 5.0,
            "q10": 10.0,
            "q25": 25.0,
            "q75": 75.0,
            "q90": 90.0,
            "q95": 95.0,
            "q99": 99.0,
        },
    )


def test_column_continuous_analyzer_get_statistics_empty() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(DataFrame({"col": []}))
    assert isinstance(section, ColumnContinuousSection)
    stats = section.get_statistics()
    assert len(stats) == 17
    assert stats["count"] == 0
    assert stats["num_nulls"] == 0
    assert stats["num_non_nulls"] == 0
    assert stats["nunique"] == 0
    for key in STATS_KEYS:
        assert math.isnan(stats[key])


def test_column_continuous_analyzer_get_statistics_missing_column() -> None:
    section = ColumnContinuousAnalyzer(column="col2").analyze(DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


######################################################
#     Tests for ColumnTemporalContinuousAnalyzer     #
######################################################


@fixture
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col": np.array([1.2, 4.2, np.nan, 2.2]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


def test_column_temporal_continuous_analyzer_str() -> None:
    assert str(
        ColumnTemporalContinuousAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnTemporalContinuousAnalyzer(")


def test_column_temporal_continuous_analyzer_column(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.column == "col"


def test_column_temporal_continuous_analyzer_dt_column(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.dt_column == "datetime"


def test_column_temporal_continuous_analyzer_period(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.period == "M"


def test_column_temporal_continuous_analyzer_log_y_default(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert not section.log_y


@mark.parametrize("log_y", (True, False))
def test_column_temporal_continuous_analyzer_log_y(dataframe: DataFrame, log_y: bool) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M", log_y=log_y
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.log_y == log_y


def test_column_temporal_continuous_analyzer_figsize_default(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col",
        dt_column="datetime",
        period="M",
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize is None


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_column_temporal_continuous_analyzer_figsize(
    dataframe: DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize == figsize


def test_column_temporal_continuous_analyzer_get_statistics(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_get_statistics_empty() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(DataFrame({"col": [], "int": [], "str": [], "datetime": []}))
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_get_statistics_missing_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(DataFrame({"datetime": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_get_statistics_missing_dt_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
