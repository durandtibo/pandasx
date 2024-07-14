from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnTemporalContinuousAnalyzer
from flamme.section import ColumnTemporalContinuousSection, EmptySection

######################################################
#     Tests for ColumnTemporalContinuousAnalyzer     #
######################################################


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col": np.array([1.2, 4.2, np.nan, 2.2]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


def test_column_temporal_continuous_analyzer_str() -> None:
    assert str(
        ColumnTemporalContinuousAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnTemporalContinuousAnalyzer(")


def test_column_temporal_continuous_analyzer_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.column == "col"


def test_column_temporal_continuous_analyzer_dt_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.dt_column == "datetime"


def test_column_temporal_continuous_analyzer_period(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.period == "M"


def test_column_temporal_continuous_analyzer_yscale_default(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_temporal_continuous_analyzer_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M", yscale=yscale
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.yscale == yscale


def test_column_temporal_continuous_analyzer_figsize_default(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col",
        dt_column="datetime",
        period="M",
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_continuous_analyzer_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize == figsize


def test_column_temporal_continuous_analyzer_analyze(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_empty() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": [], "int": [], "str": [], "datetime": []}))
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_missing_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"datetime": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_missing_dt_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_analyze_same_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(column="col", dt_column="col", period="M").analyze(
        pd.DataFrame(
            {"col": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"])}
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
