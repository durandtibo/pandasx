from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnContinuousTemporalDriftAnalyzer
from flamme.section import ColumnContinuousTemporalDriftSection, EmptySection

###########################################################
#     Tests for ColumnContinuousTemporalDriftAnalyzer     #
###########################################################


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col": np.array([1.2, 4.2, np.nan, 2.2]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


def test_column_continuous_temporal_drift_analyzer_repr() -> None:
    assert repr(
        ColumnContinuousTemporalDriftAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnContinuousTemporalDriftAnalyzer(")


def test_column_continuous_temporal_drift_analyzer_str() -> None:
    assert str(
        ColumnContinuousTemporalDriftAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnContinuousTemporalDriftAnalyzer(")


def test_column_continuous_temporal_drift_analyzer_column(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.column == "col"


def test_column_continuous_temporal_drift_analyzer_dt_column(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.dt_column == "datetime"


def test_column_continuous_temporal_drift_analyzer_period(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.period == "M"


def test_column_continuous_temporal_drift_analyzer_nbins_default(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.nbins is None


@pytest.mark.parametrize("nbins", [1, 10])
def test_column_continuous_temporal_drift_analyzer_nbins(
    dataframe: pd.DataFrame, nbins: int
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M", nbins=nbins
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.nbins == nbins


def test_column_continuous_temporal_drift_analyzer_density_default(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert not section.density


@pytest.mark.parametrize("density", [True, False])
def test_column_continuous_temporal_drift_analyzer_density(
    dataframe: pd.DataFrame, density: bool
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M", density=density
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.density == density


def test_column_continuous_temporal_drift_analyzer_yscale_default(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_continuous_temporal_drift_analyzer_yscale(
    dataframe: pd.DataFrame, yscale: str
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M", yscale=yscale
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.yscale == yscale


def test_column_continuous_temporal_drift_analyzer_xmin_default(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmin is None


@pytest.mark.parametrize("xmin", [1.0, "q0.1"])
def test_column_continuous_temporal_drift_analyzer_xmin(
    dataframe: pd.DataFrame, xmin: str | float
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M", xmin=xmin
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmin == xmin


def test_column_continuous_temporal_drift_analyzer_xmax_default(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmax is None


@pytest.mark.parametrize("xmax", [5.0, "q0.9"])
def test_column_continuous_temporal_drift_analyzer_xmax(
    dataframe: pd.DataFrame, xmax: str | float
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M", xmax=xmax
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmax == xmax


def test_column_continuous_temporal_drift_analyzer_figsize_default(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col",
        dt_column="datetime",
        period="M",
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_continuous_temporal_drift_analyzer_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.figsize == figsize


def test_column_continuous_temporal_drift_analyzer_analyze(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_empty() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": [], "int": [], "str": [], "datetime": []}))
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_missing_column() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"datetime": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_missing_dt_column() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_same_column() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="col", period="M"
    ).analyze(
        pd.DataFrame(
            {"col": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"])}
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
