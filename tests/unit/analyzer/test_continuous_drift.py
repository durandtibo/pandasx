from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnContinuousTemporalDriftAnalyzer
from flamme.section import ColumnContinuousTemporalDriftSection, EmptySection
from flamme.utils.data import datetime_range

###########################################################
#     Tests for ColumnContinuousTemporalDriftAnalyzer     #
###########################################################


@pytest.fixture
def dataframe() -> pl.DataFrame:
    rng = np.random.default_rng()
    return pl.DataFrame(
        {
            "col": rng.standard_normal(100),
            "datetime": datetime_range(
                start=datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
                periods=100,
                interval="1d",
                eager=True,
            ),
        },
        schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


def test_column_continuous_temporal_drift_analyzer_repr() -> None:
    assert repr(
        ColumnContinuousTemporalDriftAnalyzer(column="col", dt_column="datetime", period="1mo")
    ).startswith("ColumnContinuousTemporalDriftAnalyzer(")


def test_column_continuous_temporal_drift_analyzer_str() -> None:
    assert str(
        ColumnContinuousTemporalDriftAnalyzer(column="col", dt_column="datetime", period="1mo")
    ).startswith("ColumnContinuousTemporalDriftAnalyzer(")


def test_column_continuous_temporal_drift_analyzer_column(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.column == "col"


def test_column_continuous_temporal_drift_analyzer_dt_column(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.dt_column == "datetime"


def test_column_continuous_temporal_drift_analyzer_period(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.period == "1mo"


def test_column_continuous_temporal_drift_analyzer_nbins_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.nbins is None


@pytest.mark.parametrize("nbins", [1, 10])
def test_column_continuous_temporal_drift_analyzer_nbins(
    dataframe: pl.DataFrame, nbins: int
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo", nbins=nbins
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.nbins == nbins


def test_column_continuous_temporal_drift_analyzer_density_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert not section.density


@pytest.mark.parametrize("density", [True, False])
def test_column_continuous_temporal_drift_analyzer_density(
    dataframe: pl.DataFrame, density: bool
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo", density=density
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.density == density


def test_column_continuous_temporal_drift_analyzer_yscale_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_continuous_temporal_drift_analyzer_yscale(
    dataframe: pl.DataFrame, yscale: str
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo", yscale=yscale
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.yscale == yscale


def test_column_continuous_temporal_drift_analyzer_xmin_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmin is None


@pytest.mark.parametrize("xmin", [1.0, "q0.1"])
def test_column_continuous_temporal_drift_analyzer_xmin(
    dataframe: pl.DataFrame, xmin: str | float
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo", xmin=xmin
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmin == xmin


def test_column_continuous_temporal_drift_analyzer_xmax_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmax is None


@pytest.mark.parametrize("xmax", [5.0, "q0.9"])
def test_column_continuous_temporal_drift_analyzer_xmax(
    dataframe: pl.DataFrame, xmax: str | float
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo", xmax=xmax
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.xmax == xmax


def test_column_continuous_temporal_drift_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col",
        dt_column="datetime",
        period="1mo",
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_continuous_temporal_drift_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert section.figsize == figsize


def test_column_continuous_temporal_drift_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(dataframe)
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_empty() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(
        pl.DataFrame(
            {"col": [], "datetime": []},
            schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        )
    )
    assert isinstance(section, ColumnContinuousTemporalDriftSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_missing_column() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(
        pl.DataFrame(
            {"datetime": []}, schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")}
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_missing_dt_column() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="datetime", period="1mo"
    ).analyze(pl.DataFrame({"col": []}, schema={"col": pl.Int64}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_continuous_temporal_drift_analyzer_analyze_same_column() -> None:
    section = ColumnContinuousTemporalDriftAnalyzer(
        column="col", dt_column="col", period="1mo"
    ).analyze(
        pl.DataFrame(
            {
                "col": datetime_range(
                    start=datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
                    periods=100,
                    interval="1d",
                    eager=True,
                ),
            },
            schema={"col": pl.Datetime(time_unit="us", time_zone="UTC")},
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
