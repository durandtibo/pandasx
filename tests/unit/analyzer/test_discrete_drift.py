from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnTemporalDriftDiscreteAnalyzer
from flamme.section import ColumnTemporalDriftDiscreteSection, EmptySection
from flamme.utils.data import datetime_range


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    rng = np.random.default_rng()
    return pl.DataFrame(
        {
            "col": rng.integers(low=0, high=11, size=(100,)),
            "datetime": datetime_range(
                start=datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
                periods=100,
                interval="1d",
                eager=True,
            ),
        },
        schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


#########################################################
#     Tests for ColumnTemporalDriftDiscreteAnalyzer     #
#########################################################


def test_column_temporal_drift_discrete_analyzer_str() -> None:
    assert str(
        ColumnTemporalDriftDiscreteAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnTemporalDriftDiscreteAnalyzer(")


def test_column_temporal_drift_discrete_analyzer_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert section.column == "col"


def test_column_temporal_drift_discrete_analyzer_dt_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert section.dt_column == "datetime"


def test_column_temporal_drift_discrete_analyzer_period(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert section.period == "M"


def test_column_temporal_drift_discrete_analyzer_proportion_default(
    dataframe: pl.DataFrame,
) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert not section.proportion


@pytest.mark.parametrize("proportion", [True, False])
def test_column_temporal_drift_discrete_analyzer_proportion(
    dataframe: pl.DataFrame, proportion: bool
) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M", proportion=proportion
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert section.proportion == proportion


def test_column_temporal_drift_discrete_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_drift_discrete_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert section.figsize == figsize


def test_column_temporal_drift_discrete_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_drift_discrete_analyzer_analyze_empty() -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(
        pl.DataFrame(
            {"col": [], "datetime": []},
            schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        )
    )
    assert isinstance(section, ColumnTemporalDriftDiscreteSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_drift_discrete_analyzer_analyze_missing_column() -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(
        pl.DataFrame(
            {"datetime": []}, schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")}
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_drift_discrete_analyzer_analyze_missing_dt_column() -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pl.DataFrame({"col": []}, schema={"col": pl.Int64}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_drift_discrete_analyzer_analyze_same_column(
    dataframe: pl.DataFrame,
) -> None:
    section = ColumnTemporalDriftDiscreteAnalyzer(
        column="datetime", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
