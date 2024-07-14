from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnTemporalDiscreteAnalyzer
from flamme.section import ColumnTemporalDiscreteSection, EmptySection


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col": [1, 42, None, 22],
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


####################################################
#     Tests for ColumnTemporalDiscreteAnalyzer     #
####################################################


def test_column_temporal_discrete_analyzer_str() -> None:
    assert str(
        ColumnTemporalDiscreteAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnTemporalDiscreteAnalyzer(")


def test_column_temporal_discrete_analyzer_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.column == "col"


def test_column_temporal_discrete_analyzer_dt_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.dt_column == "datetime"


def test_column_temporal_discrete_analyzer_period(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.period == "M"


def test_column_temporal_discrete_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_discrete_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.figsize == figsize


def test_column_temporal_discrete_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_empty() -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(
        pl.DataFrame(
            {"col": [], "datetime": []},
            schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        )
    )
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_missing_column() -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(
        pl.DataFrame(
            {"datetime": []}, schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")}
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_missing_dt_column() -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pl.DataFrame({"col": []}, schema={"col": pl.Int64}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_same_column() -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="datetime", dt_column="datetime", period="M"
    ).analyze(
        pl.DataFrame(
            {
                "datetime": [
                    datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                ]
            },
            schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
