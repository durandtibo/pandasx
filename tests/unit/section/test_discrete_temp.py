from __future__ import annotations

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import polars as pl
import pytest
from coola import objects_are_equal
from jinja2 import Template

from flamme.section import ColumnTemporalDiscreteSection
from flamme.section.discrete_temp import create_section_template, create_temporal_figure
from flamme.utils.data import datetime_range


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col": [1, 42, None, 42],
            "col2": [1.2, 4.2, 4.2, 1.2],
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col": pl.Int64,
            "col2": pl.Float64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


###################################################
#     Tests for ColumnTemporalDiscreteSection     #
###################################################


def test_column_temporal_discrete_section_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ColumnTemporalDiscreteSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="1mo",
        )
    ).startswith("ColumnTemporalDiscreteSection(")


def test_column_temporal_discrete_section_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert section.column == "col"


def test_column_temporal_discrete_section_dt_column(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert section.dt_column == "datetime"


def test_column_temporal_discrete_section_period(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert section.period == "1mo"


def test_column_temporal_discrete_section_figsize_default(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnTemporalDiscreteSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="1mo",
        ).figsize
        is None
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_discrete_section_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        ColumnTemporalDiscreteSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_temporal_discrete_section_get_statistics(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_get_statistics_empty_row() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pl.DataFrame(
            {"col": [], "datetime": []},
            schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        ),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_get_statistics_empty_column() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pl.DataFrame({}),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_empty_row() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pl.DataFrame(
            {"col": [], "datetime": []},
            schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        ),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_empty_column() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pl.DataFrame({}),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_args(
    dataframe: pl.DataFrame,
) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_discrete_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_discrete_section_render_html_toc_args(
    dataframe: pl.DataFrame,
) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)


###########################################
#    Tests for create_temporal_figure     #
###########################################


@pytest.mark.parametrize("column", ["col", "col2"])
def test_create_temporal_figure(dataframe: pl.DataFrame, column: str) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column=column,
            dt_column="datetime",
            period="1mo",
        ),
        plt.Figure,
    )


def test_create_temporal_figure_20_values() -> None:
    period = "1h"
    assert isinstance(
        create_temporal_figure(
            frame=pl.DataFrame(
                {
                    "col": list(range(20)),
                    "datetime": datetime_range(
                        start=datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
                        periods=20,
                        interval="1h",
                        eager=True,
                    ),
                },
                schema={"col": pl.Int64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
            ),
            column="col",
            dt_column="datetime",
            period=period,
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_temporal_figure_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="1mo",
            figsize=figsize,
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("proportion", [True, False])
def test_create_temporal_figure_proportion(dataframe: pl.DataFrame, proportion: bool) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="1mo",
            proportion=proportion,
        ),
        plt.Figure,
    )


def test_create_temporal_figure_empty() -> None:
    assert (
        create_temporal_figure(
            frame=pl.DataFrame({}),
            column="col",
            dt_column="datetime",
            period="1mo",
        )
        is None
    )


def test_create_temporal_figure_missing_column() -> None:
    assert (
        create_temporal_figure(
            frame=pl.DataFrame(
                {
                    "datetime": [
                        datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                        datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                        datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                        datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                    ],
                },
                schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
            ),
            column="col",
            dt_column="datetime",
            period="1mo",
        )
        is None
    )


def test_create_temporal_figure_missing_dt_column() -> None:
    assert (
        create_temporal_figure(
            frame=pl.DataFrame({"col": [1, 2, 3, 4, 5]}, schema={"col": pl.Int64}),
            column="col",
            dt_column="datetime",
            period="1mo",
        )
        is None
    )
