from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pytest import fixture, mark

from flamme.section import ColumnTemporalDiscreteSection
from flamme.section.discrete_temporal import create_temporal_figure


@fixture
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col": np.array([1, 42, np.nan, 22]),
            "col2": ["a", "b", 1, "a"],
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


###################################################
#     Tests for ColumnTemporalDiscreteSection     #
###################################################


def test_column_temporal_discrete_section_column(dataframe: DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.column == "col"


def test_column_temporal_discrete_section_dt_column(dataframe: DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.dt_column == "datetime"


def test_column_temporal_discrete_section_period(dataframe: DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.period == "M"


def test_column_temporal_discrete_section_figsize_default(dataframe: DataFrame) -> None:
    assert (
        ColumnTemporalDiscreteSection(
            df=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ).figsize
        is None
    )


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_column_temporal_discrete_section_figsize(
    dataframe: DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        ColumnTemporalDiscreteSection(
            df=dataframe, column="col", dt_column="datetime", period="M", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_temporal_discrete_section_get_statistics(dataframe: DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_get_statistics_empty_row() -> None:
    section = ColumnTemporalDiscreteSection(
        df=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_get_statistics_empty_column() -> None:
    section = ColumnTemporalDiscreteSection(
        df=DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_render_html_body(dataframe: DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_empty_row() -> None:
    section = ColumnTemporalDiscreteSection(
        df=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_empty_column() -> None:
    section = ColumnTemporalDiscreteSection(
        df=DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_args(
    dataframe: DataFrame,
) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_discrete_section_render_html_toc(dataframe: DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_discrete_section_render_html_toc_args(
    dataframe: DataFrame,
) -> None:
    section = ColumnTemporalDiscreteSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


###########################################
#    Tests for create_temporal_figure     #
###########################################


@mark.parametrize("column", ["col", "col2"])
def test_create_temporal_figure(dataframe: DataFrame, column: str) -> None:
    assert isinstance(
        create_temporal_figure(
            df=dataframe,
            column=column,
            dt_column="datetime",
            period="M",
        ),
        str,
    )


def test_create_temporal_figure_20_values() -> None:
    assert isinstance(
        create_temporal_figure(
            df=DataFrame(
                {
                    "col": np.arange(20),
                    "datetime": pd.date_range(start="2020-01-03", periods=20),
                }
            ),
            column="col",
            dt_column="datetime",
            period="M",
        ),
        str,
    )


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_create_temporal_figure_figsize(dataframe: DataFrame, figsize: tuple[float, float]) -> None:
    assert isinstance(
        create_temporal_figure(
            df=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ),
        str,
    )
