from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal
from jinja2 import Template
from pandas import DataFrame

from flamme.section import ColumnTemporalContinuousSection
from flamme.section.continuous_temporal import create_temporal_figure


@pytest.fixture()
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col": np.array([1.2, 4.2, np.nan, 2.2]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


#####################################################
#     Tests for ColumnTemporalContinuousSection     #
#####################################################


def test_column_temporal_continuous_section_column(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.column == "col"


def test_column_temporal_continuous_section_dt_column(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.dt_column == "datetime"


def test_column_temporal_continuous_section_yscale_default(dataframe: DataFrame) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ).yscale
        == "auto"
    )


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_temporal_continuous_section_yscale(dataframe: DataFrame, yscale: str) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
            yscale=yscale,
        ).yscale
        == yscale
    )


def test_column_temporal_continuous_section_period(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.period == "M"


def test_column_temporal_continuous_section_figsize_default(dataframe: DataFrame) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ).figsize
        is None
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_continuous_section_figsize(
    dataframe: DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe, column="col", dt_column="datetime", period="M", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_temporal_continuous_section_get_statistics(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_section_get_statistics_empty_row() -> None:
    section = ColumnTemporalContinuousSection(
        frame=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_section_get_statistics_empty_column() -> None:
    section = ColumnTemporalContinuousSection(
        frame=DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_section_render_html_body(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_continuous_section_render_html_body_empty_row() -> None:
    section = ColumnTemporalContinuousSection(
        frame=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_continuous_section_render_html_body_empty_column() -> None:
    section = ColumnTemporalContinuousSection(
        frame=DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_continuous_section_render_html_body_args(
    dataframe: DataFrame,
) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_continuous_section_render_html_toc(dataframe: DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_continuous_section_render_html_toc_args(
    dataframe: DataFrame,
) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
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


def test_create_temporal_figure(dataframe: DataFrame) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ),
        str,
    )


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_create_temporal_figure_yscale(dataframe: DataFrame, yscale: str) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
            yscale=yscale,
        ),
        str,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_temporal_figure_figsize(dataframe: DataFrame, figsize: tuple[float, float]) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
            figsize=figsize,
        ),
        str,
    )
