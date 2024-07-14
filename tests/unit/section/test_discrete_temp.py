from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal
from jinja2 import Template

from flamme.section import ColumnTemporalDiscreteSection
from flamme.section.discrete_temp import create_section_template, create_temporal_figure


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col": np.array([1, 42, np.nan, 22]),
            "col2": ["a", "b", 1, "a"],
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


###################################################
#     Tests for ColumnTemporalDiscreteSection     #
###################################################


def test_column_temporal_discrete_section_str(dataframe: pd.DataFrame) -> None:
    assert str(
        ColumnTemporalDiscreteSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        )
    ).startswith("ColumnTemporalDiscreteSection(")


def test_column_temporal_discrete_section_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.column == "col"


def test_column_temporal_discrete_section_dt_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.dt_column == "datetime"


def test_column_temporal_discrete_section_period(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.period == "M"


def test_column_temporal_discrete_section_figsize_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnTemporalDiscreteSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ).figsize
        is None
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_discrete_section_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        ColumnTemporalDiscreteSection(
            frame=dataframe, column="col", dt_column="datetime", period="M", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_temporal_discrete_section_get_statistics(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_get_statistics_empty_row() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pd.DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_get_statistics_empty_column() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pd.DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_section_render_html_body(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_empty_row() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pd.DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_empty_column() -> None:
    section = ColumnTemporalDiscreteSection(
        frame=pd.DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_discrete_section_render_html_body_args(
    dataframe: pd.DataFrame,
) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_discrete_section_render_html_toc(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_discrete_section_render_html_toc_args(
    dataframe: pd.DataFrame,
) -> None:
    section = ColumnTemporalDiscreteSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
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
def test_create_temporal_figure(dataframe: pd.DataFrame, column: str) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column=column,
            dt_column="datetime",
            period="M",
        ),
        plt.Figure,
    )


def test_create_temporal_figure_20_values() -> None:
    assert isinstance(
        create_temporal_figure(
            frame=pd.DataFrame(
                {
                    "col": np.arange(20),
                    "datetime": pd.date_range(start="2020-01-03", periods=20),
                }
            ),
            column="col",
            dt_column="datetime",
            period="M",
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_temporal_figure_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
            figsize=figsize,
        ),
        plt.Figure,
    )


def test_create_temporal_figure_empty() -> None:
    assert (
        create_temporal_figure(
            frame=pd.DataFrame({}),
            column="col",
            dt_column="datetime",
            period="M",
        )
        is None
    )


def test_create_temporal_figure_missing_column() -> None:
    assert (
        create_temporal_figure(
            frame=pd.DataFrame({"datetime": [1, 2, 3, 4, 5]}),
            column="col",
            dt_column="datetime",
            period="M",
        )
        is None
    )


def test_create_temporal_figure_missing_dt_column() -> None:
    assert (
        create_temporal_figure(
            frame=pd.DataFrame({"col": [1, 2, 3, 4, 5]}),
            column="col",
            dt_column="datetime",
            period="M",
        )
        is None
    )
