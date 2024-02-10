from __future__ import annotations

import pandas as pd
import pytest
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import mark, raises

from flamme.section import TemporalRowCountSection
from flamme.section.count_rows import (
    create_temporal_count_figure,
    create_temporal_count_table,
    prepare_data,
)


@pytest.fixture()
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-02-03",
                    "2020-03-03",
                    "2020-04-03",
                ]
            ),
        }
    )


#############################################
#     Tests for TemporalRowCountSection     #
#############################################


def test_column_temporal_row_count_section_df(dataframe: DataFrame) -> None:
    section = TemporalRowCountSection(df=dataframe, dt_column="datetime", period="M")
    assert_frame_equal(
        section.df,
        DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2020-01-03",
                        "2020-01-04",
                        "2020-01-05",
                        "2020-02-03",
                        "2020-03-03",
                        "2020-04-03",
                    ]
                ),
            }
        ),
    )


@mark.parametrize("dt_column", ("datetime", "date"))
def test_column_temporal_row_count_section_dt_column(dt_column: str) -> None:
    section = TemporalRowCountSection(
        df=DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
                "date": pd.to_datetime(["2021-01-03", "2021-02-03", "2021-03-03", "2021-04-03"]),
            }
        ),
        dt_column=dt_column,
        period="M",
    )
    assert section.dt_column == dt_column


@mark.parametrize("period", ("M", "D"))
def test_column_temporal_row_count_section_period(dataframe: DataFrame, period: str) -> None:
    section = TemporalRowCountSection(
        df=dataframe,
        dt_column="datetime",
        period=period,
    )
    assert section.period == period


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_column_temporal_row_count_section_figsize(
    dataframe: DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalRowCountSection(
        df=dataframe,
        dt_column="datetime",
        period="M",
        figsize=figsize,
    )
    assert section.figsize == figsize


def test_column_temporal_row_count_section_figsize_default(dataframe: DataFrame) -> None:
    section = TemporalRowCountSection(
        df=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert section.figsize is None


def test_column_temporal_row_count_section_missing_dt_column(dataframe: DataFrame) -> None:
    with raises(
        ValueError, match=r"Datetime column my_datetime is not in the DataFrame \(columns:"
    ):
        TemporalRowCountSection(
            df=dataframe,
            dt_column="my_datetime",
            period="M",
        )


def test_column_temporal_row_count_section_get_statistics(dataframe: DataFrame) -> None:
    section = TemporalRowCountSection(
        df=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_row_count_section_get_statistics_empty_row() -> None:
    section = TemporalRowCountSection(
        df=DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_row_count_section_render_html_body(dataframe: DataFrame) -> None:
    section = TemporalRowCountSection(
        df=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_row_count_section_render_html_body_args(dataframe: DataFrame) -> None:
    section = TemporalRowCountSection(
        df=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_row_count_section_render_html_body_empty_rows() -> None:
    section = TemporalRowCountSection(
        df=DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_row_count_section_render_html_toc(dataframe: DataFrame) -> None:
    section = TemporalRowCountSection(df=dataframe, dt_column="datetime", period="M")
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_row_count_section_render_html_toc_args(dataframe: DataFrame) -> None:
    section = TemporalRowCountSection(df=dataframe, dt_column="datetime", period="M")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


################################################
#    Tests for create_temporal_count_table     #
################################################


def test_create_temporal_count_table(dataframe: DataFrame) -> None:
    assert isinstance(
        create_temporal_count_table(
            df=dataframe,
            dt_column="datetime",
            period="M",
        ),
        str,
    )


def test_create_temporal_count_table_empty() -> None:
    assert isinstance(
        create_temporal_count_table(
            df=DataFrame({"col1": [], "col2": [], "datetime": pd.to_datetime([])}),
            dt_column="datetime",
            period="M",
        ),
        str,
    )


#################################################
#    Tests for create_temporal_count_figure     #
#################################################


def test_create_temporal_count_figure(dataframe: DataFrame) -> None:
    assert isinstance(
        create_temporal_count_figure(
            df=dataframe,
            dt_column="datetime",
            period="M",
        ),
        str,
    )


def test_create_temporal_count_figure_empty() -> None:
    assert isinstance(
        create_temporal_count_figure(
            df=DataFrame({"datetime": []}), dt_column="datetime", period="M"
        ),
        str,
    )


#################################
#    Tests for prepare_data     #
#################################


def test_prepare_data(dataframe: DataFrame) -> None:
    assert objects_are_equal(
        prepare_data(
            df=dataframe,
            dt_column="datetime",
            period="M",
        ),
        (
            [3, 1, 1, 1],
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_prepare_data_empty() -> None:
    assert objects_are_equal(
        prepare_data(df=DataFrame({"datetime": []}), dt_column="datetime", period="M"),
        ([], []),
        show_difference=True,
    )
