from __future__ import annotations

import pandas as pd
import pytest
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from pandas.testing import assert_frame_equal

from flamme.section import TemporalRowCountSection
from flamme.section.count_rows import (
    create_temporal_count_figure,
    create_temporal_count_table,
    prepare_data,
)


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
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


def test_column_temporal_row_count_section_frame(dataframe: pd.DataFrame) -> None:
    section = TemporalRowCountSection(frame=dataframe, dt_column="datetime", period="M")
    assert_frame_equal(
        section.frame,
        pd.DataFrame(
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


@pytest.mark.parametrize("dt_column", ["datetime", "date"])
def test_column_temporal_row_count_section_dt_column(dt_column: str) -> None:
    section = TemporalRowCountSection(
        frame=pd.DataFrame(
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


@pytest.mark.parametrize("period", ["M", "D"])
def test_column_temporal_row_count_section_period(dataframe: pd.DataFrame, period: str) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period=period,
    )
    assert section.period == period


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_row_count_section_figsize(
    dataframe: pd.DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="M",
        figsize=figsize,
    )
    assert section.figsize == figsize


def test_column_temporal_row_count_section_figsize_default(dataframe: pd.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert section.figsize is None


def test_column_temporal_row_count_section_missing_dt_column(dataframe: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError, match=r"Datetime column my_datetime is not in the DataFrame \(columns:"
    ):
        TemporalRowCountSection(
            frame=dataframe,
            dt_column="my_datetime",
            period="M",
        )


def test_column_temporal_row_count_section_get_statistics(dataframe: pd.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_row_count_section_get_statistics_empty_row() -> None:
    section = TemporalRowCountSection(
        frame=pd.DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_row_count_section_render_html_body(dataframe: pd.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_row_count_section_render_html_body_args(dataframe: pd.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_row_count_section_render_html_body_empty_rows() -> None:
    section = TemporalRowCountSection(
        frame=pd.DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_row_count_section_render_html_toc(dataframe: pd.DataFrame) -> None:
    section = TemporalRowCountSection(frame=dataframe, dt_column="datetime", period="M")
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_row_count_section_render_html_toc_args(dataframe: pd.DataFrame) -> None:
    section = TemporalRowCountSection(frame=dataframe, dt_column="datetime", period="M")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


################################################
#    Tests for create_temporal_count_table     #
################################################


def test_create_temporal_count_table(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_count_table(
            frame=dataframe,
            dt_column="datetime",
            period="M",
        ),
        str,
    )


def test_create_temporal_count_table_empty() -> None:
    assert isinstance(
        create_temporal_count_table(
            frame=pd.DataFrame({"col1": [], "col2": [], "datetime": pd.to_datetime([])}),
            dt_column="datetime",
            period="M",
        ),
        str,
    )


#################################################
#    Tests for create_temporal_count_figure     #
#################################################


def test_create_temporal_count_figure(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_count_figure(
            frame=dataframe,
            dt_column="datetime",
            period="M",
        ),
        str,
    )


def test_create_temporal_count_figure_empty() -> None:
    assert isinstance(
        create_temporal_count_figure(
            frame=pd.DataFrame({"datetime": []}), dt_column="datetime", period="M"
        ),
        str,
    )


#################################
#    Tests for prepare_data     #
#################################


def test_prepare_data(dataframe: pd.DataFrame) -> None:
    assert objects_are_equal(
        prepare_data(
            frame=dataframe,
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
        prepare_data(frame=pd.DataFrame({"datetime": []}), dt_column="datetime", period="M"),
        ([], []),
    )
