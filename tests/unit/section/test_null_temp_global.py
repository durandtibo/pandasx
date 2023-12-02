from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import mark, raises

from flamme.section import GlobalTemporalNullValueSection
from flamme.section.null_temp_global import (
    create_temporal_null_figure,
    create_temporal_null_table,
    prepare_data,
)

####################################################
#     Tests for GlobalTemporalNullValueSection     #
####################################################


def test_column_temporal_null_value_section_df() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert_frame_equal(
        section.df,
        DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
    )


@mark.parametrize("dt_column", ("datetime", "date"))
def test_column_temporal_null_value_section_dt_column(dt_column: str) -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
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
def test_column_temporal_null_value_section_period(period: str) -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period=period,
    )
    assert section.period == period


@mark.parametrize("figsize", ((700, 300), (100, 100)))
def test_column_temporal_null_value_section_figsize(figsize: tuple[int, int]) -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
        figsize=figsize,
    )
    assert section.figsize == figsize


def test_column_temporal_null_value_section_figsize_default() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert section.figsize == (None, None)


def test_column_temporal_null_value_section_missing_dt_column() -> None:
    with raises(
        ValueError, match=r"Datetime column my_datetime is not in the DataFrame \(columns:"
    ):
        GlobalTemporalNullValueSection(
            df=DataFrame(
                {
                    "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                    "col2": np.array([np.nan, 1, np.nan, 1]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            dt_column="my_datetime",
            period="M",
        )


def test_column_temporal_null_value_section_get_statistics() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_null_value_section_get_statistics_empty_row() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_null_value_section_render_html_body() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_body_args() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_null_value_section_render_html_body_empty_rows() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_toc() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_null_value_section_render_html_toc_args() -> None:
    section = GlobalTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


################################################
#    Tests for create_temporal_null_figure     #
################################################


def test_create_temporal_null_figure() -> None:
    assert isinstance(
        create_temporal_null_figure(
            df=DataFrame(
                {
                    "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                    "col2": np.array([np.nan, 1, np.nan, 1]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            dt_column="datetime",
            period="M",
        ),
        str,
    )


def test_create_temporal_null_figure_empty() -> None:
    assert isinstance(
        create_temporal_null_figure(
            df=DataFrame({"col1": [], "col2": [], "datetime": pd.to_datetime([])}),
            dt_column="datetime",
            period="M",
        ),
        str,
    )


###############################################
#    Tests for create_temporal_null_table     #
###############################################


def test_create_temporal_null_table() -> None:
    assert isinstance(
        create_temporal_null_table(
            df=DataFrame(
                {
                    "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                    "col2": np.array([np.nan, 1, np.nan, 1]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            dt_column="datetime",
            period="M",
        ),
        str,
    )


def test_create_temporal_null_table_empty() -> None:
    assert isinstance(
        create_temporal_null_table(
            df=DataFrame({"col1": [], "col2": [], "datetime": pd.to_datetime([])}),
            dt_column="datetime",
            period="M",
        ),
        str,
    )


#################################
#    Tests for prepare_data     #
#################################


def test_prepare_data() -> None:
    assert objects_are_equal(
        prepare_data(
            df=DataFrame(
                {
                    "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                    "col2": np.array([np.nan, 1, np.nan, 1]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            dt_column="datetime",
            period="M",
        ),
        (
            np.array([1, 0, 2, 0]),
            np.array([2, 2, 2, 2]),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_prepare_data_empty() -> None:
    assert objects_are_equal(
        prepare_data(
            df=DataFrame({"col1": [], "col2": [], "datetime": pd.to_datetime([])}),
            dt_column="datetime",
            period="M",
        ),
        (
            np.array([], dtype=int),
            np.array([], dtype=int),
            [],
        ),
    )
