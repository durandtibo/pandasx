from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_allclose
from jinja2 import Template
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import mark, raises

from flamme.section import ColumnTemporalNullValueSection

####################################################
#     Tests for ColumnTemporalNullValueSection     #
####################################################


def test_column_temporal_null_value_section_df() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert_frame_equal(
        section.df,
        DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
    )


@mark.parametrize("column", ("col1", "col2"))
def test_column_temporal_null_value_section_column(column: str) -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, 0, 1]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column=column,
        dt_column="datetime",
        period="M",
    )
    assert section.column == column


@mark.parametrize("dt_column", ("datetime", "date"))
def test_column_temporal_null_value_section_dt_column(dt_column: str) -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
                "date": pd.to_datetime(["2021-01-03", "2021-02-03", "2021-03-03", "2021-04-03"]),
            }
        ),
        column="col",
        dt_column=dt_column,
        period="M",
    )
    assert section.dt_column == dt_column


@mark.parametrize("period", ("M", "D"))
def test_column_temporal_null_value_section_period(period: str) -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period=period,
    )
    assert section.period == period


@mark.parametrize("figsize", ((700, 300), (100, 100)))
def test_column_temporal_null_value_section_figsize(figsize: tuple[int, int]) -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period="M",
        figsize=figsize,
    )
    assert section.figsize == figsize


def test_column_temporal_null_value_section_missing_column() -> None:
    with raises(ValueError, match=r"Column my_col is not in the DataFrame \(columns:"):
        ColumnTemporalNullValueSection(
            df=DataFrame(
                {
                    "col": np.array([1.2, 4.2, np.nan, 2.2]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            column="my_col",
            dt_column="datetime",
            period="M",
        )


def test_column_temporal_null_value_section_missing_dt_column() -> None:
    with raises(
        ValueError, match=r"Datetime column my_datetime is not in the DataFrame \(columns:"
    ):
        ColumnTemporalNullValueSection(
            df=DataFrame(
                {
                    "col": np.array([1.2, 4.2, np.nan, 2.2]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            column="col",
            dt_column="my_datetime",
            period="M",
        )


def test_column_temporal_null_value_section_get_statistics() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_null_value_section_get_statistics_empty_row() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_null_value_section_render_html_body() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_body_args() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_null_value_section_render_html_body_empty_rows() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_toc() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_null_value_section_render_html_toc_args() -> None:
    section = ColumnTemporalNullValueSection(
        df=DataFrame(
            {
                "col": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
