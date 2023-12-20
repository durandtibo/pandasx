from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import mark, raises

from flamme.section import NullValueSection, TemporalNullValueSection
from flamme.section.null import create_temporal_null_figure, prepare_data

######################################
#     Tests for NullValueSection     #
######################################


def test_null_value_section_incorrect_null_count_size() -> None:
    with raises(RuntimeError, match=r"columns \(3\) and null_count \(2\) do not match"):
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1]),
            total_count=np.array([5, 5, 5]),
        )


def test_null_value_section_incorrect_total_count_size() -> None:
    with raises(RuntimeError, match=r"columns \(3\) and total_count \(2\) do not match"):
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5]),
        )


def test_null_value_section_columns() -> None:
    assert NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    ).columns == ("col1", "col2", "col3")


def test_null_value_section_null_count() -> None:
    assert objects_are_equal(
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5, 5]),
        ).null_count,
        np.array([0, 1, 2]),
    )


def test_null_value_section_total_count() -> None:
    assert objects_are_equal(
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5, 5]),
        ).total_count,
        np.array([5, 5, 5]),
    )


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_null_value_section_figsize(figsize: tuple[float, float]) -> None:
    assert (
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5, 5]),
            figsize=figsize,
        ).figsize
        == figsize
    )


def test_null_value_section_figsize_default() -> None:
    assert (
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5, 5]),
        ).figsize
        is None
    )


def test_null_value_section_get_statistics() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (0, 1, 2),
            "total_count": (5, 5, 5),
        },
    )


def test_null_value_section_get_statistics_empty_row() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 0, 0]),
        total_count=np.array([0, 0, 0]),
    )
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_null_value_section_get_statistics_empty_column() -> None:
    section = NullValueSection(columns=[], null_count=np.array([]), total_count=np.array([]))
    assert objects_are_allclose(
        section.get_statistics(),
        {"columns": (), "null_count": (), "total_count": ()},
    )


def test_null_value_section_render_html_body() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_null_value_section_render_html_body_args() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_null_value_section_render_html_body_empty() -> None:
    section = NullValueSection(
        columns=[],
        null_count=np.array([]),
        total_count=np.array([]),
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_null_value_section_render_html_toc() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_null_value_section_render_html_toc_args() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


##############################################
#     Tests for TemporalNullValueSection     #
##############################################


def test_temporal_null_value_section_df() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
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
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
    )


@mark.parametrize("dt_column", ("datetime", "str"))
def test_temporal_null_value_section_dt_column(dt_column: str) -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column=dt_column,
        period="M",
    )
    assert section.dt_column == dt_column


@mark.parametrize("period", ("M", "D"))
def test_temporal_null_value_section_period(period: str) -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period=period,
    )
    assert section.period == period


@mark.parametrize("ncols", (1, 2))
def test_temporal_null_value_section_ncols(ncols: int) -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
        ncols=ncols,
    )
    assert section.ncols == ncols


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_temporal_null_value_section_figsize(figsize: tuple[float, float]) -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
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


def test_temporal_null_value_section_figsize_default() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert section.figsize == (7, 5)


def test_temporal_null_value_section_get_statistics() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_get_statistics_empty_row() -> None:
    section = TemporalNullValueSection(
        df=DataFrame({"float": [], "int": [], "str": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_get_statistics_only_datetime_column() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_render_html_body() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_null_value_section_render_html_body_args() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
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


def test_temporal_null_value_section_render_html_body_empty() -> None:
    section = TemporalNullValueSection(
        df=DataFrame({"float": [], "int": [], "str": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_null_value_section_render_html_toc() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_temporal_null_value_section_render_html_toc_args() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
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


#################################################
#     Tests for create_temporal_null_figure     #
#################################################


def test_create_temporal_null_figure() -> None:
    assert isinstance(
        create_temporal_null_figure(
            df=DataFrame(
                {
                    "float": np.array([1.2, 4.2, np.nan, 2.2]),
                    "int": np.array([np.nan, 1, 0, 1]),
                    "str": np.array(["A", "B", None, np.nan]),
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


@mark.parametrize("ncols", (1, 2))
def test_create_temporal_null_figure_ncols(ncols: int) -> None:
    assert isinstance(
        create_temporal_null_figure(
            df=DataFrame(
                {
                    "float": np.array([1.2, 4.2, np.nan, 2.2]),
                    "int": np.array([np.nan, 1, 0, 1]),
                    "str": np.array(["A", "B", None, np.nan]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            dt_column="datetime",
            period="M",
            ncols=ncols,
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
                    "col": np.array([1.2, 4.2, np.nan, 2.2]),
                    "datetime": pd.to_datetime(
                        ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                    ),
                }
            ),
            column="col",
            dt_column="datetime",
            period="M",
        ),
        (
            np.array([0, 0, 1, 0]),
            np.array([1, 1, 1, 1]),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_prepare_data_empty() -> None:
    assert objects_are_equal(
        prepare_data(
            df=DataFrame({"col": [], "datetime": pd.to_datetime([])}),
            column="col",
            dt_column="datetime",
            period="M",
        ),
        (
            np.array([], dtype=int),
            np.array([], dtype=int),
            [],
        ),
    )
