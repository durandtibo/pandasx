from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from flamme.section import AllColumnsTemporalNullValueSection
from flamme.section.null_temp_all import (
    create_temporal_null_figure,
    create_temporal_null_figures,
    prepare_data,
)

########################################################
#     Tests for AllColumnsTemporalNullValueSection     #
########################################################


def test_all_columns_temporal_null_value_section_frame() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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
        section.frame,
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


@pytest.mark.parametrize("dt_column", ["datetime", "str"])
def test_all_columns_temporal_null_value_section_dt_column(dt_column: str) -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


@pytest.mark.parametrize("period", ["M", "D"])
def test_all_columns_temporal_null_value_section_period(period: str) -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


@pytest.mark.parametrize("ncols", [1, 2])
def test_all_columns_temporal_null_value_section_ncols(ncols: int) -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_all_columns_temporal_null_value_section_figsize(figsize: tuple[float, float]) -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


def test_all_columns_temporal_null_value_section_figsize_default() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


def test_all_columns_temporal_null_value_section_get_statistics() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


def test_all_columns_temporal_null_value_section_get_statistics_empty_row() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame({"float": [], "int": [], "str": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_all_columns_temporal_null_value_section_get_statistics_only_datetime_column() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


def test_all_columns_temporal_null_value_section_render_html_body() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


def test_all_columns_temporal_null_value_section_render_html_body_args() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


def test_all_columns_temporal_null_value_section_render_html_body_empty() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame({"float": [], "int": [], "str": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_all_columns_temporal_null_value_section_render_html_toc() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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


def test_all_columns_temporal_null_value_section_render_html_toc_args() -> None:
    section = AllColumnsTemporalNullValueSection(
        frame=DataFrame(
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
            frame=DataFrame(
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


@pytest.mark.parametrize("ncols", [1, 2])
def test_create_temporal_null_figure_ncols(ncols: int) -> None:
    assert isinstance(
        create_temporal_null_figure(
            frame=DataFrame(
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


##################################################
#     Tests for create_temporal_null_figures     #
##################################################


def test_create_temporal_null_figures() -> None:
    assert isinstance(
        create_temporal_null_figures(
            frame=DataFrame(
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
        list,
    )


def test_create_temporal_null_figures_empty() -> None:
    assert (
        create_temporal_null_figures(
            frame=DataFrame({}),
            dt_column="datetime",
            period="M",
        )
        == []
    )


def test_create_temporal_null_figures_empty_rows() -> None:
    assert (
        create_temporal_null_figures(
            frame=DataFrame({"float": [], "int": [], "str": [], "datetime": []}),
            dt_column="datetime",
            period="M",
        )
        == []
    )


#################################
#    Tests for prepare_data     #
#################################


def test_prepare_data() -> None:
    assert objects_are_equal(
        prepare_data(
            frame=DataFrame(
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
            frame=DataFrame({"col": [], "datetime": pd.to_datetime([])}),
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
