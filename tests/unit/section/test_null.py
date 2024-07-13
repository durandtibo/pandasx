from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from matplotlib import pyplot as plt

from flamme.section import NullValueSection
from flamme.section.null import (
    create_bar_figure,
    create_section_template,
    create_table,
    create_table_row,
)

######################################
#     Tests for NullValueSection     #
######################################


def test_null_value_section_str() -> None:
    assert str(
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5, 5]),
        )
    ).startswith("NullValueSection(")


def test_null_value_section_incorrect_null_count_size() -> None:
    with pytest.raises(RuntimeError, match=r"columns \(3\) and null_count \(2\) do not match"):
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1]),
            total_count=np.array([5, 5, 5]),
        )


def test_null_value_section_incorrect_total_count_size() -> None:
    with pytest.raises(RuntimeError, match=r"columns \(3\) and total_count \(2\) do not match"):
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


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
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


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)


#######################################
#     Tests for create_bar_figure     #
#######################################


def test_create_bar_figure() -> None:
    assert isinstance(
        create_bar_figure(columns=["col1", "col2", "col3"], null_count=[5, 10, 2]), plt.Figure
    )


def test_create_bar_figure_empty() -> None:
    assert isinstance(create_bar_figure(columns=[], null_count=[]), plt.Figure)


def test_create_bar_figure_incorrect_lengths() -> None:
    with pytest.raises(RuntimeError, match="columns .* and null_count .* do not match"):
        create_bar_figure(columns=["col1", "col2", "col3"], null_count=[5, 10, 2, 5])


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert isinstance(
        create_table(
            pl.DataFrame(
                {"column": ["col1", "col2", "col3"], "null": [0, 1, 2], "total": [5, 5, 5]},
                schema={"column": pl.String, "null": pl.Int64, "total": pl.Int64},
            )
        ),
        str,
    )


def test_create_table_empty() -> None:
    assert isinstance(
        create_table(
            pl.DataFrame(
                {"column": [], "null": [], "total": []},
                schema={"column": pl.String, "null": pl.Int64, "total": pl.Int64},
            )
        ),
        str,
    )


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(create_table_row(column="col", null_count=5, total_count=101), str)


def test_create_table_row_zero() -> None:
    assert isinstance(create_table_row(column="col", null_count=0, total_count=0), str)
