from __future__ import annotations

import numpy as np
from coola import objects_are_allclose
from jinja2 import Template
from pytest import raises

from pandasx.section import NullValueSection

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


def test_null_value_section_get_statistics() -> None:
    output = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert objects_are_allclose(
        output.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (0, 1, 2),
            "total_count": (5, 5, 5),
        },
    )


def test_null_value_section_get_statistics_empty_row() -> None:
    output = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 0, 0]),
        total_count=np.array([0, 0, 0]),
    )
    assert objects_are_allclose(
        output.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_null_value_section_get_statistics_empty_column() -> None:
    output = NullValueSection(columns=[], null_count=np.array([]), total_count=np.array([]))
    assert objects_are_allclose(
        output.get_statistics(),
        {"columns": (), "null_count": (), "total_count": ()},
    )


def test_null_value_section_render_html_body() -> None:
    output = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_null_value_section_render_html_body_args() -> None:
    output = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_null_value_section_render_html_body_empty() -> None:
    output = NullValueSection(
        columns=[],
        null_count=np.array([]),
        total_count=np.array([]),
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_null_value_section_render_html_toc() -> None:
    output = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_null_value_section_render_html_toc_args() -> None:
    output = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
