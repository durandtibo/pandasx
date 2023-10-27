from __future__ import annotations

import numpy as np
from coola import objects_are_allclose
from jinja2 import Template
from pytest import raises

from flamme.section import NanValueSection

#####################################
#     Tests for NanValueSection     #
#####################################


def test_nan_value_section_incorrect_nan_count_size() -> None:
    with raises(RuntimeError, match=r"columns \(3\) and nan_count \(2\) do not match"):
        NanValueSection(
            columns=["col1", "col2", "col3"],
            nan_count=np.array([0, 1]),
            total_count=np.array([5, 5, 5]),
        )


def test_nan_value_section_incorrect_total_count_size() -> None:
    with raises(RuntimeError, match=r"columns \(3\) and total_count \(2\) do not match"):
        NanValueSection(
            columns=["col1", "col2", "col3"],
            nan_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5]),
        )


def test_nan_value_section_get_statistics() -> None:
    output = NanValueSection(
        columns=["col1", "col2", "col3"],
        nan_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert objects_are_allclose(
        output.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "nan_count": (0, 1, 2),
            "total_count": (5, 5, 5),
        },
    )


def test_nan_value_section_get_statistics_empty_row() -> None:
    output = NanValueSection(
        columns=["col1", "col2", "col3"],
        nan_count=np.array([0, 0, 0]),
        total_count=np.array([0, 0, 0]),
    )
    assert objects_are_allclose(
        output.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "nan_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_nan_value_section_get_statistics_empty_column() -> None:
    output = NanValueSection(columns=[], nan_count=np.array([]), total_count=np.array([]))
    assert objects_are_allclose(
        output.get_statistics(),
        {"columns": (), "nan_count": (), "total_count": ()},
    )


def test_nan_value_section_render_html_body() -> None:
    output = NanValueSection(
        columns=["col1", "col2", "col3"],
        nan_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_nan_value_section_render_html_body_args() -> None:
    output = NanValueSection(
        columns=["col1", "col2", "col3"],
        nan_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_nan_value_section_render_html_body_empty() -> None:
    output = NanValueSection(
        columns=[],
        nan_count=np.array([]),
        total_count=np.array([]),
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_nan_value_section_render_html_toc() -> None:
    output = NanValueSection(
        columns=["col1", "col2", "col3"],
        nan_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_nan_value_section_render_html_toc_args() -> None:
    output = NanValueSection(
        columns=["col1", "col2", "col3"],
        nan_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
