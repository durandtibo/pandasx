from __future__ import annotations

from coola import objects_are_allclose
from jinja2 import Template
from numpy import dtype
from pytest import raises

from flamme.section import DataTypeSection

#####################################
#     Tests for DataTypeSection     #
#####################################


def test_data_type_section_incorrect_different_key() -> None:
    with raises(RuntimeError, match="The keys of dtypes and types do not match:"):
        DataTypeSection(
            dtypes={"col": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )


def test_data_type_section_incorrect_missing_key() -> None:
    with raises(RuntimeError, match="The keys of dtypes and types do not match:"):
        DataTypeSection(
            dtypes={"int": dtype("float64"), "str": dtype("O")},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )


def test_data_type_section_get_statistics() -> None:
    section = DataTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str, type(None)}},
    )
    assert objects_are_allclose(
        section.get_statistics(),
        {"float": {float}, "int": {int}, "str": {str, type(None)}},
    )


def test_data_type_section_get_statistics_empty() -> None:
    section = DataTypeSection(dtypes={}, types={})
    assert objects_are_allclose(section.get_statistics(), {})


def test_data_type_section_render_html_body() -> None:
    section = DataTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_data_type_section_render_html_body_args() -> None:
    section = DataTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_data_type_section_render_html_body_empty() -> None:
    section = DataTypeSection(dtypes={}, types={})
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_data_type_section_render_html_toc() -> None:
    section = DataTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_data_type_section_render_html_toc_args() -> None:
    section = DataTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
