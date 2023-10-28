from __future__ import annotations

from coola import objects_are_allclose
from jinja2 import Template
from numpy import dtype
from pandas import Series
from pytest import raises

from flamme.section import ColumnDtypeSection, ColumnTypeSection

########################################
#     Tests for ColumnDtypeSection     #
########################################


def test_column_dtype_section_get_statistics() -> None:
    output = ColumnDtypeSection(
        dtypes=Series({"col1": float, "col2": int, "col3": str}),
    )
    assert objects_are_allclose(
        output.get_statistics(),
        {"col1": float, "col2": int, "col3": str},
    )


def test_column_dtype_section_get_statistics_empty() -> None:
    output = ColumnDtypeSection(dtypes=Series({}))
    assert objects_are_allclose(output.get_statistics(), {})


def test_column_dtype_section_render_html_body() -> None:
    output = ColumnDtypeSection(
        dtypes=Series({"col1": float, "col2": int, "col3": str}),
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_column_dtype_section_render_html_body_args() -> None:
    output = ColumnDtypeSection(
        dtypes=Series({"col1": float, "col2": int, "col3": str}),
    )
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_dtype_section_render_html_body_empty() -> None:
    output = ColumnDtypeSection(
        dtypes=Series({}),
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_column_dtype_section_render_html_toc() -> None:
    output = ColumnDtypeSection(
        dtypes=Series({"col1": float, "col2": int, "col3": str}),
    )
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_column_dtype_section_render_html_toc_args() -> None:
    output = ColumnDtypeSection(
        dtypes=Series({"col1": float, "col2": int, "col3": str}),
    )
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#######################################
#     Tests for ColumnTypeSection     #
#######################################


def test_column_type_section_incorrect_different_key() -> None:
    with raises(RuntimeError, match="The keys of dtypes and types do not match:"):
        ColumnTypeSection(
            dtypes={"col": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )


def test_column_type_section_incorrect_missing_key() -> None:
    with raises(RuntimeError, match="The keys of dtypes and types do not match:"):
        ColumnTypeSection(
            dtypes={"int": dtype("float64"), "str": dtype("O")},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )


def test_column_type_section_get_statistics() -> None:
    output = ColumnTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str, type(None)}},
    )
    assert objects_are_allclose(
        output.get_statistics(),
        {"float": {float}, "int": {int}, "str": {str, type(None)}},
    )


def test_column_type_section_get_statistics_empty() -> None:
    output = ColumnTypeSection(dtypes={}, types={})
    assert objects_are_allclose(output.get_statistics(), {})


def test_column_type_section_render_html_body() -> None:
    output = ColumnTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_column_type_section_render_html_body_args() -> None:
    output = ColumnTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_type_section_render_html_body_empty() -> None:
    output = ColumnTypeSection(dtypes={}, types={})
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_column_type_section_render_html_toc() -> None:
    output = ColumnTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_column_type_section_render_html_toc_args() -> None:
    output = ColumnTypeSection(
        dtypes={"float": dtype("float64"), "int": dtype("float64"), "str": dtype("O")},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
