from __future__ import annotations

from coola import objects_are_allclose
from jinja2 import Template
from pandas import Series

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


def test_column_type_section_get_statistics() -> None:
    output = ColumnTypeSection(
        types={"col1": {float}, "col2": {int}, "col3": {str, type(None)}},
    )
    assert objects_are_allclose(
        output.get_statistics(),
        {"col1": {float}, "col2": {int}, "col3": {str, type(None)}},
    )


def test_column_type_section_get_statistics_empty() -> None:
    output = ColumnTypeSection(types={})
    assert objects_are_allclose(output.get_statistics(), {})


def test_column_type_section_render_html_body() -> None:
    output = ColumnTypeSection(
        types={"col1": {float}, "col2": {int}, "col3": {str}},
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_column_type_section_render_html_body_args() -> None:
    output = ColumnTypeSection(
        types={"col1": {float}, "col2": {int}, "col3": {str}},
    )
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_type_section_render_html_body_empty() -> None:
    output = ColumnTypeSection(
        types={},
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_column_type_section_render_html_toc() -> None:
    output = ColumnTypeSection(types={"col1": {float}, "col2": {int}, "col3": {str}})
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_column_type_section_render_html_toc_args() -> None:
    output = ColumnTypeSection(
        types={"col1": {float}, "col2": {int}, "col3": {str}},
    )
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
