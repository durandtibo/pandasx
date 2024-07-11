from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import DataTypeSection
from flamme.section.dtype import create_section_template, create_table, create_table_row

#####################################
#     Tests for DataTypeSection     #
#####################################


def test_data_type_section_repr() -> None:
    assert str(
        DataTypeSection(
            dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )
    ).startswith("DataTypeSection(")


def test_data_type_section_str() -> None:
    assert str(
        DataTypeSection(
            dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )
    ).startswith("DataTypeSection(")


def test_data_type_section_incorrect_different_key() -> None:
    with pytest.raises(RuntimeError, match="The keys of dtypes and types do not match:"):
        DataTypeSection(
            dtypes={"col": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )


def test_data_type_section_incorrect_missing_key() -> None:
    with pytest.raises(RuntimeError, match="The keys of dtypes and types do not match:"):
        DataTypeSection(
            dtypes={"int": pl.Int64(), "str": pl.String()},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        )


def test_data_type_section_get_statistics() -> None:
    section = DataTypeSection(
        dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
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
        dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_data_type_section_render_html_body_args() -> None:
    section = DataTypeSection(
        dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
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
        dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_data_type_section_render_html_toc_args() -> None:
    section = DataTypeSection(
        dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
        types={"float": {float}, "int": {int}, "str": {str}},
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert isinstance(
        create_table(
            dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
            types={"float": {float}, "int": {int}, "str": {str, type(None)}},
        ),
        str,
    )


def test_create_table_empty() -> None:
    assert isinstance(create_table(dtypes={}, types={}), str)


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(create_table_row(column="col", dtype=pl.Int64(), types={int}), str)


def test_create_table_row_empty() -> None:
    assert isinstance(
        create_table_row(column="col", dtype=pl.Int64(), types={int, type(None)}), str
    )
