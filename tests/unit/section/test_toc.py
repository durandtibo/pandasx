from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal
from jinja2 import Template

from flamme.section import DuplicatedRowSection, TableOfContentSection


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.2, 4.2, 4.2, 2.2],
            "col2": [1, 1, 1, 1],
            "col3": [1, 2, 2, 2],
        },
        schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    )


###########################################
#     Tests for TableOfContentSection     #
###########################################


def test_table_of_content_section_str(dataframe: pl.DataFrame) -> None:
    assert str(TableOfContentSection(DuplicatedRowSection(frame=dataframe))).startswith(
        "TableOfContentSection("
    )


def test_table_of_content_section_get_statistics(dataframe: pl.DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=dataframe))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_table_of_content_section_get_statistics_columns(dataframe: pl.DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=dataframe, columns=["col2", "col3"]))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_table_of_content_section_get_statistics_empty_row() -> None:
    section = TableOfContentSection(
        DuplicatedRowSection(
            frame=pl.DataFrame(
                {"col1": [], "col2": [], "col3": []},
                schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
            )
        )
    )
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_table_of_content_section_get_statistics_empty_column() -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=pl.DataFrame({})))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_table_of_content_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=dataframe))
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_table_of_content_section_render_html_body_empty_row() -> None:
    section = TableOfContentSection(
        DuplicatedRowSection(
            frame=pl.DataFrame(
                {"col1": [], "col2": [], "col3": []},
                schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
            ),
        )
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_table_of_content_section_render_html_body_empty_column() -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=pl.DataFrame({})))
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_table_of_content_section_render_html_body_args(
    dataframe: pl.DataFrame,
) -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=dataframe))
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_table_of_content_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=dataframe))
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_table_of_content_section_render_html_toc_args(
    dataframe: pl.DataFrame,
) -> None:
    section = TableOfContentSection(DuplicatedRowSection(frame=dataframe))
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
