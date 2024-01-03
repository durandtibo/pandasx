from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pytest import fixture

from flamme.section import DuplicatedRowSection
from flamme.section.toc import TableOfContentSection


@fixture
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col1": np.array([1.2, 4.2, 4.2, 2.2]),
            "col2": np.array([1, 1, 1, 1]),
            "col3": np.array([1, 2, 2, 2]),
        }
    )


###########################################
#     Tests for TableOfContentSection     #
###########################################


def test_table_of_content_section_get_statistics(dataframe: DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=dataframe))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_table_of_content_section_get_statistics_columns(dataframe: DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=dataframe, columns=["col2", "col3"]))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_table_of_content_section_get_statistics_empty_row() -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=DataFrame({"col1": [], "col2": []})))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_table_of_content_section_get_statistics_empty_column() -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=DataFrame({})))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_table_of_content_section_render_html_body(dataframe: DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=dataframe))
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_table_of_content_section_render_html_body_empty_row() -> None:
    section = TableOfContentSection(
        DuplicatedRowSection(
            df=DataFrame({"col1": [], "col2": []}),
        )
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_table_of_content_section_render_html_body_empty_column() -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=DataFrame({})))
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_table_of_content_section_render_html_body_args(
    dataframe: DataFrame,
) -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=dataframe))
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_table_of_content_section_render_html_toc(dataframe: DataFrame) -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=dataframe))
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_table_of_content_section_render_html_toc_args(
    dataframe: DataFrame,
) -> None:
    section = TableOfContentSection(DuplicatedRowSection(df=dataframe))
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
