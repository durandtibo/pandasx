from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal
from jinja2 import Template
from polars.testing import assert_frame_equal

from flamme.section import DuplicatedRowSection
from flamme.section.duplicate import create_duplicate_table


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.2, 4.2, 4.2, 2.2],
            "col2": [1, 1, 1, 1],
            "col3": [1, 2, 2, 2],
        },
        schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    )


##########################################
#     Tests for DuplicatedRowSection     #
##########################################


def test_duplicated_rows_section_str(dataframe: pl.DataFrame) -> None:
    assert str(DuplicatedRowSection(frame=dataframe)).startswith("DuplicatedRowSection(")


def test_duplicated_rows_section_frame(dataframe: pl.DataFrame) -> None:
    section = DuplicatedRowSection(frame=dataframe)
    assert_frame_equal(section.frame, dataframe)


def test_duplicated_rows_section_column_none(dataframe: pl.DataFrame) -> None:
    section = DuplicatedRowSection(frame=dataframe)
    assert section.columns is None


def test_duplicated_rows_section_column(dataframe: pl.DataFrame) -> None:
    section = DuplicatedRowSection(frame=dataframe, columns=["col1", "col2"])
    assert section.columns == ("col1", "col2")


def test_duplicated_rows_section_figsize_default(dataframe: pl.DataFrame) -> None:
    assert DuplicatedRowSection(frame=dataframe, columns=["col1", "col2"]).figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_duplicated_rows_section_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        DuplicatedRowSection(frame=dataframe, columns=["col1", "col2"], figsize=figsize).figsize
        == figsize
    )


def test_duplicated_rows_section_get_statistics(dataframe: pl.DataFrame) -> None:
    section = DuplicatedRowSection(frame=dataframe)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_duplicated_rows_section_get_statistics_columns(dataframe: pl.DataFrame) -> None:
    section = DuplicatedRowSection(frame=dataframe, columns=["col2", "col3"])
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_duplicated_rows_section_get_statistics_empty_row() -> None:
    section = DuplicatedRowSection(
        frame=pl.DataFrame(
            {"col1": [], "col2": [], "col3": []},
            schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        )
    )
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_duplicated_rows_section_get_statistics_empty_column() -> None:
    section = DuplicatedRowSection(frame=pl.DataFrame({}))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_duplicated_rows_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = DuplicatedRowSection(frame=dataframe)
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_duplicated_rows_section_render_html_body_empty_row() -> None:
    section = DuplicatedRowSection(
        frame=pl.DataFrame(
            {"col1": [], "col2": [], "col3": []},
            schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        ),
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_duplicated_rows_section_render_html_body_empty_column() -> None:
    section = DuplicatedRowSection(frame=pl.DataFrame({}))
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_duplicated_rows_section_render_html_body_args(
    dataframe: pl.DataFrame,
) -> None:
    section = DuplicatedRowSection(frame=dataframe)
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_duplicated_rows_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = DuplicatedRowSection(frame=dataframe)
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_duplicated_rows_section_render_html_toc_args(
    dataframe: pl.DataFrame,
) -> None:
    section = DuplicatedRowSection(frame=dataframe)
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


############################################
#     Tests for create_duplicate_table     #
############################################


def test_create_duplicate_table() -> None:
    assert isinstance(create_duplicate_table(num_rows=10, num_unique_rows=5), str)


def test_create_duplicate_table_0() -> None:
    assert isinstance(create_duplicate_table(num_rows=0, num_unique_rows=0), str)
