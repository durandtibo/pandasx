from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import fixture, mark

from flamme.section import DuplicatedRowSection
from flamme.section.duplicate import create_duplicate_table


@fixture
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col1": np.array([1.2, 4.2, 4.2, 2.2]),
            "col2": np.array([1, 1, 1, 1]),
            "col3": np.array([1, 2, 2, 2]),
        }
    )


##########################################
#     Tests for DuplicatedRowSection     #
##########################################


def test_duplicated_rows_section_df(dataframe: DataFrame) -> None:
    section = DuplicatedRowSection(df=dataframe)
    assert_frame_equal(section.df, dataframe)


def test_duplicated_rows_section_column_none(dataframe: DataFrame) -> None:
    section = DuplicatedRowSection(df=dataframe)
    assert section.columns is None


def test_duplicated_rows_section_column(dataframe: DataFrame) -> None:
    section = DuplicatedRowSection(df=dataframe, columns=["col1", "col2"])
    assert section.columns == ("col1", "col2")


def test_duplicated_rows_section_figsize_default(dataframe: DataFrame) -> None:
    assert DuplicatedRowSection(df=dataframe, columns=["col1", "col2"]).figsize is None


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_duplicated_rows_section_figsize(
    dataframe: DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        DuplicatedRowSection(df=dataframe, columns=["col1", "col2"], figsize=figsize).figsize
        == figsize
    )


def test_duplicated_rows_section_get_statistics(dataframe: DataFrame) -> None:
    section = DuplicatedRowSection(df=dataframe)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_duplicated_rows_section_get_statistics_columns(dataframe: DataFrame) -> None:
    section = DuplicatedRowSection(df=dataframe, columns=["col2", "col3"])
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_duplicated_rows_section_get_statistics_empty_row() -> None:
    section = DuplicatedRowSection(df=DataFrame({"col1": [], "col2": []}))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_duplicated_rows_section_get_statistics_empty_column() -> None:
    section = DuplicatedRowSection(df=DataFrame({}))
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_duplicated_rows_section_render_html_body(dataframe: DataFrame) -> None:
    section = DuplicatedRowSection(df=dataframe)
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_duplicated_rows_section_render_html_body_empty_row() -> None:
    section = DuplicatedRowSection(
        df=DataFrame({"col1": [], "col2": []}),
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_duplicated_rows_section_render_html_body_empty_column() -> None:
    section = DuplicatedRowSection(df=DataFrame({}))
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_duplicated_rows_section_render_html_body_args(
    dataframe: DataFrame,
) -> None:
    section = DuplicatedRowSection(df=dataframe)
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_duplicated_rows_section_render_html_toc(dataframe: DataFrame) -> None:
    section = DuplicatedRowSection(df=dataframe)
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_duplicated_rows_section_render_html_toc_args(
    dataframe: DataFrame,
) -> None:
    section = DuplicatedRowSection(df=dataframe)
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


############################################
#     Tests for create_duplicate_table     #
############################################


def test_create_duplicate_table() -> None:
    assert isinstance(create_duplicate_table(10, 5), str)


def test_create_duplicate_table_0() -> None:
    assert isinstance(create_duplicate_table(0, 0), str)
