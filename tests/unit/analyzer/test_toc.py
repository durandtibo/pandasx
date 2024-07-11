from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import DuplicatedRowAnalyzer, TableOfContentAnalyzer
from flamme.section import TableOfContentSection


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


############################################
#     Tests for TableOfContentAnalyzer     #
############################################


def test_table_of_content_analyzer_str() -> None:
    assert str(TableOfContentAnalyzer(DuplicatedRowAnalyzer())).startswith(
        "TableOfContentAnalyzer("
    )


def test_table_of_content_analyzer_get_statistics(dataframe: pl.DataFrame) -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer()).analyze(dataframe)
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_table_of_content_analyzer_get_statistics_columns(dataframe: pl.DataFrame) -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer(columns=["col2", "col3"])).analyze(
        dataframe
    )
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_table_of_content_analyzer_get_statistics_empty_rows() -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer()).analyze(
        pl.DataFrame(
            {"col1": [], "col2": [], "col3": []},
            schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        )
    )
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_table_of_content_analyzer_get_statistics_missing_column() -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer()).analyze(pl.DataFrame({}))
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})
