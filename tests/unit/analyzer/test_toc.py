from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from pandas import DataFrame
from pytest import fixture

from flamme.analyzer import DuplicatedRowAnalyzer, TableOfContentAnalyzer
from flamme.section import TableOfContentSection


@fixture
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col1": np.array([1.2, 4.2, 4.2, 2.2]),
            "col2": np.array([1, 1, 1, 1]),
            "col3": np.array([1, 2, 2, 2]),
        }
    )


############################################
#     Tests for TableOfContentAnalyzer     #
############################################


def test_table_of_content_analyzer_str() -> None:
    assert str(TableOfContentAnalyzer(DuplicatedRowAnalyzer())).startswith(
        "TableOfContentAnalyzer("
    )


def test_table_of_content_analyzer_get_statistics(dataframe: DataFrame) -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer()).analyze(dataframe)
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_table_of_content_analyzer_get_statistics_columns(dataframe: DataFrame) -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer(columns=["col2", "col3"])).analyze(
        dataframe
    )
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_table_of_content_analyzer_get_statistics_empty_rows() -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer()).analyze(
        DataFrame({"col1": [], "col2": []})
    )
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_table_of_content_analyzer_get_statistics_missing_column() -> None:
    section = TableOfContentAnalyzer(DuplicatedRowAnalyzer()).analyze(DataFrame({}))
    assert isinstance(section, TableOfContentSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})
