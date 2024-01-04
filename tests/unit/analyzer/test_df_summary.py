from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal
from pandas import DataFrame

from flamme.analyzer import DataFrameSummaryAnalyzer
from flamme.section import DataFrameSummarySection

NoneType = type(None)


@pytest.fixture()
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col1": np.array([1.2, 4.2, np.nan, 2.2, 1, 2.2]),
            "col2": np.array([1, 1, 0, 1, 1, 1]),
            "col3": np.array(["A", "B", None, np.nan, "C", "B"]),
        }
    )


##############################################
#     Tests for DataFrameSummaryAnalyzer     #
##############################################


def test_column_type_analyzer_str() -> None:
    assert str(DataFrameSummaryAnalyzer()).startswith("DataFrameSummaryAnalyzer(")


def test_column_type_analyzer_get_statistics(dataframe: DataFrame) -> None:
    section = DataFrameSummaryAnalyzer().analyze(dataframe)
    assert isinstance(section, DataFrameSummarySection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (1, 0, 2),
            "nunique": (5, 2, 5),
            "column_types": ({float}, {int}, {float, str, NoneType}),
        },
    )


def test_column_type_analyzer_get_statistics_empty_rows() -> None:
    section = DataFrameSummaryAnalyzer().analyze(DataFrame({"col1": [], "col2": [], "col3": []}))
    assert isinstance(section, DataFrameSummarySection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (0, 0, 0),
            "nunique": (0, 0, 0),
            "column_types": (set(), set(), set()),
        },
    )


def test_column_type_analyzer_get_statistics_empty_no_column() -> None:
    section = DataFrameSummaryAnalyzer().analyze(DataFrame({}))
    assert isinstance(section, DataFrameSummarySection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": (),
            "null_count": (),
            "nunique": (),
            "column_types": (),
        },
    )


@pytest.mark.parametrize("top", [0, 1, 2])
def test_column_type_analyzer_top(dataframe: DataFrame, top: int) -> None:
    section = DataFrameSummaryAnalyzer(top=top).analyze(dataframe)
    assert isinstance(section, DataFrameSummarySection)
    assert section.top == top


def test_column_type_analyzer_sort() -> None:
    section = DataFrameSummaryAnalyzer().analyze(
        DataFrame(
            {
                "col3": np.array([1.2, 4.2, np.nan, 2.2, 1, 2.2]),
                "col1": np.array([1, 1, 0, 1, 1, 1]),
                "col2": np.array(["A", "B", None, np.nan, "C", "B"]),
            }
        )
    )
    assert isinstance(section, DataFrameSummarySection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col3", "col1", "col2"),
            "null_count": (1, 0, 2),
            "nunique": (5, 2, 5),
            "column_types": ({float}, {int}, {float, str, NoneType}),
        },
    )
