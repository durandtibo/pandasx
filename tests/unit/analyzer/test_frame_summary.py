from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import DataFrameSummaryAnalyzer
from flamme.section import DataFrameSummarySection


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.2, 4.2, None, 2.2, 1, 2.2],
            "col2": [1, 1, 0, 1, 1, 1],
            "col3": ["A", "B", None, None, "C", "B"],
        },
        schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.String},
    )


##############################################
#     Tests for DataFrameSummaryAnalyzer     #
##############################################


def test_column_type_analyzer_str() -> None:
    assert str(DataFrameSummaryAnalyzer()).startswith("DataFrameSummaryAnalyzer(")


def test_column_type_analyzer_get_statistics(dataframe: pl.DataFrame) -> None:
    section = DataFrameSummaryAnalyzer().analyze(dataframe)
    assert isinstance(section, DataFrameSummarySection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (1, 0, 2),
            "nunique": (5, 2, 5),
            "column_types": ({float}, {int}, {float, str, type(None)}),
        },
    )


def test_column_type_analyzer_get_statistics_empty_rows() -> None:
    section = DataFrameSummaryAnalyzer().analyze(pl.DataFrame({"col1": [], "col2": [], "col3": []}))
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
    section = DataFrameSummaryAnalyzer().analyze(pl.DataFrame({}))
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
def test_column_type_analyzer_top(dataframe: pl.DataFrame, top: int) -> None:
    section = DataFrameSummaryAnalyzer(top=top).analyze(dataframe)
    assert isinstance(section, DataFrameSummarySection)
    assert section.top == top


def test_column_type_analyzer_top_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect top value .*. top must be positive"):
        DataFrameSummaryAnalyzer(top=-1)


def test_column_type_analyzer_sort() -> None:
    section = DataFrameSummaryAnalyzer(sort=True).analyze(
        pl.DataFrame(
            {
                "col2": [1, 1, 0, 1, 1, 1],
                "col3": ["A", "B", None, None, "C", "B"],
                "col1": [1.2, 4.2, None, 2.2, 1, 2.2],
            },
            schema={"col2": pl.Int64, "col3": pl.String, "col1": pl.Float64},
        )
    )
    assert isinstance(section, DataFrameSummarySection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (1, 0, 2),
            "nunique": (5, 2, 5),
            "column_types": ({float}, {int}, {float, str, type(None)}),
        },
    )
