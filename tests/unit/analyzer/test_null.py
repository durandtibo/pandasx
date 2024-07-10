from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import NullValueAnalyzer
from flamme.section import NullValueSection


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [1.2, 4.2, None, 2.2],
            "int": [None, 1, 0, 1],
            "str": ["A", "B", None, None],
        },
        schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
    )


#######################################
#     Tests for NullValueAnalyzer     #
#######################################


def test_null_value_analyzer_str() -> None:
    assert str(NullValueAnalyzer()).startswith("NullValueAnalyzer(")


def test_null_value_analyzer_figsize(dataframe: pl.DataFrame) -> None:
    section = NullValueAnalyzer(figsize=(200, 100)).analyze(dataframe)
    assert isinstance(section, NullValueSection)
    assert section.figsize == (200, 100)


def test_null_value_analyzer_get_statistics(dataframe: pl.DataFrame) -> None:
    section = NullValueAnalyzer().analyze(dataframe)
    assert isinstance(section, NullValueSection)
    assert section.columns == ("float", "int", "str")
    assert objects_are_equal(section.null_count, np.array([1, 1, 2]))
    assert objects_are_equal(section.total_count, np.array([4, 4, 4]))
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (1, 1, 2),
            "total_count": (4, 4, 4),
        },
    )


def test_null_value_analyzer_get_statistics_empty() -> None:
    section = NullValueAnalyzer().analyze(
        pl.DataFrame(
            {"float": [], "int": [], "str": []},
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_null_value_analyzer_get_statistics_empty_no_column() -> None:
    section = NullValueAnalyzer().analyze(pl.DataFrame({}))
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"columns": (), "null_count": (), "total_count": ()},
    )
