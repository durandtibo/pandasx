from __future__ import annotations

import polars as pl
from coola import objects_are_equal

from flamme.analyzer import ColumnSubsetAnalyzer, NullValueAnalyzer
from flamme.section import NullValueSection

##########################################
#     Tests for ColumnSubsetAnalyzer     #
##########################################


def test_column_subset_analyzer_str() -> None:
    assert str(
        ColumnSubsetAnalyzer(columns=["float", "str"], analyzer=NullValueAnalyzer())
    ).startswith("ColumnSubsetAnalyzer(")


def test_column_subset_analyzer_get_statistics() -> None:
    section = ColumnSubsetAnalyzer(columns=["float", "str"], analyzer=NullValueAnalyzer()).analyze(
        pl.DataFrame(
            {
                "float": [1.2, 4.2, None, 2.2],
                "int": [None, 1, 0, 1],
                "str": ["A", "B", None, None],
            },
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "str"),
            "null_count": (1, 2),
            "total_count": (4, 4),
        },
    )


def test_column_subset_analyzer_get_statistics_empty() -> None:
    section = ColumnSubsetAnalyzer(columns=["float", "str"], analyzer=NullValueAnalyzer()).analyze(
        pl.DataFrame(
            {
                "float": [],
                "int": [],
                "str": [],
            },
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "str"),
            "null_count": (0, 0),
            "total_count": (0, 0),
        },
    )
