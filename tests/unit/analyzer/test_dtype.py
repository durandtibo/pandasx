from __future__ import annotations

import polars as pl
from coola import objects_are_equal

from flamme.analyzer import DataTypeAnalyzer
from flamme.section import DataTypeSection

########################################
#     Tests for ColumnTypeAnalyzer     #
########################################


def test_column_type_analyzer_str() -> None:
    assert str(DataTypeAnalyzer()) == "DataTypeAnalyzer()"


def test_column_type_analyzer_get_statistics() -> None:
    section = DataTypeAnalyzer().analyze(
        pl.DataFrame(
            {
                "int": [None, 1, 0, 1],
                "float": [1.2, 4.2, float("nan"), 2.2],
                "str": ["A", "B", None, None],
            },
            schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
        )
    )
    assert isinstance(section, DataTypeSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"float": {float}, "int": {int, type(None)}, "str": {str, type(None)}},
    )


def test_column_type_analyzer_get_statistics_empty() -> None:
    section = DataTypeAnalyzer().analyze(
        pl.DataFrame(
            {"int": [], "float": [], "str": []},
            schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
        )
    )
    assert isinstance(section, DataTypeSection)
    assert objects_are_equal(section.get_statistics(), {"float": set(), "int": set(), "str": set()})


def test_column_type_analyzer_get_statistics_empty_no_column() -> None:
    section = DataTypeAnalyzer().analyze(pl.DataFrame({}))
    assert isinstance(section, DataTypeSection)
    assert objects_are_equal(section.get_statistics(), {})
