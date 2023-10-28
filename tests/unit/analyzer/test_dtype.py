from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from pandas import DataFrame

from flamme.analyzer import ColumnTypeAnalyzer
from flamme.section import ColumnTypeSection

########################################
#     Tests for ColumnTypeAnalyzer     #
########################################


def test_column_type_analyzer_str() -> None:
    assert str(ColumnTypeAnalyzer()).startswith("ColumnTypeAnalyzer(")


def test_column_type_analyzer_get_statistics() -> None:
    section = ColumnTypeAnalyzer().analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, ColumnTypeSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"float": {float}, "int": {float}, "str": {str, type(None), float}},
    )


def test_column_type_analyzer_get_statistics_empty() -> None:
    section = ColumnTypeAnalyzer().analyze(
        DataFrame({"float": np.array([]), "int": np.array([]), "str": np.array([])})
    )
    assert isinstance(section, ColumnTypeSection)
    assert objects_are_equal(section.get_statistics(), {"float": set(), "int": set(), "str": set()})


def test_column_type_analyzer_get_statistics_empty_no_column() -> None:
    section = ColumnTypeAnalyzer().analyze(DataFrame({}))
    assert isinstance(section, ColumnTypeSection)
    assert objects_are_equal(section.get_statistics(), {})
