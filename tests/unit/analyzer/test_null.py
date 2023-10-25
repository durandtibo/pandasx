from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from pandas import DataFrame

from pandasx.analyzer import NullValueAnalyzer

#######################################
#     Tests for NullValueAnalyzer     #
#######################################


def test_null_value_analyzer_str() -> None:
    assert str(NullValueAnalyzer()).startswith("NullValueAnalyzer(")


def test_null_value_analyzer_get_statistics() -> None:
    section = NullValueAnalyzer().analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
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
        DataFrame({"float": np.array([]), "int": np.array([]), "str": np.array([])})
    )
    print(section.get_statistics())
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_null_value_analyzer_get_statistics_empty_no_column() -> None:
    section = NullValueAnalyzer().analyze(DataFrame({}))
    assert objects_are_equal(
        section.get_statistics(),
        {"columns": (), "null_count": (), "total_count": ()},
    )
