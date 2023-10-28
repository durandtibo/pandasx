from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from pandas import DataFrame

from flamme.analyzer import NanValueAnalyzer
from flamme.section import NanValueSection

######################################
#     Tests for NanValueAnalyzer     #
######################################


def test_nan_value_analyzer_str() -> None:
    assert str(NanValueAnalyzer()).startswith("NanValueAnalyzer(")


def test_nan_value_analyzer_get_statistics() -> None:
    section = NanValueAnalyzer().analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, NanValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "nan_count": (1, 1, 2),
            "total_count": (4, 4, 4),
        },
    )


def test_nan_value_analyzer_get_statistics_empty() -> None:
    section = NanValueAnalyzer().analyze(
        DataFrame({"float": np.array([]), "int": np.array([]), "str": np.array([])})
    )
    assert isinstance(section, NanValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "nan_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_nan_value_analyzer_get_statistics_empty_no_column() -> None:
    section = NanValueAnalyzer().analyze(DataFrame({}))
    assert isinstance(section, NanValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"columns": (), "nan_count": (), "total_count": ()},
    )
