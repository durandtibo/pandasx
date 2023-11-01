from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from pandas import DataFrame

from flamme.analyzer import FilteredAnalyzer, NullValueAnalyzer
from flamme.section import NullValueSection

######################################
#     Tests for FilteredAnalyzer     #
######################################


def test_filter_analyzer_str() -> None:
    assert str(FilteredAnalyzer(query="A >= 1", analyzer=NullValueAnalyzer())).startswith(
        "FilteredAnalyzer("
    )


def test_filter_analyzer_get_statistics() -> None:
    section = FilteredAnalyzer(query="float >= 2.0", analyzer=NullValueAnalyzer()).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (0, 0, 1),
            "total_count": (2, 2, 2),
        },
    )


def test_filter_analyzer_get_statistics_empty() -> None:
    section = FilteredAnalyzer(query="float >= 2.0", analyzer=NullValueAnalyzer()).analyze(
        DataFrame({"float": np.array([]), "int": np.array([]), "str": np.array([])})
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
