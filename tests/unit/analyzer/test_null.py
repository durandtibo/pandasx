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
    output = NullValueAnalyzer().analyze(
        DataFrame(
            {
                "int": np.array([np.nan, 1, 0, 1]),
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert objects_are_equal(output.get_statistics(), {})


def test_null_value_analyzer_get_statistics_empty() -> None:
    output = NullValueAnalyzer().analyze(
        DataFrame({"int": np.array([]), "float": np.array([]), "str": np.array([])})
    )
    assert objects_are_equal(output.get_statistics(), {})
