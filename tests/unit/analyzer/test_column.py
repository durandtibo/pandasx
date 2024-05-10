from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_equal

from flamme.analyzer import ColumnSubsetAnalyzer, NullValueAnalyzer
from flamme.section import NullValueSection

##########################################
#     Tests for ColumnSubsetAnalyzer     #
##########################################


def test_column_subset_analyzer_str() -> None:
    assert str(
        ColumnSubsetAnalyzer(columns=["col1", "col2"], analyzer=NullValueAnalyzer())
    ).startswith("ColumnSubsetAnalyzer(")


def test_column_subset_analyzer_get_statistics() -> None:
    section = ColumnSubsetAnalyzer(columns=["col1", "col2"], analyzer=NullValueAnalyzer()).analyze(
        pd.DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, 0, 1]),
                "col3": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col1", "col2"),
            "null_count": (1, 1),
            "total_count": (4, 4),
        },
    )


def test_column_subset_analyzer_get_statistics_empty() -> None:
    section = ColumnSubsetAnalyzer(columns=["col1", "col2"], analyzer=NullValueAnalyzer()).analyze(
        pd.DataFrame({"col1": np.array([]), "col2": np.array([]), "col3": np.array([])})
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col1", "col2"),
            "null_count": (0, 0),
            "total_count": (0, 0),
        },
    )
