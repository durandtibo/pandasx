from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from flamme.analyzer import DuplicateRowAnalyzer
from flamme.section import DuplicatedRowSection

##########################################
#     Tests for DuplicateRowAnalyzer     #
##########################################


def test_duplicate_row_analyzer_str() -> None:
    assert str(DuplicateRowAnalyzer()).startswith("DuplicateRowAnalyzer(")


def test_duplicate_row_analyzer_df() -> None:
    section = DuplicateRowAnalyzer().analyze(
        DataFrame(
            {
                "col1": np.array([1.2, 4.2, 4.2, 2.2]),
                "col2": np.array([1, 1, 1, 1]),
                "col3": np.array([1, 2, 2, 2]),
            }
        )
    )
    assert_frame_equal(
        section.df,
        DataFrame(
            {
                "col1": np.array([1.2, 4.2, 4.2, 2.2]),
                "col2": np.array([1, 1, 1, 1]),
                "col3": np.array([1, 2, 2, 2]),
            }
        ),
    )


def test_duplicate_row_analyzer_columns() -> None:
    section = DuplicateRowAnalyzer(columns=["col2", "col3"]).analyze(
        DataFrame(
            {
                "col1": np.array([1.2, 4.2, 4.2, 2.2]),
                "col2": np.array([1, 1, 1, 1]),
                "col3": np.array([1, 2, 2, 2]),
            }
        )
    )
    assert section.columns == ("col2", "col3")


def test_duplicate_row_analyzer_get_statistics() -> None:
    section = DuplicateRowAnalyzer().analyze(
        DataFrame(
            {
                "col1": np.array([1.2, 4.2, 4.2, 2.2]),
                "col2": np.array([1, 1, 1, 1]),
                "col3": np.array([1, 2, 2, 2]),
            }
        )
    )
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_duplicate_row_analyzer_get_statistics_columns() -> None:
    section = DuplicateRowAnalyzer(columns=["col2", "col3"]).analyze(
        DataFrame(
            {
                "col1": np.array([1.2, 4.2, 4.2, 2.2]),
                "col2": np.array([1, 1, 1, 1]),
                "col3": np.array([1, 2, 2, 2]),
            }
        )
    )
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_duplicate_row_analyzer_get_statistics_empty_rows() -> None:
    section = DuplicateRowAnalyzer().analyze(DataFrame({"col1": [], "col2": []}))
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_duplicate_row_analyzer_get_statistics_missing_column() -> None:
    section = DuplicateRowAnalyzer().analyze(DataFrame({}))
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})
