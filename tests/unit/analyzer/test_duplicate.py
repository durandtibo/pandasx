from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal
from pandas.testing import assert_frame_equal

from flamme.analyzer import DuplicatedRowAnalyzer
from flamme.section import DuplicatedRowSection


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col1": np.array([1.2, 4.2, 4.2, 2.2]),
            "col2": np.array([1, 1, 1, 1]),
            "col3": np.array([1, 2, 2, 2]),
        }
    )


##########################################
#     Tests for DuplicateRowAnalyzer     #
##########################################


def test_duplicated_row_analyzer_str() -> None:
    assert str(DuplicatedRowAnalyzer()).startswith("DuplicatedRowAnalyzer(")


def test_duplicated_row_analyzer_frame(dataframe: pd.DataFrame) -> None:
    section = DuplicatedRowAnalyzer().analyze(dataframe)
    assert_frame_equal(
        section.frame,
        pd.DataFrame(
            {
                "col1": np.array([1.2, 4.2, 4.2, 2.2]),
                "col2": np.array([1, 1, 1, 1]),
                "col3": np.array([1, 2, 2, 2]),
            }
        ),
    )


def test_duplicated_row_analyzer_columns(dataframe: pd.DataFrame) -> None:
    section = DuplicatedRowAnalyzer(columns=["col2", "col3"]).analyze(dataframe)
    assert section.columns == ("col2", "col3")


def test_column_discrete_analyzer_figsize_default(dataframe: pd.DataFrame) -> None:
    section = DuplicatedRowAnalyzer(columns=["col2", "col3"]).analyze(dataframe)
    assert isinstance(section, DuplicatedRowSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_discrete_analyzer_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    section = DuplicatedRowAnalyzer(columns=["col2", "col3"], figsize=figsize).analyze(dataframe)
    assert isinstance(section, DuplicatedRowSection)
    assert section.figsize == figsize


def test_duplicated_row_analyzer_get_statistics(dataframe: pd.DataFrame) -> None:
    section = DuplicatedRowAnalyzer().analyze(dataframe)
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


def test_duplicated_row_analyzer_get_statistics_columns(dataframe: pd.DataFrame) -> None:
    section = DuplicatedRowAnalyzer(columns=["col2", "col3"]).analyze(dataframe)
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 2})


def test_duplicated_row_analyzer_get_statistics_empty_rows() -> None:
    section = DuplicatedRowAnalyzer().analyze(pd.DataFrame({"col1": [], "col2": []}))
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})


def test_duplicated_row_analyzer_get_statistics_missing_column() -> None:
    section = DuplicatedRowAnalyzer().analyze(pd.DataFrame({}))
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_equal(section.get_statistics(), {"num_rows": 0, "num_unique_rows": 0})
