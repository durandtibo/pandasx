from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import raises

from flamme.preprocessor import ColumnSelectionPreprocessor

#################################################
#     Tests for ColumnSelectionPreprocessor     #
#################################################


def test_column_selection_preprocessor_str() -> None:
    assert str(ColumnSelectionPreprocessor(columns=["col1", "col2"])).startswith(
        "ColumnSelectionPreprocessor("
    )


def test_column_selection_preprocessor_preprocess() -> None:
    df = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    preprocessor = ColumnSelectionPreprocessor(columns=["col1", "col2"])
    df = preprocessor.preprocess(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_column_selection_preprocessor_preprocess_empty_row() -> None:
    preprocessor = ColumnSelectionPreprocessor(columns=["col1", "col2"])
    df = preprocessor.preprocess(pd.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(df, pd.DataFrame({"col1": [], "col2": []}))


def test_column_selection_preprocessor_preprocess_empty() -> None:
    preprocessor = ColumnSelectionPreprocessor(columns=["col1", "col2"])
    with raises(RuntimeError, match=r"Column col1 is not in the DataFrame \(columns:"):
        preprocessor.preprocess(pd.DataFrame({}))
