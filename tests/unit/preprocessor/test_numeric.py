from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.preprocessor import ToNumericPreprocessor

###########################################
#     Tests for ToNumericPreprocessor     #
###########################################


def test_to_numeric_preprocessor_str() -> None:
    assert (
        str(ToNumericPreprocessor(columns=["col1", "col3"]))
        == "ToNumericPreprocessor(columns=('col1', 'col3'))"
    )


def test_to_numeric_preprocessor_str_kwargs() -> None:
    assert (
        str(ToNumericPreprocessor(columns=["col1", "col3"], errors="ignore"))
        == "ToNumericPreprocessor(columns=('col1', 'col3'), errors=ignore)"
    )


def test_to_numeric_preprocessor_preprocess() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    preprocessor = ToNumericPreprocessor(columns=["col1", "col3"])
    df = preprocessor.preprocess(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_to_numeric_preprocessor_preprocess_kwargs() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "a5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    preprocessor = ToNumericPreprocessor(columns=["col1", "col3"], errors="coerce")
    df = preprocessor.preprocess(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, float("nan")],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )
