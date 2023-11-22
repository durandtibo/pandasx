from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.preprocessor import (
    SequentialPreprocessor,
    StripStrPreprocessor,
    ToNumericPreprocessor,
)

############################################
#     Tests for SequentialPreprocessor     #
############################################


def test_sequential_preprocessor_str() -> None:
    assert str(
        SequentialPreprocessor(
            [
                StripStrPreprocessor(columns=["col1", "col3"]),
                ToNumericPreprocessor(columns=["col1", "col2"]),
            ]
        )
    ).startswith("SequentialPreprocessor(")


def test_sequential_preprocessor_str_empty() -> None:
    assert str(SequentialPreprocessor([])).startswith("SequentialPreprocessor()")


def test_sequential_preprocessor_preprocess_1() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, "  "],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )
    preprocessor = SequentialPreprocessor(
        [ToNumericPreprocessor(columns=["col1", "col2"], errors="coerce")]
    )
    df = preprocessor.preprocess(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["a ", " b", "  c  ", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
            }
        ),
    )


def test_sequential_preprocessor_preprocess_2() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, "  "],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )
    preprocessor = SequentialPreprocessor(
        [
            StripStrPreprocessor(columns=["col1", "col3"]),
            ToNumericPreprocessor(columns=["col1", "col2"]),
        ]
    )
    df = preprocessor.preprocess(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
            }
        ),
    )
