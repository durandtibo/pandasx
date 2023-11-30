from __future__ import annotations

import pandas as pd
from pandas._testing import assert_frame_equal

from flamme.preprocessor import NullColumnPreprocessor

############################################
#     Tests for NullColumnPreprocessor     #
############################################


def test_null_column_preprocessor_str() -> None:
    assert str(NullColumnPreprocessor(threshold=1.0)) == "NullColumnPreprocessor(threshold=1.0)"


def test_null_column_preprocessor_preprocess_threshold_1() -> None:
    df = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    preprocessor = NullColumnPreprocessor(threshold=1.0)
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


def test_null_column_preprocessor_preprocess_threshold_0_4() -> None:
    df = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    preprocessor = NullColumnPreprocessor(threshold=0.4)
    df = preprocessor.preprocess(df)
    assert_frame_equal(
        df, pd.DataFrame({"col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None]})
    )


def test_null_column_preprocessor_preprocess_threshold_0_2() -> None:
    df = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    preprocessor = NullColumnPreprocessor(threshold=0.2)
    df = preprocessor.preprocess(df)
    assert df.shape == (5, 0)


def test_null_column_preprocessor_preprocess_empty_row() -> None:
    preprocessor = NullColumnPreprocessor(threshold=0.5)
    df = preprocessor.preprocess(pd.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(df, pd.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_null_column_preprocessor_preprocess_empty() -> None:
    preprocessor = NullColumnPreprocessor(threshold=0.5)
    df = preprocessor.preprocess(pd.DataFrame({}))
    assert_frame_equal(df, pd.DataFrame({}))
