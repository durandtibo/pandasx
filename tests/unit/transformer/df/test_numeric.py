from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.transformer.df import ToNumeric

###################################################
#     Tests for ToNumericDataFrameTransformer     #
###################################################


def test_to_numeric_dataframe_transformer_str() -> None:
    assert (
        str(ToNumeric(columns=["col1", "col3"]))
        == "ToNumericDataFrameTransformer(columns=('col1', 'col3'))"
    )


def test_to_numeric_dataframe_transformer_str_kwargs() -> None:
    assert (
        str(ToNumeric(columns=["col1", "col3"], errors="ignore"))
        == "ToNumericDataFrameTransformer(columns=('col1', 'col3'), errors=ignore)"
    )


def test_to_numeric_dataframe_transformer_transform() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    transformer = ToNumeric(columns=["col1", "col3"])
    df = transformer.transform(df)
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


def test_to_numeric_dataframe_transformer_transform_kwargs() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "a5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    transformer = ToNumeric(columns=["col1", "col3"], errors="coerce")
    df = transformer.transform(df)
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
