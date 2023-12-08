from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import raises

from flamme.transformer.df import ColumnSelection

#########################################################
#     Tests for ColumnSelectionDataFrameTransformer     #
#########################################################


def test_column_selection_dataframe_transformer_str() -> None:
    assert str(ColumnSelection(columns=["col1", "col2"])).startswith(
        "ColumnSelectionDataFrameTransformer("
    )


def test_column_selection_dataframe_transformer_transform() -> None:
    df = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    transformer = ColumnSelection(columns=["col1", "col2"])
    df = transformer.transform(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_column_selection_dataframe_transformer_transform_empty_row() -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    df = transformer.transform(pd.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(df, pd.DataFrame({"col1": [], "col2": []}))


def test_column_selection_dataframe_transformer_transform_empty() -> None:
    transformer = ColumnSelection(columns=["col1", "col2"])
    with raises(RuntimeError, match=r"Column col1 is not in the DataFrame \(columns:"):
        transformer.transform(pd.DataFrame({}))
