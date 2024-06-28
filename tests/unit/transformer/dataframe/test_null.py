from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.transformer.dataframe import NullColumn

####################################################
#     Tests for NullColumnDataFrameTransformer     #
####################################################


def test_null_column_dataframe_transformer_str() -> None:
    assert str(NullColumn(threshold=1.0)) == "NullColumnDataFrameTransformer(threshold=1.0)"


def test_null_column_dataframe_transformer_transform_threshold_1() -> None:
    frame = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    transformer = NullColumn(threshold=1.0)
    out = transformer.transform(frame)
    assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
                "col2": [1, None, 3, None, 5],
            }
        ),
    )


def test_null_column_dataframe_transformer_transform_threshold_0_4() -> None:
    frame = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    transformer = NullColumn(threshold=0.4)
    out = transformer.transform(frame)
    assert_frame_equal(
        out, pd.DataFrame({"col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None]})
    )


def test_null_column_dataframe_transformer_transform_threshold_0_2() -> None:
    frame = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", None],
            "col2": [1, None, 3, None, 5],
            "col3": [None, None, None, None, None],
        }
    )
    transformer = NullColumn(threshold=0.2)
    out = transformer.transform(frame)
    assert out.shape == (5, 0)


def test_null_column_dataframe_transformer_transform_empty_row() -> None:
    transformer = NullColumn(threshold=0.5)
    out = transformer.transform(pd.DataFrame({"col1": [], "col2": [], "col3": []}))
    assert_frame_equal(out, pd.DataFrame({"col1": [], "col2": [], "col3": []}))


def test_null_column_dataframe_transformer_transform_empty() -> None:
    transformer = NullColumn(threshold=0.5)
    out = transformer.transform(pd.DataFrame({}))
    assert_frame_equal(out, pd.DataFrame({}))
