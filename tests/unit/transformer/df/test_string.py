from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.transformer.df import StripStr

##################################################
#     Tests for StripStrDataFrameTransformer     #
##################################################


def test_strip_str_dataframe_transformer_str() -> None:
    assert (
        str(StripStr(columns=["col1", "col3"]))
        == "StripStrDataFrameTransformer(columns=('col1', 'col3'))"
    )


def test_strip_str_dataframe_transformer_transform() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, "  "],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )
    transformer = StripStr(columns=["col1", "col3"])
    df = transformer.transform(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, ""],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["a ", " b", "  c  ", "d", "e"],
            }
        ),
    )
