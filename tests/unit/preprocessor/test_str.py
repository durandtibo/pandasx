from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.preprocessor import StripStrPreprocessor

##########################################
#     Tests for StripStrPreprocessor     #
##########################################


def test_strip_str_preprocessor_str() -> None:
    assert (
        str(StripStrPreprocessor(columns=["col1", "col3"]))
        == "StripStrPreprocessor(columns=('col1', 'col3'))"
    )


def test_strip_str_preprocessor_preprocess() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, "  "],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )
    preprocessor = StripStrPreprocessor(columns=["col1", "col3"])
    df = preprocessor.preprocess(df)
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
