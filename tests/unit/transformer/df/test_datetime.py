from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.transformer.df import ToDatetime

####################################################
#     Tests for ToDatetimeDataFrameTransformer     #
####################################################


def test_to_datetime_dataframe_transformer_str() -> None:
    assert (
        str(ToDatetime(columns=["col1", "col3"]))
        == "ToDatetimeDataFrameTransformer(columns=('col1', 'col3'))"
    )


def test_to_datetime_dataframe_transformer_str_kwargs() -> None:
    assert (
        str(ToDatetime(columns=["col1", "col3"], errors="ignore"))
        == "ToDatetimeDataFrameTransformer(columns=('col1', 'col3'), errors=ignore)"
    )


def test_to_datetime_dataframe_transformer_transform() -> None:
    dataframe = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
        }
    )
    transformer = ToDatetime(columns=["col1"])
    dataframe = transformer.transform(dataframe)
    assert_frame_equal(
        dataframe,
        pd.DataFrame(
            {
                "col1": pd.to_datetime(
                    ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"]
                ),
                "col2": [1, 2, 3, 4, 5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
            }
        ),
    )


def test_to_datetime_dataframe_transformer_transform_kwargs() -> None:
    dataframe = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "abc"],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
        }
    )
    transformer = ToDatetime(columns=["col1", "col3"], errors="coerce", format="%Y-%m-%d")
    dataframe = transformer.transform(dataframe)
    assert_frame_equal(
        dataframe,
        pd.DataFrame(
            {
                "col1": pd.to_datetime(
                    ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", pd.NaT],
                    errors="coerce",
                ),
                "col2": [1, 2, 3, 4, 5],
                "col3": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
            }
        ),
    )
