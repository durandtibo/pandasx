from __future__ import annotations

import logging

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import LogCaptureFixture, raises

from flamme.transformer.df import Column
from flamme.transformer.series import StripString, ToNumeric

################################################
#     Tests for ColumnDataFrameTransformer     #
################################################


def test_column_dataframe_transformer_str() -> None:
    assert str(Column({"col2": StripString(), "col3": ToNumeric()})).startswith(
        "ColumnDataFrameTransformer("
    )


def test_column_dataframe_transformer_str_empty() -> None:
    assert str(Column({})).startswith("ColumnDataFrameTransformer(")


def test_column_dataframe_transformer_transform_1() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [" a", "b ", " c ", "  d  ", "e"],
        }
    )
    transformer = Column({"col2": ToNumeric()})
    df = transformer.transform(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [" a", "b ", " c ", "  d  ", "e"],
            }
        ),
    )


def test_column_dataframe_transformer_transform_2() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [" a", "b ", " c ", "  d  ", "e"],
        }
    )
    transformer = Column({"col2": ToNumeric(), "col3": StripString()})
    df = transformer.transform(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_column_dataframe_transformer_transform_empty() -> None:
    transformer = Column({})
    df = transformer.transform(pd.DataFrame({}))
    assert_frame_equal(df, pd.DataFrame({}))


def test_column_dataframe_transformer_transform_missing_column() -> None:
    transformer = Column({"col": ToNumeric()})
    with raises(RuntimeError, match="Column .* is not in the DataFrame"):
        transformer.transform(
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": [" a", "b ", " c ", "  d  ", "e"],
                }
            )
        )


def test_column_dataframe_transformer_transform_missing_column_ignore_missing(
    caplog: LogCaptureFixture,
) -> None:
    transformer = Column({"col": ToNumeric()}, ignore_missing=True)
    with caplog.at_level(level=logging.WARNING):
        df = transformer.transform(
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": [" a", "b ", " c ", "  d  ", "e"],
                }
            )
        )
        assert caplog.messages
        assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": [" a", "b ", " c ", "  d  ", "e"],
                }
            ),
        )
