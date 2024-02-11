from __future__ import annotations

import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

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
    dataframe = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [" a", "b ", " c ", "  d  ", "e"],
        }
    )
    transformer = Column({"col2": ToNumeric()})
    dataframe = transformer.transform(dataframe)
    assert_frame_equal(
        dataframe,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
                "col3": [" a", "b ", " c ", "  d  ", "e"],
            }
        ),
    )


def test_column_dataframe_transformer_transform_2() -> None:
    dataframe = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": [" a", "b ", " c ", "  d  ", "e"],
        }
    )
    transformer = Column({"col2": ToNumeric(), "col3": StripString()})
    dataframe = transformer.transform(dataframe)
    assert_frame_equal(
        dataframe,
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
    dataframe = transformer.transform(pd.DataFrame({}))
    assert_frame_equal(dataframe, pd.DataFrame({}))


def test_column_dataframe_transformer_transform_missing_column() -> None:
    transformer = Column({"col": ToNumeric()})
    with pytest.raises(RuntimeError, match="Column .* is not in the DataFrame"):
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
    caplog: pytest.LogCaptureFixture,
) -> None:
    transformer = Column({"col": ToNumeric()}, ignore_missing=True)
    with caplog.at_level(level=logging.WARNING):
        dataframe = transformer.transform(
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
            dataframe,
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": [" a", "b ", " c ", "  d  ", "e"],
                }
            ),
        )


def test_column_dataframe_transformer_add_transformer() -> None:
    transformer = Column({"col2": ToNumeric(), "col3": StripString()})
    transformer.add_transformer(column="col1", transformer=ToNumeric())
    assert len(transformer.transformers) == 3
    assert isinstance(transformer.transformers["col1"], ToNumeric)
    assert isinstance(transformer.transformers["col2"], ToNumeric)
    assert isinstance(transformer.transformers["col3"], StripString)


def test_column_dataframe_transformer_add_transformer_replace_ok_false() -> None:
    transformer = Column({"col2": ToNumeric(), "col3": StripString()})
    with pytest.raises(KeyError, match="`col3` is already used to register a transformer."):
        transformer.add_transformer(column="col3", transformer=ToNumeric())


def test_column_dataframe_transformer_add_transformer_replace_ok_true() -> None:
    transformer = Column({"col2": ToNumeric(), "col3": StripString()})
    transformer.add_transformer(column="col3", transformer=ToNumeric(), replace_ok=True)
    assert len(transformer.transformers) == 2
    assert isinstance(transformer.transformers["col2"], ToNumeric)
    assert isinstance(transformer.transformers["col3"], ToNumeric)
