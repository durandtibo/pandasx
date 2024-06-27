from __future__ import annotations

import logging

import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

from flamme.transformer.dataframe import ToNumeric


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )


###################################################
#     Tests for ToNumericDataFrameTransformer     #
###################################################


def test_to_numeric_dataframe_transformer_str() -> None:
    assert (
        str(ToNumeric(columns=["col1", "col3"]))
        == "ToNumericDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False)"
    )


def test_to_numeric_dataframe_transformer_str_kwargs() -> None:
    assert (
        str(ToNumeric(columns=["col1", "col3"], errors="ignore"))
        == "ToNumericDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False, errors=ignore)"
    )


def test_to_numeric_dataframe_transformer_transform(dataframe: pd.DataFrame) -> None:
    transformer = ToNumeric(columns=["col1", "col3"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_to_numeric_dataframe_transformer_transform_kwargs(dataframe: pd.DataFrame) -> None:
    dataframe = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "a5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    transformer = ToNumeric(columns=["col1", "col3"], errors="coerce")
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, float("nan")],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_to_numeric_dataframe_transformer_transform_ignore_missing_false(
    dataframe: pd.DataFrame,
) -> None:
    transformer = ToNumeric(columns=["col1", "col3", "col5"])
    with pytest.raises(RuntimeError, match="column col5 is not in the DataFrame"):
        transformer.transform(dataframe)


def test_to_numeric_dataframe_transformer_transform_ignore_missing_true(
    dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ToNumeric(columns=["col1", "col3", "col5"], ignore_missing=True)
    with caplog.at_level(logging.WARNING):
        out = transformer.transform(dataframe)
        assert_frame_equal(
            out,
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": [1, 2, 3, 4, 5],
                    "col4": ["a", "b", "c", "d", "e"],
                }
            ),
        )
        assert caplog.messages[-1].startswith(
            "skipping transformation for column col5 because the column is missing"
        )


def test_to_numeric_dataframe_transformer_from_schema(dataframe: pd.DataFrame) -> None:
    transformer = ToNumeric.from_schema(
        schema=pa.schema([("col1", pa.int64()), ("col2", pa.string()), ("col3", pa.float64())])
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, 5],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_to_numeric_dataframe_transformer_from_schema_kwargs() -> None:
    dataframe = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "a5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    transformer = ToNumeric.from_schema(
        schema=pa.schema([("col1", pa.int64()), ("col2", pa.string()), ("col3", pa.float64())]),
        errors="coerce",
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": [1, 2, 3, 4, float("nan")],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_to_numeric_dataframe_transformer_from_schema_ignore_missing(
    dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ToNumeric.from_schema(
        schema=pa.schema(
            [
                ("col1", pa.int64()),
                ("col2", pa.string()),
                ("col3", pa.float64()),
                ("col5", pa.float64()),
            ]
        ),
        ignore_missing=True,
    )
    with caplog.at_level(logging.WARNING):
        out = transformer.transform(dataframe)
        assert_frame_equal(
            out,
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": [1, 2, 3, 4, 5],
                    "col4": ["a", "b", "c", "d", "e"],
                }
            ),
        )
        assert caplog.messages[-1].startswith(
            "skipping transformation for column col5 because the column is missing"
        )
