from __future__ import annotations

import logging

import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

from flamme.transformer.dataframe import ToDatetime


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
        }
    )


####################################################
#     Tests for ToDatetimeDataFrameTransformer     #
####################################################


def test_to_datetime_dataframe_transformer_str() -> None:
    assert (
        str(ToDatetime(columns=["col1", "col3"]))
        == "ToDatetimeDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False)"
    )


def test_to_datetime_dataframe_transformer_str_kwargs() -> None:
    assert str(ToDatetime(columns=["col1", "col3"], errors="ignore")) == (
        "ToDatetimeDataFrameTransformer(columns=('col1', 'col3'), "
        "ignore_missing=False, errors=ignore)"
    )


def test_to_datetime_dataframe_transformer_transform(dataframe: pd.DataFrame) -> None:
    transformer = ToDatetime(columns=["col1"])
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
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
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
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


def test_to_datetime_dataframe_transformer_transform_ignore_missing_false(
    dataframe: pd.DataFrame,
) -> None:
    transformer = ToDatetime(columns=["col1", "col5"])
    with pytest.raises(RuntimeError, match="column col5 is not in the DataFrame"):
        transformer.transform(dataframe)


def test_to_datetime_dataframe_transformer_transform_ignore_missing_true(
    dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ToDatetime(columns=["col1", "col5"], ignore_missing=True)
    with caplog.at_level(logging.WARNING):
        out = transformer.transform(dataframe)
        assert_frame_equal(
            out,
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
        assert caplog.messages[-1].startswith(
            "skipping transformation for column col5 because the column is missing"
        )


def test_to_datetime_dataframe_transformer_from_schema(dataframe: pd.DataFrame) -> None:
    transformer = ToDatetime.from_schema(
        schema=pa.schema([("col1", pa.date64()), ("col2", pa.string()), ("col3", pa.int64())])
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
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


def test_to_datetime_dataframe_transformer_from_schema_kwargs() -> None:
    dataframe = pd.DataFrame(
        {
            "col1": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "abc"],
            "col2": [1, 2, 3, 4, 5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
        }
    )
    transformer = ToDatetime.from_schema(
        schema=pa.schema([("col1", pa.date64()), ("col2", pa.string()), ("col3", pa.date64())]),
        errors="coerce",
        format="%Y-%m-%d",
    )
    out = transformer.transform(dataframe)
    assert_frame_equal(
        out,
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


def test_to_datetime_dataframe_transformer_from_schema_ignore_missing(
    dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    transformer = ToDatetime.from_schema(
        schema=pa.schema(
            [
                ("col1", pa.date64()),
                ("col2", pa.string()),
                ("col3", pa.int64()),
                ("col5", pa.date64()),
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
                    "col1": pd.to_datetime(
                        ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"]
                    ),
                    "col2": [1, 2, 3, 4, 5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"],
                }
            ),
        )
        assert caplog.messages[-1].startswith(
            "skipping transformation for column col5 because the column is missing"
        )
