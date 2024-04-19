from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest
from pandas import DataFrame, Series

from flamme.utils.dtype import (
    find_date_columns_from_dtypes,
    find_numeric_columns_from_dtypes,
    frame_column_types,
    get_dtypes_from_schema,
    series_column_types,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.parquet")
    nrows = 10
    frame = DataFrame(
        {
            "col_float": np.arange(nrows, dtype=float) + 0.5,
            "col_int": np.arange(nrows, dtype=int),
            "col_str": [f"a{i}" for i in range(nrows)],
        }
    )
    frame.to_parquet(path)
    return path


@pytest.fixture()
def dtypes() -> dict[str, pa.DataType]:
    return {
        "col_bool": pa.bool_(),
        "col_date32": pa.date32(),
        "col_date64": pa.date64(),
        "col_decimal128_12_0": pa.decimal128(12, 0),
        "col_decimal128_5_2": pa.decimal128(5, 2),
        "col_decimal256_12_0": pa.decimal256(12, 0),
        "col_decimal256_5_2": pa.decimal256(5, 2),
        "col_float16": pa.float16(),
        "col_float32": pa.float32(),
        "col_float64": pa.float64(),
        "col_int16": pa.int16(),
        "col_int32": pa.int32(),
        "col_int64": pa.int64(),
        "col_int8": pa.int8(),
        "col_str": pa.string(),
        "col_time32_s": pa.time32("s"),
        "col_time32_ms": pa.time32("ms"),
        "col_time64_ns": pa.time64("ns"),
        "col_time64_us": pa.time64("us"),
        "col_timestamp_ns": pa.timestamp("ns"),
        "col_timestamp_us": pa.timestamp("us"),
        "col_uint16": pa.uint16(),
        "col_uint32": pa.uint32(),
        "col_uint64": pa.uint64(),
        "col_uint8": pa.uint8(),
    }


##################################
#     Tests for column_types     #
##################################


def test_frame_column_types() -> None:
    assert frame_column_types(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    ) == {"float": {float}, "int": {float}, "str": {str, type(None), float}}


def test_frame_column_types_empty() -> None:
    assert frame_column_types(DataFrame({})) == {}


#########################################
#     Tests for series_column_types     #
#########################################


def test_series_column_types() -> None:
    assert series_column_types(Series(["abc", 1, 4.2, np.nan, None])) == {
        float,
        str,
        int,
        type(None),
    }


def test_series_column_types_empty() -> None:
    assert series_column_types(Series([], dtype=object)) == set()


######################################################
#     Tests for find_numeric_columns_from_dtypes     #
######################################################


def test_find_numeric_columns_from_dtypes(dtypes: dict[str, pa.DataType]) -> None:
    assert find_numeric_columns_from_dtypes(dtypes) == [
        "col_decimal128_12_0",
        "col_decimal128_5_2",
        "col_decimal256_12_0",
        "col_decimal256_5_2",
        "col_float16",
        "col_float32",
        "col_float64",
        "col_int16",
        "col_int32",
        "col_int64",
        "col_int8",
        "col_uint16",
        "col_uint32",
        "col_uint64",
        "col_uint8",
    ]


def test_find_numeric_columns_from_dtypes_empty() -> None:
    assert find_numeric_columns_from_dtypes({}) == []


###################################################
#     Tests for find_date_columns_from_dtypes     #
###################################################


def test_find_date_columns_from_dtypes(dtypes: dict[str, pa.DataType]) -> None:
    assert find_date_columns_from_dtypes(dtypes) == [
        "col_date32",
        "col_date64",
    ]


def test_find_date_columns_from_dtypes_empty() -> None:
    assert find_date_columns_from_dtypes({}) == []


#############################################
#     Tests for read_dtypes_from_schema     #
#############################################


def test_get_dtypes_from_schema() -> None:
    assert get_dtypes_from_schema(
        pa.schema([("col_float", pa.float64()), ("col_int", pa.int32()), ("col_str", pa.string())])
    ) == {
        "col_float": pa.float64(),
        "col_int": pa.int32(),
        "col_str": pa.string(),
    }
