from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from flamme.utils.dtype2 import compact_type_name, frame_types, series_types


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


#################################
#     Tests for frame_types     #
#################################


def test_frame_types() -> None:
    assert frame_types(
        pl.DataFrame(
            {
                "float": [1.2, 4.2, float("nan"), 2.2],
                "int": [None, 1, 0, 1],
                "str": ["A", "B", None, "D"],
            },
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    ) == {"float": {float}, "int": {int, type(None)}, "str": {str, type(None)}}


def test_frame_types_empty() -> None:
    assert frame_types(pl.DataFrame({})) == {}


##################################
#     Tests for series_types     #
##################################


def test_series_types() -> None:
    assert series_types(pl.Series(["abc", 1, 4.2, float("nan"), None], dtype=pl.Object)) == {
        float,
        str,
        int,
        type(None),
    }


def test_series_types_int() -> None:
    assert series_types(pl.Series([1, 2, 3, 4, None], dtype=pl.Int64)) == {int, type(None)}


def test_series_types_string() -> None:
    assert series_types(pl.Series(["A", "B", "c", "d", None], dtype=pl.String)) == {str, type(None)}


def test_series_types_empty() -> None:
    assert series_types(pl.Series([], dtype=pl.Object)) == set()


#######################################
#     Tests for compact_type_name     #
#######################################


@pytest.mark.parametrize(
    ("typ", "name"),
    [
        (float, "float"),
        (int, "int"),
        (str, "str"),
        (type(None), "NoneType"),
        (pd.Timestamp, "pandas.Timestamp"),
        (np.ndarray, "numpy.ndarray"),
    ],
)
def test_compact_type_name(typ: type, name: str) -> None:
    assert compact_type_name(typ) == name
