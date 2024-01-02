from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pandas import DataFrame, Series

from flamme.utils.dtype import (
    df_column_types,
    read_dtypes_from_schema,
    series_column_types,
)


@pytest.fixture(scope="module")
def df_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.parquet")
    nrows = 10
    df = DataFrame(
        {
            "col_float": np.arange(nrows, dtype=float) + 0.5,
            "col_int": np.arange(nrows, dtype=int),
            "col_str": [f"a{i}" for i in range(nrows)],
        }
    )
    df.to_parquet(path)
    return path


##################################
#     Tests for column_types     #
##################################


def test_df_column_types() -> None:
    assert df_column_types(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    ) == {"float": {float}, "int": {float}, "str": {str, type(None), float}}


def test_df_column_types_empty() -> None:
    assert df_column_types(DataFrame({})) == {}


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
    assert series_column_types(Series([])) == set()


#############################################
#     Tests for read_dtypes_from_schema     #
#############################################


def test_read_dtypes_from_schema(df_path: Path) -> None:
    dtypes = read_dtypes_from_schema(df_path)
    assert dtypes == {
        "col_float": pa.float64(),
        "col_int": pa.int64(),
        "col_str": pa.string(),
    }
