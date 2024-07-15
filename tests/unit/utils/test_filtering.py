from __future__ import annotations

import polars as pl
import pytest

from flamme.utils.filtering import (
    find_columns_decimal,
    find_columns_str,
    find_columns_type,
)


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col0": [1, 2, 3, 4, 5],
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
            "col5": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col6": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col7": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        schema={
            "col0": pl.Int32,
            "col1": pl.Int64,
            "col2": pl.String,
            "col3": pl.String,
            "col4": pl.String,
            "col5": pl.Float64,
            "col6": pl.Decimal,
            "col7": pl.Float32,
        },
    )


#######################################
#     Tests for find_columns_type     #
#######################################


def test_find_columns_type_str(dataframe: pl.DataFrame) -> None:
    assert find_columns_type(dataframe, str) == ("col2", "col3", "col4")


def test_find_columns_type_int(dataframe: pl.DataFrame) -> None:
    assert find_columns_type(dataframe, int) == ("col0", "col1")


def test_find_columns_type_float(dataframe: pl.DataFrame) -> None:
    assert find_columns_type(dataframe, float) == ("col5", "col7")


def test_find_columns_type_empty() -> None:
    assert find_columns_type(pl.DataFrame({}), str) == ()


##########################################
#     Tests for find_columns_decimal     #
##########################################


def test_find_columns_decimal(dataframe: pl.DataFrame) -> None:
    assert find_columns_decimal(dataframe) == ("col6",)


def test_find_columns_decimal_empty() -> None:
    assert find_columns_decimal(pl.DataFrame({})) == ()


######################################
#     Tests for find_columns_str     #
######################################


def test_find_columns_str(dataframe: pl.DataFrame) -> None:
    assert find_columns_str(dataframe) == ("col2", "col3", "col4")


def test_find_columns_str_empty() -> None:
    assert find_columns_str(pl.DataFrame({})) == ()
