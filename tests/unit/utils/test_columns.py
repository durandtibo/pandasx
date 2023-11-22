from __future__ import annotations

import pandas as pd

from flamme.utils.columns import find_columns_str, find_columns_type

#######################################
#     Tests for find_columns_type     #
#######################################


def test_find_columns_type_str() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    assert find_columns_type(df, str) == ("col2", "col3", "col4")


def test_find_columns_type_int() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    assert find_columns_type(df, int) == ("col1",)


def test_find_columns_type_float() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    assert find_columns_type(df, float) == tuple()


def test_find_columns_type_empty() -> None:
    assert find_columns_type(pd.DataFrame({}), str) == tuple()


#######################################
#     Tests for find_columns_str     #
#######################################


def test_find_columns_str() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    assert find_columns_str(df) == ("col2", "col3", "col4")


def test_find_columns_str_empty() -> None:
    assert find_columns_str(pd.DataFrame({})) == tuple()
