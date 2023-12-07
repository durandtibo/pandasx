from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series

from flamme.utils.dtype import df_column_types, series_column_types

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
