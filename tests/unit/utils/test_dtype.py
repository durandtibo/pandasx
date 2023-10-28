from __future__ import annotations

import numpy as np
from pandas import DataFrame

from flamme.utils.dtype import column_types

##################################
#     Tests for column_types     #
##################################


def test_column_types() -> None:
    assert column_types(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    ) == {"float": {float}, "int": {float}, "str": {str, type(None), float}}


def test_column_types_empty() -> None:
    assert column_types(DataFrame({})) == {}
