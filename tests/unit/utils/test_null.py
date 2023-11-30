import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.utils.null import compute_null_per_col

##########################################
#     Tests for compute_null_per_col     #
##########################################


def test_compute_null_per_col() -> None:
    assert_frame_equal(
        compute_null_per_col(
            pd.DataFrame(
                {
                    "int": np.array([np.nan, 1, 0, 1]),
                    "float": np.array([1.2, 4.2, np.nan, 2.2]),
                    "str": np.array(["A", "B", None, np.nan]),
                }
            )
        ),
        pd.DataFrame(
            {
                "column": ["int", "float", "str"],
                "null": [1, 1, 2],
                "total": [4, 4, 4],
                "null_pct": [0.25, 0.25, 0.5],
            }
        ),
    )


def test_compute_null_per_col_empty_row() -> None:
    assert_frame_equal(
        compute_null_per_col(pd.DataFrame({"int": [], "float": [], "str": []})),
        pd.DataFrame(
            {
                "column": ["int", "float", "str"],
                "null": [0, 0, 0],
                "total": [0, 0, 0],
                "null_pct": [np.nan, np.nan, np.nan],
            }
        ),
    )


def test_compute_null_per_col_empty() -> None:
    assert_frame_equal(
        compute_null_per_col(pd.DataFrame({})),
        pd.DataFrame(
            {
                "column": [],
                "null": np.array([], dtype=int),
                "total": np.array([], dtype=int),
                "null_pct": np.array([], dtype=float),
            }
        ),
    )
