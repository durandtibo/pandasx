from __future__ import annotations

from decimal import Decimal

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.transformer.df import DecimalToNumeric

##########################################################
#     Tests for DecimalToNumericDataFrameTransformer     #
##########################################################


def test_decimal_to_numeric_dataframe_transformer_str() -> None:
    assert str(DecimalToNumeric()) == "DecimalToNumericDataFrameTransformer()"


def test_decimal_to_numeric_dataframe_transformer_str_kwargs() -> None:
    assert (
        str(DecimalToNumeric(errors="ignore"))
        == "DecimalToNumericDataFrameTransformer(errors=ignore)"
    )


def test_decimal_to_numeric_dataframe_transformer_transform() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, Decimal(5)],
            "col2": [Decimal(1), Decimal(2), Decimal(3), Decimal(4), Decimal(5)],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    transformer = DecimalToNumeric()
    df = transformer.transform(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )


def test_decimal_to_numeric_dataframe_transformer_transform_kwargs() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, Decimal(5)],
            "col2": [Decimal(1), Decimal(2), Decimal(3), Decimal(4), "nan"],
            "col3": ["1", "2", "3", "4", "5"],
            "col4": ["a", "b", "c", "d", "e"],
        }
    )
    transformer = DecimalToNumeric(errors="coerce")
    df = transformer.transform(df)
    assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, float("nan")],
                "col3": ["1", "2", "3", "4", "5"],
                "col4": ["a", "b", "c", "d", "e"],
            }
        ),
    )
