from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from flamme.transformer.series import ToNumeric

################################################
#     Tests for ToNumericSeriesTransformer     #
################################################


def test_to_numeric_series_transformer_str() -> None:
    assert str(ToNumeric()) == "ToNumericSeriesTransformer()"


def test_to_numeric_series_transformer_str_kwargs() -> None:
    assert str(ToNumeric(errors="ignore")) == "ToNumericSeriesTransformer(errors=ignore)"


def test_to_numeric_series_transformer_transform() -> None:
    transformer = ToNumeric()
    series = transformer.transform(pd.Series(["1", "2", "3", "4", 5]))
    assert_series_equal(series, pd.Series([1, 2, 3, 4, 5]))


def test_to_numeric_series_transformer_transform_kwargs() -> None:
    transformer = ToNumeric(errors="coerce")
    series = transformer.transform(pd.Series(["1", "2", "3", "4", "a5"]))
    assert_series_equal(series, pd.Series([1, 2, 3, 4, float("nan")]))
