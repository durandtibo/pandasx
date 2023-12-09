from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from flamme.transformer.series import Sequential, StripString, ToNumeric

#################################################
#     Tests for SequentialSeriesTransformer     #
#################################################


def test_sequential_series_transformer_str() -> None:
    assert str(Sequential([StripString(), ToNumeric()])).startswith("SequentialSeriesTransformer(")


def test_sequential_series_transformer_str_empty() -> None:
    assert str(Sequential([])) == "SequentialSeriesTransformer()"


def test_sequential_series_transformer_transform_1() -> None:
    series = pd.Series(["abc", "2 ", " 3 ", "4", "5"])
    transformer = Sequential([ToNumeric(errors="coerce")])
    series = transformer.transform(series)
    assert_series_equal(series, pd.Series([float("nan"), 2.0, 3.0, 4.0, 5.0]))


def test_sequential_series_transformer_transform_2() -> None:
    series = pd.Series([" 1", "2 ", " 3 ", "4", "5"])
    transformer = Sequential([StripString(), ToNumeric()])
    series = transformer.transform(series)
    assert_series_equal(series, pd.Series([1, 2, 3, 4, 5]))
