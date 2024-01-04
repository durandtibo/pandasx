from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from flamme.transformer.series import StripString

##################################################
#     Tests for StripStringSeriesTransformer     #
##################################################


def test_strip_string_series_transformer_str() -> None:
    assert str(StripString()) == "StripStringSeriesTransformer()"


def test_strip_string_series_transformer_transform() -> None:
    transformer = StripString()
    series = transformer.transform(pd.Series(["a ", " b", "  c  ", " d ", "e"]))
    assert_series_equal(series, pd.Series(["a", "b", "c", "d", "e"]))


def test_strip_string_series_transformer_transform_none() -> None:
    transformer = StripString()
    series = transformer.transform(pd.Series(["a ", " b", "  c  ", " d ", "e", None, 42, 4.2]))
    assert_series_equal(series, pd.Series(["a", "b", "c", "d", "e", None, 42, 4.2]))


def test_strip_string_series_transformer_transform_empty() -> None:
    transformer = StripString()
    series = transformer.transform(pd.Series([], dtype=object))
    assert_series_equal(series, pd.Series([], dtype=object))
