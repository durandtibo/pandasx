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
