from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from flamme.transformer.series import ToDatetime

#################################################
#     Tests for ToDatetimeSeriesTransformer     #
#################################################


def test_to_datetime_series_transformer_str() -> None:
    assert str(ToDatetime()) == "ToDatetimeSeriesTransformer()"


def test_to_datetime_series_transformer_str_kwargs() -> None:
    assert str(ToDatetime(errors="ignore")) == "ToDatetimeSeriesTransformer(errors=ignore)"


def test_to_datetime_series_transformer_transform() -> None:
    transformer = ToDatetime()
    out = transformer.transform(
        pd.Series(["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"])
    )
    assert_series_equal(
        out,
        pd.to_datetime(["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "2021-12-31"]).to_series(
            index=[0, 1, 2, 3, 4]
        ),
    )


def test_to_datetime_series_transformer_transform_kwargs() -> None:
    transformer = ToDatetime(errors="coerce")
    out = transformer.transform(
        pd.Series(["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", "abc"])
    )
    assert_series_equal(
        out,
        pd.to_datetime(
            ["2020-1-1", "2020-1-2", "2020-1-31", "2020-12-31", pd.NaT],
            errors="coerce",
        ).to_series(index=[0, 1, 2, 3, 4]),
    )
