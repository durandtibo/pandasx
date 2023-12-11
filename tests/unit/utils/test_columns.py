from __future__ import annotations

from objectory import OBJECT_TARGET
from pytest import mark

from flamme.analyzer import (
    BaseAnalyzer,
    ColumnContinuousAnalyzer,
    ColumnDiscreteAnalyzer,
)
from flamme.transformer.series import BaseSeriesTransformer, StripString, ToNumeric
from flamme.utils.columns import Column

############################
#     Tests for Column     #
############################


def test_column_str() -> None:
    assert str(
        Column(
            can_be_null=False,
            analyzer=ColumnContinuousAnalyzer(column="col"),
            transformer=ToNumeric(),
        )
    ).startswith("Column(")


@mark.parametrize("can_be_null", (True, False))
def test_column_can_be_null(can_be_null: bool) -> None:
    assert (
        Column(
            can_be_null=can_be_null,
            analyzer=ColumnContinuousAnalyzer(column="col"),
            transformer=ToNumeric(),
        ).can_be_null
        == can_be_null
    )


@mark.parametrize(
    "analyzer",
    (
        ColumnContinuousAnalyzer(column="col"),
        ColumnDiscreteAnalyzer(column="col"),
        {OBJECT_TARGET: "flamme.analyzer.ColumnContinuousAnalyzer", "column": "col"},
    ),
)
def test_column_get_analyzer(analyzer: BaseAnalyzer | dict) -> None:
    assert isinstance(
        Column(can_be_null=False, analyzer=analyzer, transformer=ToNumeric()).get_analyzer(),
        BaseAnalyzer,
    )


@mark.parametrize(
    "transformer",
    (ToNumeric(), StripString(), {OBJECT_TARGET: "flamme.transformer.series.ToNumeric"}),
)
def test_column_get_transformer(transformer: BaseSeriesTransformer | dict) -> None:
    assert isinstance(
        Column(
            can_be_null=False,
            analyzer=ColumnContinuousAnalyzer(column="col"),
            transformer=transformer,
        ).get_transformer(),
        BaseSeriesTransformer,
    )
