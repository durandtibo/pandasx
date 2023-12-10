from __future__ import annotations

from objectory import OBJECT_TARGET
from pytest import mark

from flamme.transformer.series import BaseSeriesTransformer, StripString, ToNumeric
from flamme.utils.columns_ import Column, NumericColumn, StringColumn

############################
#     Tests for Column     #
############################


def test_column_str() -> None:
    assert str(Column(can_be_null=False, transformer=ToNumeric())).startswith("Column(")


@mark.parametrize("can_be_null", (True, False))
def test_column_can_be_null(can_be_null: bool) -> None:
    assert Column(can_be_null=can_be_null, transformer=ToNumeric()).can_be_null == can_be_null


@mark.parametrize(
    "transformer",
    (ToNumeric(), StripString(), {OBJECT_TARGET: "flamme.transformer.series.ToNumeric"}),
)
def test_column_get_transformer(transformer: BaseSeriesTransformer | dict) -> None:
    assert isinstance(
        Column(can_be_null=False, transformer=transformer).get_transformer(), BaseSeriesTransformer
    )


##################################
#     Tests for NumericColumn     #
##################################


@mark.parametrize(
    "transformer",
    (ToNumeric(), StripString(), {OBJECT_TARGET: "flamme.transformer.series.ToNumeric"}),
)
def test_numeric_column_get_transformer(transformer: BaseSeriesTransformer | dict) -> None:
    assert isinstance(
        NumericColumn(can_be_null=False, transformer=transformer).get_transformer(),
        BaseSeriesTransformer,
    )


def test_numeric_column_get_transformer_default() -> None:
    assert isinstance(NumericColumn(can_be_null=False).get_transformer(), ToNumeric)


##################################
#     Tests for StringColumn     #
##################################


@mark.parametrize(
    "transformer",
    (ToNumeric(), StripString(), {OBJECT_TARGET: "flamme.transformer.series.ToNumeric"}),
)
def test_string_column_get_transformer(transformer: BaseSeriesTransformer | dict) -> None:
    assert isinstance(
        StringColumn(can_be_null=False, transformer=transformer).get_transformer(),
        BaseSeriesTransformer,
    )


def test_string_column_get_transformer_default() -> None:
    assert isinstance(StringColumn(can_be_null=False).get_transformer(), StripString)
