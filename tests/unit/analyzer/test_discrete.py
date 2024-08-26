from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnDiscreteAnalyzer
from flamme.section import ColumnDiscreteSection, EmptySection


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame({"col": [1, 42, None, 22]}, schema={"col": pl.Int64})


############################################
#     Tests for ColumnDiscreteAnalyzer     #
############################################


def test_column_discrete_analyzer_str() -> None:
    assert str(ColumnDiscreteAnalyzer(column="col")).startswith("ColumnDiscreteAnalyzer(")


def test_column_discrete_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = ColumnDiscreteAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_discrete_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnDiscreteAnalyzer(column="col", figsize=figsize).analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.figsize == figsize


def test_column_discrete_analyzer_yscale_default(dataframe: pl.DataFrame) -> None:
    section = ColumnDiscreteAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_discrete_analyzer_yscale(dataframe: pl.DataFrame, yscale: str) -> None:
    section = ColumnDiscreteAnalyzer(column="col", yscale=yscale).analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.yscale == yscale


def test_column_discrete_analyzer_analyze() -> None:
    section = ColumnDiscreteAnalyzer(column="int").analyze(
        pl.DataFrame(
            {
                "float": [1.2, 4.2, None, 2.2],
                "int": [None, 1, 0, 1],
                "str": ["A", "B", None, None],
            },
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    )
    assert isinstance(section, ColumnDiscreteSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"most_common": [(1, 2), (None, 1), (0, 1)], "null_values": 1, "nunique": 3, "total": 4},
    )


def test_column_discrete_analyzer_analyze_drop_nulls_true() -> None:
    section = ColumnDiscreteAnalyzer(column="int", drop_nulls=True).analyze(
        pl.DataFrame(
            {
                "float": [1.2, 4.2, None, 2.2],
                "int": [None, 1, 0, 1],
                "str": ["A", "B", None, None],
            }
        )
    )
    assert isinstance(section, ColumnDiscreteSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"most_common": [(1, 2), (0, 1)], "null_values": 0, "nunique": 2, "total": 3},
    )


def test_column_discrete_analyzer_analyze_empty_no_row() -> None:
    section = ColumnDiscreteAnalyzer(column="int").analyze(
        pl.DataFrame({"float": [], "int": [], "str": []})
    )
    assert isinstance(section, ColumnDiscreteSection)
    assert objects_are_equal(
        section.get_statistics(), {"most_common": [], "null_values": 0, "nunique": 0, "total": 0}
    )


def test_column_discrete_analyzer_analyze_empty_no_column() -> None:
    section = ColumnDiscreteAnalyzer(column="col").analyze(pl.DataFrame({}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
