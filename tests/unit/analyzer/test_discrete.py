from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_equal

from flamme.analyzer import ColumnDiscreteAnalyzer, ColumnTemporalDiscreteAnalyzer
from flamme.section import (
    ColumnDiscreteSection,
    ColumnTemporalDiscreteSection,
    EmptySection,
)


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col": np.array([1, 42, np.nan, 22]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


############################################
#     Tests for ColumnDiscreteAnalyzer     #
############################################


def test_column_discrete_analyzer_str() -> None:
    assert str(ColumnDiscreteAnalyzer(column="col")).startswith("ColumnDiscreteAnalyzer(")


def test_column_discrete_analyzer_figsize_default(dataframe: pd.DataFrame) -> None:
    section = ColumnDiscreteAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_discrete_analyzer_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnDiscreteAnalyzer(column="col", figsize=figsize).analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.figsize == figsize


def test_column_discrete_analyzer_yscale_default(dataframe: pd.DataFrame) -> None:
    section = ColumnDiscreteAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_discrete_analyzer_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    section = ColumnDiscreteAnalyzer(column="col", yscale=yscale).analyze(dataframe)
    assert isinstance(section, ColumnDiscreteSection)
    assert section.yscale == yscale


def test_column_discrete_analyzer_analyze() -> None:
    section = ColumnDiscreteAnalyzer(column="int").analyze(
        pd.DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, ColumnDiscreteSection)
    stats = section.get_statistics()
    assert stats["nunique"] == 3
    assert stats["total"] == 4


def test_column_discrete_analyzer_analyze_dropna_true() -> None:
    section = ColumnDiscreteAnalyzer(column="int", dropna=True).analyze(
        pd.DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, ColumnDiscreteSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"most_common": [(1.0, 2), (0.0, 1)], "nunique": 2, "total": 3},
    )


def test_column_discrete_analyzer_analyze_empty_no_row() -> None:
    section = ColumnDiscreteAnalyzer(column="int").analyze(
        pd.DataFrame({"float": np.array([]), "int": np.array([]), "str": np.array([])})
    )
    assert isinstance(section, ColumnDiscreteSection)
    assert objects_are_equal(
        section.get_statistics(), {"most_common": [], "nunique": 0, "total": 0}
    )


def test_column_discrete_analyzer_analyze_empty_no_column() -> None:
    section = ColumnDiscreteAnalyzer(column="col").analyze(pd.DataFrame({}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


####################################################
#     Tests for ColumnTemporalDiscreteAnalyzer     #
####################################################


def test_column_temporal_discrete_analyzer_str() -> None:
    assert str(
        ColumnTemporalDiscreteAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnTemporalDiscreteAnalyzer(")


def test_column_temporal_discrete_analyzer_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.column == "col"


def test_column_temporal_discrete_analyzer_dt_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.dt_column == "datetime"


def test_column_temporal_discrete_analyzer_period(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.period == "M"


def test_column_temporal_discrete_analyzer_figsize_default(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_discrete_analyzer_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert section.figsize == figsize


def test_column_temporal_discrete_analyzer_analyze(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_empty() -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": [], "int": [], "str": [], "datetime": []}))
    assert isinstance(section, ColumnTemporalDiscreteSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_missing_column() -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"datetime": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_missing_dt_column() -> None:
    section = ColumnTemporalDiscreteAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_discrete_analyzer_analyze_same_column() -> None:
    section = ColumnTemporalDiscreteAnalyzer(column="col", dt_column="col", period="M").analyze(
        pd.DataFrame(
            {"col": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"])}
        )
    )
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
