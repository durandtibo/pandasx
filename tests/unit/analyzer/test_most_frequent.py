from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose, objects_are_equal

from flamme.analyzer import MostFrequentValuesAnalyzer
from flamme.section import EmptySection, MostFrequentValuesSection


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"col": [1.0, 42.0, None, 22.0, 1.0, 2.0, 1.0, None]}, schema={"col": pl.Float64}
    )


################################################
#     Tests for MostFrequentValuesAnalyzer     #
################################################


def test_most_frequent_values_analyzer_str() -> None:
    assert str(MostFrequentValuesAnalyzer(column="col")).startswith("MostFrequentValuesAnalyzer(")


def test_most_frequent_values_analyzer_get_statistics(dataframe: pl.DataFrame) -> None:
    section = MostFrequentValuesAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, MostFrequentValuesSection)
    assert objects_are_allclose(
        section.get_statistics(),
        {"most_common": [(1.0, 3), (None, 2), (42.0, 1), (22.0, 1), (2.0, 1)]},
    )


def test_most_frequent_values_analyzer_get_statistics_drop_nulls_true(
    dataframe: pl.DataFrame,
) -> None:
    section = MostFrequentValuesAnalyzer(column="col", drop_nulls=True).analyze(dataframe)
    assert isinstance(section, MostFrequentValuesSection)
    assert objects_are_equal(
        section.get_statistics(), {"most_common": [(1.0, 3), (42.0, 1), (22.0, 1), (2.0, 1)]}
    )


def test_most_frequent_values_analyzer_get_statistics_empty_no_row() -> None:
    section = MostFrequentValuesAnalyzer(column="col").analyze(
        pl.DataFrame({"col": []}, schema={"col": pl.Int64})
    )
    assert isinstance(section, MostFrequentValuesSection)
    assert objects_are_equal(section.get_statistics(), {"most_common": []})


def test_most_frequent_values_analyzer_get_statistics_empty_no_column() -> None:
    section = MostFrequentValuesAnalyzer(column="col").analyze(pl.DataFrame({}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
