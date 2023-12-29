from __future__ import annotations

import numpy as np
from coola import objects_are_allclose
from pandas import DataFrame
from pytest import mark

from flamme.analyzer import ChoiceAnalyzer, DuplicatedRowAnalyzer, NullValueAnalyzer
from flamme.analyzer.choice import NumUniqueSelection
from flamme.section import DuplicatedRowSection, NullValueSection

####################################
#     Tests for ChoiceAnalyzer     #
####################################


def selection_fn(df: DataFrame) -> str:
    return "null" if df.isnull().values.any() else "duplicate"


def test_mapping_analyzer_str() -> None:
    assert str(
        ChoiceAnalyzer(
            {
                "null": NullValueAnalyzer(),
                "duplicate": DuplicatedRowAnalyzer(),
            },
            selection_fn=selection_fn,
        )
    ).startswith("ChoiceAnalyzer(")


def test_mapping_analyzer_analyzers() -> None:
    analyzer = ChoiceAnalyzer(
        {
            "null": NullValueAnalyzer(),
            "duplicate": DuplicatedRowAnalyzer(),
        },
        selection_fn=selection_fn,
    )
    assert isinstance(analyzer.analyzers, dict)
    assert len(analyzer.analyzers) == 2
    assert isinstance(analyzer.analyzers["null"], NullValueAnalyzer)
    assert isinstance(analyzer.analyzers["duplicate"], DuplicatedRowAnalyzer)


def test_mapping_analyzer_get_statistics_null() -> None:
    section = ChoiceAnalyzer(
        {
            "null": NullValueAnalyzer(),
            "duplicate": DuplicatedRowAnalyzer(),
        },
        selection_fn=selection_fn,
    ).analyze(DataFrame({"col": np.asarray([1.2, 4.2, np.nan, 2.2])}))
    assert isinstance(section, NullValueSection)
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "columns": ("col",),
            "null_count": (1,),
            "total_count": (4,),
        },
    )


def test_mapping_analyzer_get_statistics_duplicate() -> None:
    section = ChoiceAnalyzer(
        {
            "null": NullValueAnalyzer(),
            "duplicate": DuplicatedRowAnalyzer(),
        },
        selection_fn=selection_fn,
    ).analyze(DataFrame({"col": np.asarray([1.2, 4.2, 1.2, 2.2])}))
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_allclose(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


########################################
#     Tests for NumUniqueSelection     #
########################################


def test_num_unique_selection_str() -> None:
    assert str(NumUniqueSelection(column="col")).startswith("NumUniqueSelection(")


@mark.parametrize(
    "df",
    [
        DataFrame({"col": []}),
        DataFrame({"col": np.ones(100)}),
        DataFrame({"col": np.arange(10)}),
        DataFrame({"col": np.arange(100)}),
    ],
)
def test_num_unique_selection_call_small(df: DataFrame) -> None:
    assert NumUniqueSelection(column="col")(df) == "small"


@mark.parametrize("df", [DataFrame({"col": np.arange(101)}), DataFrame({"col": np.arange(201)})])
def test_num_unique_selection_call_large(df: DataFrame) -> None:
    assert NumUniqueSelection(column="col")(df) == "large"


@mark.parametrize("small", ["discrete", "bear"])
@mark.parametrize("large", ["continuous", "meow"])
def test_num_unique_selection_small(small: str, large: str) -> None:
    select = NumUniqueSelection(column="col", small=small, large=large)
    assert select(DataFrame({"col": np.arange(10)})) == small
    assert select(DataFrame({"col": np.arange(101)})) == large


def test_num_unique_selection_call_threshold_10() -> None:
    select = NumUniqueSelection(column="col", threshold=10)
    assert select(DataFrame({"col": np.arange(10)})) == "small"
    assert select(DataFrame({"col": np.arange(11)})) == "large"
