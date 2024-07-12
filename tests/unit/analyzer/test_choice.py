from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose

from flamme.analyzer import ChoiceAnalyzer, DuplicatedRowAnalyzer, NullValueAnalyzer
from flamme.analyzer.choice import NumUniqueSelection
from flamme.section import DuplicatedRowSection, NullValueSection
from flamme.utils.null import compute_null_count

####################################
#     Tests for ChoiceAnalyzer     #
####################################


def selection_fn(frame: pl.DataFrame) -> str:
    return "null" if compute_null_count(frame).sum() > 0 else "duplicate"


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
    ).analyze(pl.DataFrame({"col": [1.2, 4.2, None, 2.2]}, schema={"col": pl.Float64}))
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
    ).analyze(pl.DataFrame({"col": [1.2, 4.2, 1.2, 2.2]}, schema={"col": pl.Float64}))
    assert isinstance(section, DuplicatedRowSection)
    assert objects_are_allclose(section.get_statistics(), {"num_rows": 4, "num_unique_rows": 3})


########################################
#     Tests for NumUniqueSelection     #
########################################


def test_num_unique_selection_str() -> None:
    assert str(NumUniqueSelection(column="col")).startswith("NumUniqueSelection(")


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"col": []}, schema={"col": pl.Float64}),
        pl.DataFrame({"col": [1] * 100}, schema={"col": pl.Float64}),
        pl.DataFrame({"col": list(range(10))}, schema={"col": pl.Float64}),
        pl.DataFrame({"col": list(range(100))}, schema={"col": pl.Float64}),
    ],
)
def test_num_unique_selection_call_small(frame: pl.DataFrame) -> None:
    assert NumUniqueSelection(column="col")(frame) == "small"


@pytest.mark.parametrize(
    "frame", [pl.DataFrame({"col": list(range(101))}), pl.DataFrame({"col": list(range(201))})]
)
def test_num_unique_selection_call_large(frame: pl.DataFrame) -> None:
    assert NumUniqueSelection(column="col")(frame) == "large"


@pytest.mark.parametrize("small", ["discrete", "bear"])
@pytest.mark.parametrize("large", ["continuous", "meow"])
def test_num_unique_selection_small(small: str, large: str) -> None:
    select = NumUniqueSelection(column="col", small=small, large=large)
    assert select(pl.DataFrame({"col": list(range(10))})) == small
    assert select(pl.DataFrame({"col": list(range(101))})) == large


def test_num_unique_selection_call_threshold_10() -> None:
    select = NumUniqueSelection(column="col", threshold=10)
    assert select(pl.DataFrame({"col": list(range(10))})) == "small"
    assert select(pl.DataFrame({"col": list(range(11))})) == "large"
