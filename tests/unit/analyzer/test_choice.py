from __future__ import annotations

import numpy as np
from coola import objects_are_allclose
from pandas import DataFrame

from flamme.analyzer import ChoiceAnalyzer, DuplicatedRowAnalyzer, NullValueAnalyzer
from flamme.section import DuplicatedRowSection, NullValueSection, SectionDict

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
