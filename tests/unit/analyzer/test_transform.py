from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.transformer import BaseTransformer, SqlTransformer
from objectory import OBJECT_TARGET

from flamme.analyzer import BaseAnalyzer, NullValueAnalyzer, TransformAnalyzer
from flamme.section import NullValueSection

#######################################
#     Tests for TransformAnalyzer     #
#######################################


def test_transform_analyzer_repr() -> None:
    assert repr(
        TransformAnalyzer(
            transformer=SqlTransformer("SELECT * FROM self WHERE float > 1"),
            analyzer=NullValueAnalyzer(),
        )
    ).startswith("TransformAnalyzer(")


def test_transform_analyzer_str() -> None:
    assert str(
        TransformAnalyzer(
            transformer=SqlTransformer("SELECT * FROM self WHERE float > 1"),
            analyzer=NullValueAnalyzer(),
        )
    ).startswith("TransformAnalyzer(")


@pytest.mark.parametrize(
    "transformer",
    [
        SqlTransformer("SELECT * FROM self WHERE float > 1"),
        {
            OBJECT_TARGET: "grizz.transformer.SqlTransformer",
            "query": "SELECT * FROM self WHERE float > 1",
        },
    ],
)
@pytest.mark.parametrize(
    "analyzer",
    [
        NullValueAnalyzer(),
        {OBJECT_TARGET: "flamme.analyzer.NullValueAnalyzer"},
    ],
)
def test_transform_analyzer_get_statistics(
    transformer: BaseTransformer | dict, analyzer: BaseAnalyzer | dict
) -> None:
    section = TransformAnalyzer(
        transformer=transformer,
        analyzer=analyzer,
    ).analyze(
        pl.DataFrame(
            {
                "float": [1.2, 4.2, None, 2.2],
                "int": [None, 1, 0, 1],
                "str": ["A", "B", None, None],
            },
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (0, 1, 1),
            "total_count": (3, 3, 3),
        },
    )


def test_transform_analyzer_get_statistics_empty() -> None:
    section = TransformAnalyzer(
        transformer=SqlTransformer("SELECT * FROM self WHERE float > 1"),
        analyzer=NullValueAnalyzer(),
    ).analyze(
        pl.DataFrame(
            {"float": [], "int": [], "str": []},
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    )
    assert isinstance(section, NullValueSection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("float", "int", "str"),
            "null_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )
