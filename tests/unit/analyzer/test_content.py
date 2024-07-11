from __future__ import annotations

import polars as pl

from flamme.analyzer import ContentAnalyzer
from flamme.section import ContentSection

#####################################
#     Tests for ContentAnalyzer     #
#####################################


def test_content_analyzer_str() -> None:
    assert str(ContentAnalyzer(content="meow")).startswith("ContentAnalyzer(")


def test_content_analyzer_get_statistics() -> None:
    section = ContentAnalyzer(content="meow").analyze(
        pl.DataFrame(
            {
                "col1": [1.2, 4.2, 4.2, 2.2],
                "col2": [1, 1, 1, 1],
                "col3": [1, 2, 2, 2],
            },
            schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        )
    )
    assert isinstance(section, ContentSection)
    assert section.get_statistics() == {}


def test_content_analyzer_get_statistics_empty_rows() -> None:
    section = ContentAnalyzer(content="meow").analyze(
        pl.DataFrame(
            {"col1": [], "col2": [], "col3": []},
            schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        )
    )
    assert isinstance(section, ContentSection)
    assert section.get_statistics() == {}


def test_content_analyzer_get_statistics_empty() -> None:
    section = ContentAnalyzer(content="meow").analyze(pl.DataFrame({}))
    assert isinstance(section, ContentSection)
    assert section.get_statistics() == {}
