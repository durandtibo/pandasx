from __future__ import annotations

import polars as pl

from flamme.analyzer import MarkdownAnalyzer
from flamme.section import MarkdownSection
from flamme.testing import markdown_available

######################################
#     Tests for MarkdownAnalyzer     #
######################################


@markdown_available
def test_markdown_analyzer_str() -> None:
    assert str(MarkdownAnalyzer(desc="hello cats!")).startswith("MarkdownAnalyzer(")


@markdown_available
def test_markdown_analyzer_get_statistics() -> None:
    section = MarkdownAnalyzer(desc="hello cats!").analyze(
        pl.DataFrame(
            {
                "col1": [1.2, 4.2, 4.2, 2.2],
                "col2": [1, 1, 1, 1],
                "col3": [1, 2, 2, 2],
            },
            schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        )
    )
    assert isinstance(section, MarkdownSection)
    assert section.get_statistics() == {}


@markdown_available
def test_markdown_analyzer_get_statistics_empty_rows() -> None:
    section = MarkdownAnalyzer(desc="hello cats!").analyze(
        pl.DataFrame(
            {"col1": [], "col2": [], "col3": []},
            schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        )
    )
    assert isinstance(section, MarkdownSection)
    assert section.get_statistics() == {}


@markdown_available
def test_markdown_analyzer_get_statistics_empty() -> None:
    section = MarkdownAnalyzer(desc="hello cats!").analyze(pl.DataFrame({}))
    assert isinstance(section, MarkdownSection)
    assert section.get_statistics() == {}
