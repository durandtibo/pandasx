from __future__ import annotations

import numpy as np
import pandas as pd

from flamme.analyzer import MarkdownAnalyzer
from flamme.section import MarkdownSection

######################################
#     Tests for MarkdownAnalyzer     #
######################################


def test_markdown_analyzer_str() -> None:
    assert str(MarkdownAnalyzer(desc="hello cats!")).startswith("MarkdownAnalyzer(")


def test_markdown_analyzer_get_statistics() -> None:
    section = MarkdownAnalyzer(desc="hello cats!").analyze(
        pd.DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, MarkdownSection)
    assert section.get_statistics() == {}


def test_markdown_analyzer_get_statistics_empty() -> None:
    section = MarkdownAnalyzer(desc="hello cats!").analyze(pd.DataFrame({}))
    assert isinstance(section, MarkdownSection)
    assert section.get_statistics() == {}
