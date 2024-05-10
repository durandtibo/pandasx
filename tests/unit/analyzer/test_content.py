from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flamme.analyzer import ContentAnalyzer
from flamme.section import ContentSection


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col1": np.array([1.2, 4.2, 4.2, 2.2]),
            "col2": np.array([1, 1, 1, 1]),
            "col3": np.array([1, 2, 2, 2]),
        }
    )


#####################################
#     Tests for ContentAnalyzer     #
#####################################


def test_content_analyzer_str() -> None:
    assert str(ContentAnalyzer(content="meow")).startswith("ContentAnalyzer(")


def test_content_analyzer_get_statistics(dataframe: pd.DataFrame) -> None:
    section = ContentAnalyzer(content="meow").analyze(dataframe)
    assert isinstance(section, ContentSection)
    assert section.get_statistics() == {}


def test_content_analyzer_get_statistics_empty_rows() -> None:
    section = ContentAnalyzer(content="meow").analyze(pd.DataFrame({"col1": [], "col2": []}))
    assert isinstance(section, ContentSection)
    assert section.get_statistics() == {}
