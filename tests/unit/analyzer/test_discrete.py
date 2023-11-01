from __future__ import annotations

import numpy as np
from coola import objects_are_equal
from pandas import DataFrame

from flamme.analyzer import DiscreteDistributionAnalyzer
from flamme.section import DiscreteDistributionSection, EmptySection

##################################################
#     Tests for DiscreteDistributionAnalyzer     #
##################################################


def test_discrete_distribution_analyzer_str() -> None:
    assert str(DiscreteDistributionAnalyzer(column="col")).startswith(
        "DiscreteDistributionAnalyzer("
    )


def test_discrete_distribution_analyzer_get_statistics() -> None:
    section = DiscreteDistributionAnalyzer(column="int").analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, DiscreteDistributionSection)
    stats = section.get_statistics()
    assert stats["nunique"] == 3
    assert stats["total"] == 4


def test_discrete_distribution_analyzer_get_statistics_dropna_true() -> None:
    section = DiscreteDistributionAnalyzer(column="int", dropna=True).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, DiscreteDistributionSection)
    assert objects_are_equal(
        section.get_statistics(),
        {"most_common": [(1.0, 2), (0.0, 1)], "nunique": 2, "total": 3},
    )


def test_discrete_distribution_analyzer_get_statistics_empty_no_row() -> None:
    section = DiscreteDistributionAnalyzer(column="int").analyze(
        DataFrame({"float": np.array([]), "int": np.array([]), "str": np.array([])})
    )
    assert isinstance(section, DiscreteDistributionSection)
    assert objects_are_equal(
        section.get_statistics(), {"most_common": [], "nunique": 0, "total": 0}
    )


def test_discrete_distribution_analyzer_get_statistics_empty_no_column() -> None:
    section = DiscreteDistributionAnalyzer(column="col").analyze(DataFrame({}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
