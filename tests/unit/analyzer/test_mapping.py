import numpy as np
from coola import objects_are_allclose
from pandas import DataFrame

from flamme.analyzer import MappingAnalyzer, NullValueAnalyzer
from flamme.section import SectionDict

#####################################
#     Tests for MappingAnalyzer     #
#####################################


def test_mapping_analyzer_str() -> None:
    assert str(MappingAnalyzer({})).startswith("MappingAnalyzer(")


def test_mapping_analyzer_get_statistics() -> None:
    section = MappingAnalyzer(
        {
            "section1": NullValueAnalyzer(),
            "section2": NullValueAnalyzer(),
        }
    ).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
            }
        )
    )
    assert isinstance(section, SectionDict)
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "section1": {
                "columns": ("float", "int", "str"),
                "null_count": (1, 1, 2),
                "total_count": (4, 4, 4),
            },
            "section2": {
                "columns": ("float", "int", "str"),
                "null_count": (1, 1, 2),
                "total_count": (4, 4, 4),
            },
        },
    )


def test_mapping_analyzer_get_statistics_empty() -> None:
    section = MappingAnalyzer(
        {
            "section1": NullValueAnalyzer(),
            "section2": NullValueAnalyzer(),
        }
    ).analyze(
        DataFrame(
            {
                "float": np.array([]),
                "int": np.array([]),
                "str": np.array([]),
            }
        )
    )
    assert isinstance(section, SectionDict)
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "section1": {
                "columns": ("float", "int", "str"),
                "null_count": (0, 0, 0),
                "total_count": (0, 0, 0),
            },
            "section2": {
                "columns": ("float", "int", "str"),
                "null_count": (0, 0, 0),
                "total_count": (0, 0, 0),
            },
        },
    )
