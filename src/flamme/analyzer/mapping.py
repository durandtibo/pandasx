from __future__ import annotations

__all__ = ["MappingAnalyzer"]

from collections.abc import Mapping

from coola.utils import str_indent, str_mapping
from pandas import DataFrame

from flamme.analyzer.base import BaseAnalyzer, setup_analyzer
from flamme.section import SectionDict


class MappingAnalyzer(BaseAnalyzer):
    r"""Implements an analyzer to analyze multiple analyzers.

    Args:
    ----
        analyzers (``Mapping``): Specifies the mappings to analyze.
            The key of each analyzer is used to organize the metrics
            and report.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import pandas as pd
        >>> from flamme.analyzer import FilteredAnalyzer, NullValueAnalyzer, DuplicatedRowAnalyzer
        >>> analyzer = MappingAnalyzer(
        ...     {"null": NullValueAnalyzer(), "duplicate": DuplicatedRowAnalyzer()}
        ... )
        >>> analyzer
        MappingAnalyzer(
          (null): NullValueAnalyzer(figsize=None)
          (duplicate): DuplicatedRowAnalyzer(columns=None)
        )
        >>> df = pd.DataFrame(
        ...     {
        ...         "int": np.array([np.nan, 1, 0, 1]),
        ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
        ...         "str": np.array(["A", "B", None, np.nan]),
        ...     }
        ... )
        >>> section = analyzer.analyze(df)
    """

    def __init__(self, analyzers: Mapping[str, BaseAnalyzer | dict]) -> None:
        self._analyzers = {name: setup_analyzer(analyzer) for name, analyzer in analyzers.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._analyzers))}\n)"

    @property
    def analyzers(self) -> dict[str, BaseAnalyzer]:
        return self._analyzers

    def analyze(self, df: DataFrame) -> SectionDict:
        return SectionDict(
            {name: analyzer.analyze(df) for name, analyzer in self._analyzers.items()}
        )
