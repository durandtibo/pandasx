r"""Implement an analyzer that generates a section to analyze the number
of null values for each column."""

from __future__ import annotations

__all__ = ["ColumnTemporalNullValueAnalyzer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import ColumnTemporalNullValueSection, EmptySection

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

logger = logging.getLogger(__name__)


class ColumnTemporalNullValueAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer to show the temporal distribution of null
    values for all columns.

    A plot is generated for each column.

    Args:
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        columns: The list of columns to analyze. A plot is generated
            for each column. ``None`` means all the columns.
        ncols: The number of columns.
        figsize: The figure size in inches. The first
            dimension is the width and the second is the height.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.analyzer import ColumnTemporalNullValueAnalyzer
    >>> analyzer = ColumnTemporalNullValueAnalyzer("datetime", period="M")
    >>> analyzer
    ColumnTemporalNullValueAnalyzer(
      (columns): None
      (dt_column): datetime
      (period): M
      (ncols): 2
      (figsize): (7, 5)
    )
    >>> frame = pd.DataFrame(
    ...     {
    ...         "int": np.array([np.nan, 1, 0, 1]),
    ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
    ...         "str": np.array(["A", "B", None, np.nan]),
    ...         "datetime": pd.to_datetime(
    ...             ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
    ...         ),
    ...     }
    ... )
    >>> section = analyzer.analyze(frame)

    ```
    """

    def __init__(
        self,
        dt_column: str,
        period: str,
        columns: Sequence[str] | None = None,
        ncols: int = 2,
        figsize: tuple[float, float] = (7, 5),
    ) -> None:
        self._dt_column = dt_column
        self._period = period
        self._columns = columns
        self._ncols = ncols
        self._figsize = figsize

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "columns": self._columns,
                    "dt_column": self._dt_column,
                    "period": self._period,
                    "ncols": self._ncols,
                    "figsize": self._figsize,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def analyze(self, frame: pd.DataFrame) -> ColumnTemporalNullValueSection | EmptySection:
        logger.info(
            "Analyzing the temporal null value distribution of all columns | "
            f"datetime column: {self._dt_column} | period: {self._period}"
        )
        if self._dt_column not in frame:
            logger.info(
                "Skipping monthly null value analysis because the datetime column "
                f"({self._dt_column}) is not in the DataFrame: {sorted(frame.columns)}"
            )
            return EmptySection()
        columns = self._columns
        if columns is None:
            # Exclude the datetime column because it does not make sense to analyze it because
            # we cannot know the date/time if the value is null.
            columns = sorted([col for col in frame.columns if col != self._dt_column])
        return ColumnTemporalNullValueSection(
            frame=frame,
            columns=columns,
            dt_column=self._dt_column,
            period=self._period,
            ncols=self._ncols,
            figsize=self._figsize,
        )
