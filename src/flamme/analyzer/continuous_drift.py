r"""Implement continuous values analyzers."""

from __future__ import annotations

__all__ = ["ColumnContinuousTemporalDriftAnalyzer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import ColumnContinuousTemporalDriftSection, EmptySection

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ColumnContinuousTemporalDriftAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer to show the temporal drift of a column with
    continuous values.

    Args:
        column: The column name.
        dt_column: The datetime column used to analyze
            the temporal distribution.
        period: The temporal period e.g. monthly or daily.
        nbins: The number of bins in the histogram.
        density: If True, draw and return a probability density:
            each bin will display the bin's raw count divided by the
            total number of counts and the bin width, so that the area
            under the histogram integrates to 1.
        yscale: The y-axis scale. If ``'auto'``, the
            ``'linear'`` or ``'log'/'symlog'`` scale is chosen based
            on the distribution.
        xmin: The minimum value of the range or its
            associated quantile. ``q0.1`` means the 10% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        xmax: The maximum value of the range or its
            associated quantile. ``q0.9`` means the 90% quantile.
            ``0`` is the minimum value and ``1`` is the maximum value.
        figsize: The figure size in inches. The first
            dimension is the width and the second is the height.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.analyzer import ColumnContinuousTemporalDriftAnalyzer
    >>> analyzer = ColumnContinuousTemporalDriftAnalyzer(
    ...     column="col", dt_column="date", period="M"
    ... )
    >>> analyzer
    ColumnContinuousAdvancedAnalyzer(column=float, nbins=None, yscale=auto, figsize=None)
    >>> rng = np.random.default_rng()
    >>> frame = pd.DataFrame(
    ...     {
    ...         "col": rng.standard_normal(10),
    ...         "date": pd.date_range(start="2017-01-01", periods=10, freq="1D"),
    ...     }
    ... )
    >>> section = analyzer.analyze(frame)

    ```
    """

    def __init__(
        self,
        column: str,
        dt_column: str,
        period: str,
        nbins: int | None = None,
        density: bool = False,
        yscale: str = "auto",
        xmin: float | str | None = None,
        xmax: float | str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self._column = column
        self._dt_column = dt_column
        self._period = period
        self._nbins = nbins
        self._density = density
        self._yscale = yscale
        self._xmin = xmin
        self._xmax = xmax
        self._figsize = figsize

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "column": self._column,
                    "dt_column": self._dt_column,
                    "period": self._period,
                    "nbins": self._nbins,
                    "density": self._density,
                    "yscale": self._yscale,
                    "xmin": self._xmin,
                    "xmax": self._xmax,
                    "figsize": self._figsize,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def analyze(self, frame: pd.DataFrame) -> ColumnContinuousTemporalDriftSection | EmptySection:
        logger.info(f"Analyzing the temporal drift of {self._column}")
        for column in [self._column, self._dt_column]:
            if column not in frame:
                logger.info(
                    f"Skipping temporal drift analysis because the column ({column}) is not "
                    f"in the DataFrame"
                )
                return EmptySection()
        if self._column == self._dt_column:
            logger.info(
                "Skipping temporal continuous distribution analysis because the datetime column "
                f"({self._column}) is the column to analyze"
            )
            return EmptySection()
        return ColumnContinuousTemporalDriftSection(
            frame=frame,
            column=self._column,
            dt_column=self._dt_column,
            period=self._period,
            nbins=self._nbins,
            yscale=self._yscale,
            figsize=self._figsize,
            xmin=self._xmin,
            xmax=self._xmax,
            density=self._density,
        )
