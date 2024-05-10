r"""Implement an analyzer to analyze only a subset of the columns."""

from __future__ import annotations

__all__ = ["ContentAnalyzer"]

import logging
from typing import TYPE_CHECKING

from flamme.analyzer.base import BaseAnalyzer
from flamme.section import ContentSection

if TYPE_CHECKING:

    import pandas as pd

    from flamme.section import BaseSection

logger = logging.getLogger(__name__)


class ContentAnalyzer(BaseAnalyzer):
    r"""Implement an analyzer that generates the given custom content.

    Args:
        content: The content to use in the HTML code.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flamme.analyzer import ContentAnalyzer
    >>> analyzer = ContentAnalyzer(content="meow")
    >>> analyzer
    ContentAnalyzer()
    >>> frame = pd.DataFrame(
    ...     {
    ...         "int": np.array([np.nan, 1, 0, 1]),
    ...         "float": np.array([1.2, 4.2, np.nan, 2.2]),
    ...         "str": np.array(["A", "B", None, np.nan]),
    ...     }
    ... )
    >>> section = analyzer.analyze(frame)

    ```
    """

    def __init__(self, content: str) -> None:
        self._content = str(content)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def analyze(self, frame: pd.DataFrame) -> BaseSection:  # noqa: ARG002
        logger.info("Generating the given custom content...")
        return ContentSection(content=self._content)
