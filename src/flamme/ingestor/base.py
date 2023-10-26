from __future__ import annotations

__all__ = ["BaseIngestor"]

import logging
from abc import ABC

from objectory import AbstractFactory
from pandas import DataFrame

logger = logging.getLogger(__name__)


class BaseIngestor(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement a DataFrame ingestor.

    Example usage:

    .. code-block:: pycon

        >>> from flamme.ingestor import ParquetIngestor
        >>> ingestor = ParquetIngestor(path="/path/to/df.parquet")
        >>> ingestor
        ParquetIngestor(path=/path/to/df.parquet)
        >>> df = ingestor.ingest()  # doctest: +SKIP
    """

    def ingest(self) -> DataFrame:
        r"""Ingests a DataFrame.

        Returns:
            ``pandas.DataFrame``: The ingested DataFrame.

        Example usage:

        .. code-block:: pycon

            >>> from flamme.ingestor import ParquetIngestor
            >>> ingestor = ParquetIngestor(path="/path/to/df.parquet")
            >>> df = ingestor.ingest()  # doctest: +SKIP
        """
