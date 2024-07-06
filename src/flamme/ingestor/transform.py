r"""Contain a wrapper around an ingestor to transform the data after
ingestion."""

from __future__ import annotations

__all__ = ["TransformedIngestor"]

import logging

import polars as pl
from coola.utils import str_indent, str_mapping

from flamme.ingestor.base import BaseIngestor, setup_ingestor
from flamme.transformer.dataframe.base import (
    BaseDataFrameTransformer,
    setup_dataframe_transformer,
)

logger = logging.getLogger(__name__)


class TransformedIngestor(BaseIngestor):
    r"""Implement an ingestor that also transforms the DataFrame.

    Args:
        ingestor: The base ingestor.
        transformer: Specifies a ``polars.DataFrame`` transformer or
            its configuration.

    Example usage:

    ```pycon

    >>> from flamme.ingestor import TransformedIngestor, ParquetIngestor
    >>> from flamme.transformer.dataframe import ToNumeric
    >>> ingestor = TransformedIngestor(
    ...     ingestor=ParquetIngestor(path="/path/to/frame.csv"),
    ...     transformer=ToNumeric(columns=["col1", "col3"]),
    ... )
    >>> ingestor
    TransformedIngestor(
      (ingestor): ParquetIngestor(path=/path/to/frame.csv)
      (transformer): ToNumericDataFrameTransformer(columns=('col1', 'col3'), ignore_missing=False)
    )
    >>> frame = ingestor.ingest()  # doctest: +SKIP

    ```
    """

    def __init__(
        self, ingestor: BaseIngestor | dict, transformer: BaseDataFrameTransformer | dict
    ) -> None:
        self._ingestor = setup_ingestor(ingestor)
        self._transformer = setup_dataframe_transformer(transformer)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"ingestor": self._ingestor, "transformer": self._transformer})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def ingest(self) -> pl.DataFrame:
        frame = self._ingestor.ingest().to_pandas()
        return pl.from_pandas(self._transformer.transform(frame))
