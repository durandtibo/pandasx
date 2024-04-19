from __future__ import annotations

__all__ = ["SchemaToNumericDataFrameTransformerFactory"]

import logging


from flamme.factory.transformer.dataframe import BaseDataFrameTransformerFactory
from flamme.transformer.df import (
    BaseDataFrameTransformer,
    ToNumericDataFrameTransformer,
)
from flamme.utils.dtype import find_numeric_columns_from_dtypes, get_dtypes_from_schema
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)


class SchemaToNumericDataFrameTransformerFactory(BaseDataFrameTransformerFactory):
    r"""Implement a DataFrame transformer factory that automatically
    finds the numeric columns based on the schema and instantiate a
    ``ToNumericDataFrameTransformer``.

    Args:
        columns: The columns to convert.
        **kwargs: The keyword arguments for ``pandas.to_numeric``.
    """

    def __init__(self, schema: pa.Schema, **kwargs: Any) -> None:
        self._schema = schema
        self._kwargs = kwargs

    def create(self) -> BaseDataFrameTransformer:
        dtypes = get_dtypes_from_schema(self._schema)
        columns = sorted(find_numeric_columns_from_dtypes(dtypes))
        logger.info(f"found {len(columns):,} numeric columns: {columns}")
        return ToNumericDataFrameTransformer(columns=columns, **self._kwargs)
