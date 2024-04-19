r"""Contain schema readers."""

from __future__ import annotations

__all__ = [
    "BaseSchemaReader",
    "ParquetSchemaReader",
    "SchemaReader",
    "is_schema_reader_config",
    "setup_schema_reader",
]

from flamme.schema.reader.base import (
    BaseSchemaReader,
    is_schema_reader_config,
    setup_schema_reader,
)
from flamme.schema.reader.parquet import ParquetSchemaReader
from flamme.schema.reader.vanilla import SchemaReader
