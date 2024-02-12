r"""Contain some clickhouse utility functions."""

from __future__ import annotations

__all__ = ["get_table_schema"]


from typing import TYPE_CHECKING

import pyarrow as pa

from flamme.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path


def get_table_schema(path: Path | str) -> pa.Schema:
    r"""Return the table schema.

    Args:
        path: Specifies the path to the parquet file.

    Returns:
        The table schema.
    """
    return pa.parquet.read_schema(sanitize_path(path))
