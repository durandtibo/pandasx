r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = ["clickhouse_connect_available", "colorlog_available"]

import pytest

from flamme.utils.imports import is_clickhouse_connect_available, is_colorlog_available

clickhouse_connect_available = pytest.mark.skipif(
    not is_clickhouse_connect_available(), reason="requires clickhouse_connect"
)
colorlog_available = pytest.mark.skipif(not is_colorlog_available(), reason="requires colorlog")
