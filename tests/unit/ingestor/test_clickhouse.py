from __future__ import annotations

from unittest.mock import Mock

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from flamme.ingestor import ClickHouseIngestor
from flamme.testing import clickhouse_connect_available
from flamme.utils.imports import is_clickhouse_connect_available

if is_clickhouse_connect_available():
    from clickhouse_connect.driver import Client


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return pa.Table.from_pydict(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )


########################################
#     Tests for ClickHouseIngestor     #
########################################


@clickhouse_connect_available
def test_clickhouse_ingestor_str() -> None:
    assert str(
        ClickHouseIngestor(query="select * from source.dataset", client=Mock(spec=Client))
    ).startswith("ClickHouseIngestor(")


@clickhouse_connect_available
def test_clickhouse_ingestor_ingest(table: pa.Table) -> None:
    client_mock = Mock(spec=Client, query_arrow=Mock(return_value=table))
    ingestor = ClickHouseIngestor(query="select * from source.dataset", client=client_mock)
    out = ingestor.ingest()
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
    client_mock.query_arrow.assert_called_once_with(query="select * from source.dataset")
