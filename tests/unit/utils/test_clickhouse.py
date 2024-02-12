from __future__ import annotations

from unittest.mock import Mock

import pyarrow as pa
import pytest

from flamme.testing import clickhouse_connect_available
from flamme.utils.clickhouse import get_table_schema
from flamme.utils.imports import is_clickhouse_connect_available

if is_clickhouse_connect_available():  # pragma: no cover
    from clickhouse_connect.driver import Client


@pytest.fixture()
def table() -> pa.Table:
    return pa.Table.from_pydict(
        {
            "number": pa.array([2, 4, 5, 100], type=pa.int32()),
            "string": pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"]),
        }
    )


######################################
#     Tests for get_table_schema     #
######################################


@clickhouse_connect_available
def test_get_table_schema(table: pa.Table) -> None:
    client = Mock(spec=Client, query_arrow=Mock(return_value=table))
    assert get_table_schema(client, "source.table") == pa.schema(
        [("number", pa.int32()), ("string", pa.string())]
    )
