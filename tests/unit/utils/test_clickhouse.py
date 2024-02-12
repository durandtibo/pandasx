from __future__ import annotations

from unittest.mock import Mock

import pyarrow
import pytest
from clickhouse_connect.driver import Client

from flamme.testing import clickhouse_connect_available
from flamme.utils.clickhouse import get_table_schema


@pytest.fixture()
def table() -> pyarrow.Table:
    return pyarrow.Table.from_pydict(
        {
            "number": pyarrow.array([2, 4, 5, 100], type=pyarrow.int32()),
            "string": pyarrow.array(["Flamingo", "Horse", "Brittle stars", "Centipede"]),
        }
    )


######################################
#     Tests for get_table_schema     #
######################################


@clickhouse_connect_available
def test_get_table_schema(table: pyarrow.Table) -> None:
    client = Mock(spec=Client, query_arrow=Mock(return_value=table))
    assert get_table_schema(client, "source.table") == pyarrow.schema(
        [("number", pyarrow.int32()), ("string", pyarrow.string())]
    )
