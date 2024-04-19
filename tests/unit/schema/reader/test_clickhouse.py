from __future__ import annotations

from unittest.mock import Mock

import pyarrow as pa
import pytest
from coola import objects_are_equal

from flamme.schema.reader import ClickHouseSchemaReader
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


############################################
#     Tests for ClickHouseSchemaReader     #
############################################


def test_parquet_schema_reader_str() -> None:
    assert str(
        ClickHouseSchemaReader(query="select * from source.dataset", client=Mock(spec=Client))
    ).startswith("ClickHouseSchemaReader(")


def test_parquet_schema_reader_read(table: pa.Table) -> None:
    schema = ClickHouseSchemaReader(
        query="select * from source.dataset",
        client=Mock(spec=Client, query_arrow=Mock(return_value=table)),
    ).read()
    assert objects_are_equal(
        schema,
        pa.schema(
            [
                ("col1", pa.int64()),
                ("col2", pa.string()),
                ("col3", pa.float64()),
            ]
        ),
    )
