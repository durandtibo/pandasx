from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from flamme.utils.parquet import get_table_schema

if TYPE_CHECKING:
    from pathlib import Path


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


def test_get_table_schema(table: pa.Table, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.parquet")
    pa.parquet.write_table(table=table, where=path)
    assert get_table_schema(path) == pa.schema([("number", pa.int32()), ("string", pa.string())])
