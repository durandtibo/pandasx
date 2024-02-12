from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow
import pytest

from flamme.utils.parquet import get_table_schema

if TYPE_CHECKING:
    from pathlib import Path


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


def test_get_table_schema(table: pyarrow.Table, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.parquet")
    pyarrow.parquet.write_table(table=table, where=path)
    assert get_table_schema(path) == pyarrow.schema(
        [("number", pyarrow.int32()), ("string", pyarrow.string())]
    )
