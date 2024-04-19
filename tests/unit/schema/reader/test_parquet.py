from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from coola import objects_are_equal
from pandas import DataFrame

from flamme.schema.reader import ParquetSchemaReader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def df_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("data.parquet")
    df = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    df.to_parquet(path)
    return path


#########################################
#     Tests for ParquetSchemaReader     #
#########################################


def test_parquet_schema_reader_str(df_path: Path) -> None:
    assert str(ParquetSchemaReader(df_path)).startswith("ParquetSchemaReader(")


def test_parquet_schema_reader_read(df_path: Path) -> None:
    schema = ParquetSchemaReader(df_path).read()
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
