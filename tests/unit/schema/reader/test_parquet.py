from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pytest
from coola import objects_are_equal

from flamme.schema.reader import ParquetSchemaReader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("data.parquet")
    frame = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    frame.to_parquet(path)
    return path


#########################################
#     Tests for ParquetSchemaReader     #
#########################################


def test_parquet_schema_reader_str(frame_path: Path) -> None:
    assert str(ParquetSchemaReader(frame_path)).startswith("ParquetSchemaReader(")


def test_parquet_schema_reader_read(frame_path: Path) -> None:
    schema = ParquetSchemaReader(frame_path).read()
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
