from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from flamme.utils.parquet import get_dtypes, get_table_schema

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.parquet")
    nrows = 10
    frame = pd.DataFrame(
        {
            "col_float": np.arange(nrows, dtype=float) + 0.5,
            "col_int": np.arange(nrows, dtype=np.int64),
            "col_str": [f"a{i}" for i in range(nrows)],
        }
    )
    frame.to_parquet(path)
    return path


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


################################
#     Tests for get_dtypes     #
################################


def test_get_dtypes(frame_path: Path) -> None:
    assert get_dtypes(frame_path) == {
        "col_float": pa.float64(),
        "col_int": pa.int64(),
        "col_str": pa.string(),
    }
