from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from flamme.ingestor import ParquetIngestor, TransformedIngestor
from flamme.transformer.dataframe import ToNumeric

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def df_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("df.parquet")
    df = DataFrame(
        {
            "col1": ["1", "2", "3", "4", "5"],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    df.to_parquet(path)
    return path


#########################################
#     Tests for TransformedIngestor     #
#########################################


def test_transformed_ingestor_str(df_path: Path) -> None:
    assert str(
        TransformedIngestor(
            ingestor=ParquetIngestor(path=df_path),
            transformer=ToNumeric(columns=["col1", "col3"]),
        )
    ).startswith("TransformedIngestor(")


def test_transformed_ingestor_ingest(df_path: Path) -> None:
    ingestor = TransformedIngestor(
        ingestor=ParquetIngestor(path=df_path),
        transformer=ToNumeric(columns=["col1", "col3"]),
    )
    assert_frame_equal(
        ingestor.ingest(),
        DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
