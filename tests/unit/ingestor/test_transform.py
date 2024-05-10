from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from flamme.ingestor import ParquetIngestor, TransformedIngestor
from flamme.transformer.dataframe import ToNumeric

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def frame_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("frame.parquet")
    frame = pd.DataFrame(
        {
            "col1": ["1", "2", "3", "4", "5"],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    frame.to_parquet(path)
    return path


#########################################
#     Tests for TransformedIngestor     #
#########################################


def test_transformed_ingestor_str(frame_path: Path) -> None:
    assert str(
        TransformedIngestor(
            ingestor=ParquetIngestor(path=frame_path),
            transformer=ToNumeric(columns=["col1", "col3"]),
        )
    ).startswith("TransformedIngestor(")


def test_transformed_ingestor_ingest(frame_path: Path) -> None:
    ingestor = TransformedIngestor(
        ingestor=ParquetIngestor(path=frame_path),
        transformer=ToNumeric(columns=["col1", "col3"]),
    )
    assert_frame_equal(
        ingestor.ingest(),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
