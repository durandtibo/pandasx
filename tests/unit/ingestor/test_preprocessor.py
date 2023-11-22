from __future__ import annotations

from pathlib import Path

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import TempPathFactory, fixture

from flamme.ingestor import ParquetIngestor, PreprocessorIngestor
from flamme.preprocessor import ToNumericPreprocessor


@fixture(scope="module")
def df_path(tmp_path_factory: TempPathFactory) -> Path:
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


##########################################
#     Tests for PreprocessorIngestor     #
##########################################


def test_preprocessor_ingestor_str(df_path: Path) -> None:
    assert str(
        PreprocessorIngestor(
            ingestor=ParquetIngestor(path=df_path),
            preprocessor=ToNumericPreprocessor(columns=["col1", "col3"]),
        )
    ).startswith("PreprocessorIngestor(")


def test_preprocessor_ingestor_ingest(df_path: Path) -> None:
    ingestor = PreprocessorIngestor(
        ingestor=ParquetIngestor(path=df_path),
        preprocessor=ToNumericPreprocessor(columns=["col1", "col3"]),
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
