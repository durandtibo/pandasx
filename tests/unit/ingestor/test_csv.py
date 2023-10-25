from __future__ import annotations

from pathlib import Path

from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import TempPathFactory, fixture

from pandasx.ingestor import CsvIngestor


@fixture(scope="module")
def df_path(tmp_path_factory: TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("df.csv")
    df = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    df.to_csv(path, index=False)
    return path


#################################
#     Tests for CsvIngestor     #
#################################


def test_csv_ingestor_str(df_path: Path) -> None:
    assert str(CsvIngestor(df_path)).startswith("CsvIngestor(")


def test_csv_ingestor_str_with_kwargs(df_path: Path) -> None:
    assert str(CsvIngestor(df_path, usecols=["col1", "col3"])).startswith("CsvIngestor(")


def test_csv_ingestor_ingest(df_path: Path) -> None:
    assert_frame_equal(
        CsvIngestor(df_path).ingest(),
        DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_csv_ingestor_ingest_with_kwargs(df_path: Path) -> None:
    print(CsvIngestor(df_path, names=["col1", "col3"]).ingest())
    assert_frame_equal(
        CsvIngestor(df_path, usecols=["col1", "col3"]).ingest(),
        DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
