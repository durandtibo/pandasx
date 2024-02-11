from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pandas import DataFrame

from flamme.analyzer import NullValueAnalyzer
from flamme.ingestor import ParquetIngestor
from flamme.reporter import Reporter
from flamme.transformer.df import Sequential

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def df_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("df.parquet")
    dataframe = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    dataframe.to_parquet(path)
    return path


##############################
#     Tests for Reporter     #
##############################


def test_reporter_str(df_path: Path, tmp_path: Path) -> None:
    report_path = tmp_path.joinpath("report.html")
    assert str(
        Reporter(
            ingestor=ParquetIngestor(df_path),
            transformer=Sequential(transformers=[]),
            analyzer=NullValueAnalyzer(),
            report_path=report_path,
        )
    ).startswith("Reporter(")


def test_reporter_compute(df_path: Path, tmp_path: Path) -> None:
    report_path = tmp_path.joinpath("report.html")
    Reporter(
        ingestor=ParquetIngestor(df_path),
        transformer=Sequential(transformers=[]),
        analyzer=NullValueAnalyzer(),
        report_path=report_path,
    ).compute()
    assert report_path.is_file()
