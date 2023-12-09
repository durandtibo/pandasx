from __future__ import annotations

import logging
from pathlib import Path

from pandas import DataFrame
from pytest import LogCaptureFixture, TempPathFactory, fixture

from flamme.analyzer import NullValueAnalyzer
from flamme.ingestor import ParquetIngestor
from flamme.reporter import NoRepeatReporter, Reporter
from flamme.transformer.df import Sequential
from flamme.utils.io import load_text, save_text


@fixture(scope="module")
def df_path(tmp_path_factory: TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data").joinpath("df.parquet")
    df = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    df.to_parquet(path)
    return path


######################################
#     Tests for NoRepeatReporter     #
######################################


def test_no_repeat_reporter_str(df_path: Path, tmp_path: Path) -> None:
    report_path = tmp_path.joinpath("report.html")
    assert str(
        NoRepeatReporter(
            Reporter(
                ingestor=ParquetIngestor(df_path),
                transformer=Sequential(transformers=[]),
                analyzer=NullValueAnalyzer(),
                report_path=report_path,
            ),
            report_path=report_path,
        )
    ).startswith("NoRepeatReporter(")


def test_no_repeat_reporter_compute(df_path: Path, tmp_path: Path) -> None:
    report_path = tmp_path.joinpath("report.html")
    NoRepeatReporter(
        Reporter(
            ingestor=ParquetIngestor(df_path),
            transformer=Sequential(transformers=[]),
            analyzer=NullValueAnalyzer(),
            report_path=report_path,
        ),
        report_path=report_path,
    ).compute()
    assert report_path.is_file()


def test_no_repeat_reporter_compute_already_exist(
    df_path: Path, tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    report_path = tmp_path.joinpath("report.html")
    save_text("abc", report_path)
    reporter = NoRepeatReporter(
        Reporter(
            ingestor=ParquetIngestor(df_path),
            transformer=Sequential(transformers=[]),
            analyzer=NullValueAnalyzer(),
            report_path=report_path,
        ),
        report_path=report_path,
    )
    with caplog.at_level(level=logging.WARNING):
        reporter.compute()
    assert caplog.messages
    assert load_text(report_path) == "abc"
