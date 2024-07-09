from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from grizz.ingestor import ParquetIngestor
from objectory import OBJECT_TARGET

from flamme.analyzer import NullValueAnalyzer
from flamme.reporter import Reporter, is_reporter_config, setup_reporter
from flamme.transformer.dataframe import Sequential

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

########################################
#     Tests for is_reporter_config     #
########################################


def test_is_reporter_config_true() -> None:
    assert is_reporter_config(
        {
            OBJECT_TARGET: "flamme.reporter.Reporter",
            "ingestor": {OBJECT_TARGET: "grizz.ingestor.CsvIngestor", "path": "/path/to/data.csv"},
            "transformer": {
                OBJECT_TARGET: "flamme.transformer.dataframe.ToNumeric",
                "columns": ["col1", "col3"],
            },
            "analyzer": {OBJECT_TARGET: "flamme.analyzer.NullValueAnalyzer"},
            "report_path": "/path/to/report.html",
        }
    )


def test_is_reporter_config_false() -> None:
    assert not is_reporter_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_reporter     #
####################################


def test_setup_reporter_object(tmp_path: Path) -> None:
    reporter = Reporter(
        ingestor=ParquetIngestor(tmp_path.joinpath("data.parquet")),
        transformer=Sequential(transformers=[]),
        analyzer=NullValueAnalyzer(),
        report_path=tmp_path.joinpath("report.html"),
    )
    assert setup_reporter(reporter) is reporter


def test_setup_reporter_dict() -> None:
    assert isinstance(
        setup_reporter(
            {
                OBJECT_TARGET: "flamme.reporter.Reporter",
                "ingestor": {
                    OBJECT_TARGET: "grizz.ingestor.CsvIngestor",
                    "path": "/path/to/data.csv",
                },
                "transformer": {
                    OBJECT_TARGET: "flamme.transformer.dataframe.ToNumeric",
                    "columns": ["col1", "col3"],
                },
                "analyzer": {OBJECT_TARGET: "flamme.analyzer.NullValueAnalyzer"},
                "report_path": "/path/to/report.html",
            }
        ),
        Reporter,
    )


def test_setup_reporter_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_reporter({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
