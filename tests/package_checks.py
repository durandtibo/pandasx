from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from coola import objects_are_equal
from grizz.ingestor import Ingestor
from grizz.transformer import Sequential
from matplotlib import pyplot as plt

from flamme.analyzer import DataFrameSummaryAnalyzer, NullValueAnalyzer
from flamme.plot import boxplot_continuous
from flamme.reporter import Reporter
from flamme.section import DataFrameSummarySection, DataTypeSection
from flamme.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def create_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.2, 4.2, None, 2.2, 1, 2.2],
            "col2": [1, 1, 0, 1, 1, 1],
            "col3": ["A", "B", None, None, "C", "B"],
        },
        schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.String},
    )


def check_analyzer() -> None:
    logger.info("Checking flamme.analyzer package...")

    section = DataFrameSummaryAnalyzer().analyze(create_dataframe())
    assert isinstance(section, DataFrameSummarySection)
    assert objects_are_equal(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "dtypes": (pl.Float64(), pl.Int64(), pl.String()),
            "null_count": (1, 0, 2),
            "nunique": (5, 2, 4),
        },
    )


def check_plot() -> None:
    logger.info("Checking flamme.plot package...")

    fig, ax = plt.subplots()
    boxplot_continuous(ax=ax, array=np.arange(101))


def check_reporter() -> None:
    logger.info("Checking flamme.reporter package...")

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir).joinpath("report.html")
        Reporter(
            ingestor=Ingestor(create_dataframe()),
            transformer=Sequential(transformers=[]),
            analyzer=NullValueAnalyzer(),
            report_path=report_path,
        ).compute()
        assert report_path.is_file()


def check_section() -> None:
    logger.info("Checking flamme.section package...")

    section = DataTypeSection(
        dtypes={"float": pl.Float64(), "int": pl.Int64(), "str": pl.String()},
        types={"float": {float}, "int": {int}, "str": {str, type(None)}},
    )
    assert objects_are_equal(
        section.get_statistics(),
        {"float": {float}, "int": {int}, "str": {str, type(None)}},
    )


def main() -> None:
    check_analyzer()
    check_plot()
    check_reporter()
    check_section()


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
