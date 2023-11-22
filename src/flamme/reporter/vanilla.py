from __future__ import annotations

__all__ = ["Reporter"]

import logging
from pathlib import Path

from coola.utils import str_indent, str_mapping

from flamme.analyzer.base import BaseAnalyzer, setup_analyzer
from flamme.ingestor.base import BaseIngestor, setup_ingestor
from flamme.preprocessor.base import BasePreprocessor, setup_preprocessor
from flamme.reporter.base import BaseReporter
from flamme.reporter.utils import create_html_report
from flamme.utils.io import save_text
from flamme.utils.path import sanitize_path

logger = logging.getLogger(__name__)


class Reporter(BaseReporter):
    r"""Implements a simple reporter.

    Args:
    ----
        ingestor (``BaseIngestor`` or dict): Specifies the ingestor
            or its configuration.
        preprocessor (``BasePreprocessor`` or dict): Specifies the
            DataFrame preprocessor or its configuration.
        ingestor (``BaseAnalyzer`` or dict): Specifies the analyzer
            or its configuration.
        report_path (``Path`` or str): Specifies the path where to
            save the HTML report.

    Example usage:

    .. code-block:: pycon

        >>> from flamme.analyzer import NullValueAnalyzer
        >>> from flamme.ingestor import ParquetIngestor
        >>> from flamme.preprocessor import SequentialPreprocessor
        >>> from flamme.reporter import Reporter
        >>> reporter = Reporter(
        ...     ingestor=ParquetIngestor("/path/to/data.parquet"),
        ...     preprocessor=SequentialPreprocessor(preprocessors=[]),
        ...     analyzer=NullValueAnalyzer(),
        ...     report_path="/path/to/report.html",
        ... )
        >>> report = reporter.compute()  # doctest: +SKIP
    """

    def __init__(
        self,
        ingestor: BaseIngestor | dict,
        preprocessor: BasePreprocessor | dict,
        analyzer: BaseAnalyzer | dict,
        report_path: Path | str,
    ) -> None:
        self._ingestor = setup_ingestor(ingestor)
        logger.info(f"ingestor:\n{ingestor}")
        self._preprocessor = setup_preprocessor(preprocessor)
        logger.info(f"preprocessor:\n{preprocessor}")
        self._analyzer = setup_analyzer(analyzer)
        logger.info(f"analyzer:\n{analyzer}")
        self._report_path = sanitize_path(report_path)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "ingestor": self._ingestor,
                    "preprocessor": self._preprocessor,
                    "analyzer": self._analyzer,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def compute(self) -> None:
        logger.info("Ingesting the DataFrame...")
        df = self._ingestor.ingest()
        logger.info(f"Preprocessing the DataFrame ({df.shape})...")
        df = self._preprocessor.preprocess(df)
        logger.info(f"Analyzing the DataFrame ({df.shape})...")
        section = self._analyzer.analyze(df)
        logger.info("Creating the HTML report...")
        report = create_html_report(
            toc=section.render_html_toc(max_depth=6),
            body=section.render_html_body(),
        )
        logger.info(f"Saving HTML report at {self._report_path}...")
        save_text(report, self._report_path)
