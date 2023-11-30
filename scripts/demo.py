from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from flamme.analyzer import (
    BaseAnalyzer,
    ColumnSubsetAnalyzer,
    ColumnTypeAnalyzer,
    ContinuousDistributionAnalyzer,
    DiscreteDistributionAnalyzer,
    MappingAnalyzer,
    MarkdownAnalyzer,
    NullValueAnalyzer,
    TemporalContinuousDistributionAnalyzer,
    TemporalDiscreteDistributionAnalyzer,
    TemporalNullValueAnalyzer,
)
from flamme.ingestor import Ingestor
from flamme.preprocessor import (
    BasePreprocessor,
    SequentialPreprocessor,
    StripStrPreprocessor,
    ToDatetimePreprocessor,
    ToNumericPreprocessor,
)
from flamme.reporter import BaseReporter, Reporter

logger = logging.getLogger(__name__)


def create_dataframe(nrows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "bool": np.random.randint(0, 2, (nrows,), dtype=bool),
            "float": np.random.randn(nrows) * 3 + 1,
            "int": np.random.randint(0, 10, (nrows,)),
            "str": np.random.choice(["A", "B", "C"], size=(nrows,), p=[0.6, 0.3, 0.1]),
            "cauchy": np.abs(rng.standard_cauchy(size=(nrows,))),
        }
    )
    mask = rng.choice([True, False], size=df.shape, p=[0.2, 0.8])
    mask[:, 0] = rng.choice([True, False], size=(mask.shape[0]), p=[0.4, 0.6])
    mask[:, 1] = rng.choice([True, False], size=(mask.shape[0]), p=[0.8, 0.2])
    mask[:, 2] = rng.choice([True, False], size=(mask.shape[0]), p=[0.6, 0.4])
    mask[:, 3] = rng.choice([True, False], size=(mask.shape[0]), p=[0.2, 0.8])
    mask[mask.all(1), -1] = 0
    df = df.mask(mask)
    df["discrete"] = np.random.randint(0, 1001, (nrows,))
    df["datetime"] = pd.date_range("2018-01-01", periods=nrows, freq="H")
    return df


def create_dataframe2(nrows: int = 1000) -> pd.DataFrame:
    ncols = 100
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.normal(size=(nrows, ncols)), columns=[f"feature{i}" for i in range(ncols)]
    )
    mask = rng.choice([True, False], size=df.shape, p=[0.2, 0.8])
    mask[mask.all(1), -1] = 0
    df = df.mask(mask)
    df["datetime"] = pd.date_range("2018-01-01", periods=nrows, freq="H")
    return df


def create_analyzer() -> BaseAnalyzer:
    def create_discrete_column(column: str) -> BaseAnalyzer:
        return MappingAnalyzer(
            {
                "overall": DiscreteDistributionAnalyzer(column=column),
                "monthly": TemporalDiscreteDistributionAnalyzer(
                    column=column, dt_column="datetime", period="M"
                ),
                "weekly": TemporalDiscreteDistributionAnalyzer(
                    column=column, dt_column="datetime", period="W"
                ),
                "daily": TemporalDiscreteDistributionAnalyzer(
                    column=column, dt_column="datetime", period="D"
                ),
            }
        )

    def create_continuous_column(column: str, log_y: bool = False) -> BaseAnalyzer:
        return MappingAnalyzer(
            {
                "overall": ContinuousDistributionAnalyzer(column=column, log_y=log_y),
                "monthly": TemporalContinuousDistributionAnalyzer(
                    column=column, dt_column="datetime", period="M", log_y=log_y
                ),
                "weekly": TemporalContinuousDistributionAnalyzer(
                    column=column, dt_column="datetime", period="W", log_y=log_y
                ),
                "daily": TemporalContinuousDistributionAnalyzer(
                    column=column, dt_column="datetime", period="D", log_y=log_y
                ),
            }
        )

    return MappingAnalyzer(
        {
            "column type": ColumnTypeAnalyzer(),
            "null values": MappingAnalyzer(
                {
                    "overall": NullValueAnalyzer(),
                    "monthly": TemporalNullValueAnalyzer(dt_column="datetime", period="M"),
                    "weekly": TemporalNullValueAnalyzer(
                        dt_column="datetime", period="W", figsize=(700, 500)
                    ),
                    "daily": TemporalNullValueAnalyzer(
                        dt_column="datetime", period="D", ncols=1, figsize=(1400, 600)
                    ),
                }
            ),
            "columns": MappingAnalyzer(
                {
                    "str": DiscreteDistributionAnalyzer(column="str"),
                    "int": create_discrete_column(column="int"),
                    "discrete": create_discrete_column(column="discrete"),
                    "missing": DiscreteDistributionAnalyzer(column="missing"),
                    "float": create_continuous_column(column="float"),
                    "cauchy": MappingAnalyzer(
                        {
                            "description": MarkdownAnalyzer(
                                desc="""
- **Link:** URL
- **Description:** blabla
- **Valid values:** float values
"""
                            ),
                            "overall": ContinuousDistributionAnalyzer(
                                column="cauchy", log_y=True, xmax="q0.99"
                            ),
                            "monthly": TemporalContinuousDistributionAnalyzer(
                                column="cauchy", dt_column="datetime", period="M", log_y=True
                            ),
                            "weekly": TemporalContinuousDistributionAnalyzer(
                                column="cauchy", dt_column="datetime", period="W", log_y=True
                            ),
                            "daily": TemporalContinuousDistributionAnalyzer(
                                column="cauchy", dt_column="datetime", period="D", log_y=True
                            ),
                        }
                    ),
                }
            ),
            "subset": ColumnSubsetAnalyzer(
                columns=["discrete", "str"], analyzer=NullValueAnalyzer()
            ),
        }
    )


def create_preprocessor() -> BasePreprocessor:
    return SequentialPreprocessor(
        [
            StripStrPreprocessor(columns=["str"]),
            ToNumericPreprocessor(columns=["float", "int", "cauchy"]),
            ToDatetimePreprocessor(columns=["datetime"]),
        ]
    )


def create_preprocessor2() -> BasePreprocessor:
    return SequentialPreprocessor([])


def create_reporter() -> BaseReporter:
    return Reporter(
        ingestor=Ingestor(df=create_dataframe(nrows=50000)),
        preprocessor=create_preprocessor(),
        analyzer=create_analyzer(),
        report_path=Path.cwd().joinpath("tmp/report.html"),
    )


def main_report() -> None:
    reporter = create_reporter()
    logger.info(f"reporter:\n{reporter}")
    reporter.compute()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_report()
