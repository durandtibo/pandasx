from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from flamme.analyzer import (
    BaseAnalyzer,
    ColumnContinuousAnalyzer,
    ColumnDiscreteAnalyzer,
    ColumnTemporalContinuousAnalyzer,
    ColumnTemporalDiscreteAnalyzer,
    ColumnTemporalNullValueAnalyzer,
    DataTypeAnalyzer,
    DuplicatedRowAnalyzer,
    GlobalTemporalNullValueAnalyzer,
    MappingAnalyzer,
    MarkdownAnalyzer,
    NullValueAnalyzer,
    TemporalNullValueAnalyzer, ColumnContinuousAdvancedAnalyzer,
)
from flamme.ingestor import Ingestor
from flamme.reporter import BaseReporter, Reporter
from flamme.transformer.df import (
    BaseDataFrameTransformer,
    SequentialDataFrameTransformer,
    StripString,
    ToDatetime,
    ToNumeric,
)

logger = logging.getLogger(__name__)

FIGSIZE = (14, 6)


def create_dataframe(nrows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "bool": np.random.randint(0, 2, (nrows,), dtype=bool),
            "float": np.random.randn(nrows) * 3 + 1,
            "int": np.random.randint(0, 10, (nrows,)),
            "str": np.random.choice(["A", "B", "C"], size=(nrows,), p=[0.6, 0.3, 0.1]),
            "cauchy": rng.standard_cauchy(size=(nrows,)),
            "half cauchy": np.abs(rng.standard_cauchy(size=(nrows,))),
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
    df["datetime_str"] = pd.date_range("2018-01-01", periods=nrows, freq="H").astype(str)
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
                "overall": ColumnDiscreteAnalyzer(column=column, figsize=FIGSIZE),
                "monthly": ColumnTemporalDiscreteAnalyzer(
                    column=column, dt_column="datetime", period="M", figsize=FIGSIZE
                ),
                # "weekly": ColumnTemporalDiscreteAnalyzer(
                #     column=column, dt_column="datetime", period="W", figsize=FIGSIZE
                # ),
                # "daily": ColumnTemporalDiscreteAnalyzer(
                #     column=column, dt_column="datetime", period="D", figsize=FIGSIZE
                # ),
                "null monthly": ColumnTemporalNullValueAnalyzer(
                    column=column, dt_column="datetime", period="M", figsize=FIGSIZE
                ),
            }
        )

    def create_continuous_column(
        column: str,
        yscale: str = "linear",
        xmin: float | str | None = None,
        xmax: float | str | None = None,
    ) -> BaseAnalyzer:
        return MappingAnalyzer(
            {
                "overall": ColumnContinuousAnalyzer(
                    column=column, yscale=yscale, nbins=100, figsize=FIGSIZE, xmin=xmin, xmax=xmax
                ),
                "monthly": ColumnTemporalContinuousAnalyzer(
                    column=column, dt_column="datetime", period="M", yscale=yscale, figsize=FIGSIZE
                ),
                "weekly": ColumnTemporalContinuousAnalyzer(
                    column=column, dt_column="datetime", period="W", yscale=yscale, figsize=FIGSIZE
                ),
                "daily": ColumnTemporalContinuousAnalyzer(
                    column=column, dt_column="datetime", period="D", yscale=yscale, figsize=FIGSIZE
                ),
                "advanced": ColumnContinuousAdvancedAnalyzer(
                    column=column, yscale=yscale, nbins=100, figsize=FIGSIZE
                ),
            }
        )

    return MappingAnalyzer(
        {
            "duplicate": DuplicatedRowAnalyzer(),
            "column type": DataTypeAnalyzer(),
            "null values": MappingAnalyzer(
                {
                    "overall": NullValueAnalyzer(figsize=FIGSIZE),
                    "temporal": GlobalTemporalNullValueAnalyzer(
                        dt_column="datetime", period="M", figsize=FIGSIZE
                    ),
                    "monthly": TemporalNullValueAnalyzer(dt_column="datetime", period="M"),
                    "weekly": TemporalNullValueAnalyzer(
                        dt_column="datetime", period="W", figsize=(7, 5)
                    ),
                    "daily": TemporalNullValueAnalyzer(
                        dt_column="datetime", period="D", ncols=1, figsize=FIGSIZE
                    ),
                }
            ),
            "columns": MappingAnalyzer(
                {
                    "str": create_discrete_column(column="str"),
                    "int": create_discrete_column(column="int"),
                    "discrete": create_discrete_column(column="discrete"),
                    "missing": ColumnDiscreteAnalyzer(column="missing"),
                    "float": create_continuous_column(column="float"),
                    "cauchy": create_continuous_column(
                        column="cauchy", yscale="symlog", xmin="g0.001", xmax="g0.999"
                    ),
                    "half cauchy": MappingAnalyzer(
                        {
                            "description": MarkdownAnalyzer(
                                desc="""
- **Link:** URL
- **Description:** blabla
- **Valid values:** float values
"""
                            ),
                            "overall": ColumnContinuousAnalyzer(
                                column="half cauchy", yscale="log", xmax="q0.99"
                            ),
                            "monthly": ColumnTemporalContinuousAnalyzer(
                                column="half cauchy", dt_column="datetime", period="M", yscale="log"
                            ),
                            "weekly": ColumnTemporalContinuousAnalyzer(
                                column="half cauchy", dt_column="datetime", period="W", yscale="log"
                            ),
                            "daily": ColumnTemporalContinuousAnalyzer(
                                column="half cauchy", dt_column="datetime", period="D", yscale="log"
                            ),
                        }
                    ),
                }
            ),
            # "subset": ColumnSubsetAnalyzer(
            #     columns=["discrete", "str"], analyzer=NullValueAnalyzer()
            # ),
        }
    )


def create_transformer() -> BaseDataFrameTransformer:
    return SequentialDataFrameTransformer(
        [
            StripString(columns=["str"]),
            ToNumeric(columns=["float", "int", "cauchy"]),
            ToDatetime(columns=["datetime", "datetime_str"]),
        ]
    )


def create_transformer2() -> BaseDataFrameTransformer:
    return SequentialDataFrameTransformer([])


def create_reporter() -> BaseReporter:
    return Reporter(
        ingestor=Ingestor(df=create_dataframe(nrows=10000)),
        transformer=create_transformer(),
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
