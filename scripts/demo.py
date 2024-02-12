# noqa: INP001
r"""Contain a demo example to generate a report."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from flamme.analyzer import (
    BaseAnalyzer,
    ColumnContinuousAdvancedAnalyzer,
    ColumnContinuousAnalyzer,
    ColumnDiscreteAnalyzer,
    ColumnTemporalContinuousAnalyzer,
    ColumnTemporalDiscreteAnalyzer,
    ColumnTemporalNullValueAnalyzer,
    DataFrameSummaryAnalyzer,
    DataTypeAnalyzer,
    DuplicatedRowAnalyzer,
    GlobalTemporalNullValueAnalyzer,
    MappingAnalyzer,
    MostFrequentValuesAnalyzer,
    NullValueAnalyzer,
    TemporalNullValueAnalyzer,
    TemporalRowCountAnalyzer,
)
from flamme.ingestor import Ingestor
from flamme.reporter import BaseReporter, Reporter
from flamme.transformer.df import (
    BaseDataFrameTransformer,
    ColumnSelection,
    SequentialDataFrameTransformer,
    StripString,
    ToDatetime,
    ToNumeric,
)
from flamme.utils.logging import configure_logging

logger = logging.getLogger(__name__)

FIGSIZE = (14, 6)


def create_dataframe(nrows: int = 1000) -> pd.DataFrame:
    r"""Create a DataFrame.

    Args:
        nrows: Specifies the number of rows.

    Returns:
        The generated DayaFrame.
    """
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
    r"""Create a DataFrame.

    Args:
        nrows: Specifies the number of rows.

    Returns:
        The generated DayaFrame.
    """
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


def create_null_value_analyzer() -> BaseAnalyzer:
    r"""Instantiate an analyzer about null values.

    Returns:
        The instantiated analyzer.
    """
    return MappingAnalyzer(
        {
            "overall": NullValueAnalyzer(figsize=FIGSIZE),
            "temporal": GlobalTemporalNullValueAnalyzer(
                dt_column="datetime", period="M", figsize=FIGSIZE
            ),
            "monthly": TemporalNullValueAnalyzer(dt_column="datetime", period="M"),
            "weekly": TemporalNullValueAnalyzer(dt_column="datetime", period="W", figsize=(7, 5)),
            "daily": TemporalNullValueAnalyzer(
                dt_column="datetime", period="D", ncols=1, figsize=FIGSIZE
            ),
        }
    )


def create_discrete_column_analyzer(column: str) -> BaseAnalyzer:
    r"""Instantiate an analyzer about discrete values for a given column.

    Args:
        column: Specifies the column to analyze.

    Returns:
        The instantiated analyzer.
    """
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
            "most frequent": MostFrequentValuesAnalyzer(column=column, top=10),
        }
    )


def create_continuous_column_analyzer(
    column: str,
    yscale: str = "linear",
    xmin: float | str | None = None,
    xmax: float | str | None = None,
) -> BaseAnalyzer:
    r"""Instantiate an analyzer about continuous values for a given
    column.

    Args:
        column: Specifies the column to analyze.
        yscale: Specifies the y-axis scale.
        xmin: Specifies the minimum value of the range or its
            associated quantile.
        xmax: Specifies the maximum value of the range or its
            associated quantile.

    Returns:
        The instantiated analyzer.
    """
    return MappingAnalyzer(
        {
            "overall": ColumnContinuousAnalyzer(
                column=column,
                yscale=yscale,
                nbins=100,
                figsize=FIGSIZE,
                xmin=xmin,
                xmax=xmax,
            ),
            "monthly": ColumnTemporalContinuousAnalyzer(
                column=column,
                dt_column="datetime",
                period="M",
                yscale=yscale,
                figsize=FIGSIZE,
            ),
            "weekly": ColumnTemporalContinuousAnalyzer(
                column=column,
                dt_column="datetime",
                period="W",
                yscale=yscale,
                figsize=FIGSIZE,
            ),
            "daily": ColumnTemporalContinuousAnalyzer(
                column=column,
                dt_column="datetime",
                period="D",
                yscale=yscale,
                figsize=FIGSIZE,
            ),
            "advanced": ColumnContinuousAdvancedAnalyzer(
                column=column, yscale=yscale, nbins=100, figsize=FIGSIZE
            ),
            "most frequent": MostFrequentValuesAnalyzer(column=column, top=10),
        },
        max_toc_depth=1,
    )


def create_analyzer() -> BaseAnalyzer:
    r"""Instantiate an analyzer.

    Returns:
        The instantiated analyzer.
    """
    columns = MappingAnalyzer(
        {
            "str": create_discrete_column_analyzer(column="str"),
            "int": create_discrete_column_analyzer(column="int"),
            "discrete": create_discrete_column_analyzer(column="discrete"),
            "missing": ColumnDiscreteAnalyzer(column="missing"),
            "float": create_continuous_column_analyzer(column="float"),
            "cauchy": create_continuous_column_analyzer(
                column="cauchy", yscale="auto", xmin="g0.001", xmax="g0.999"
            ),
            "half cauchy": create_continuous_column_analyzer(
                column="half cauchy", yscale="log", xmax="q0.99"
            ),
        }
    )
    # columns = MappingAnalyzer({})
    return MappingAnalyzer(
        {
            "summary": DataFrameSummaryAnalyzer(),
            "monthly count": TemporalRowCountAnalyzer(
                dt_column="datetime",
                period="M",
                figsize=FIGSIZE,
            ),
            "duplicate": DuplicatedRowAnalyzer(),
            "column type": DataTypeAnalyzer(),
            "null values": create_null_value_analyzer(),
            "columns": columns,
            # "subset": ColumnSubsetAnalyzer(
            #     columns=["discrete", "str"], analyzer=NullValueAnalyzer()
            # ),
        }
    )


def create_transformer() -> BaseDataFrameTransformer:
    r"""Instantiate a ``pandas.DataFrame`` transformer.

    Returns:
        The instantiated transformer.
    """
    return SequentialDataFrameTransformer(
        [
            ColumnSelection(
                columns=["str", "float", "int", "cauchy", "datetime", "datetime_str", "missing"],
                ignore_missing=True,
            ),
            StripString(columns=["str"]),
            ToNumeric(columns=["float", "int", "cauchy"]),
            ToDatetime(columns=["datetime", "datetime_str"]),
        ]
    )


def create_transformer2() -> BaseDataFrameTransformer:
    r"""Instantiate a ``pandas.DataFrame`` transformer.

    Returns:
        The instantiated transformer.
    """
    return SequentialDataFrameTransformer([])


def create_reporter() -> BaseReporter:
    r"""Instantiate a reporter.

    Returns:
        The instantiated reporter.
    """
    return Reporter(
        ingestor=Ingestor(df=create_dataframe(nrows=10000)),
        transformer=create_transformer(),
        analyzer=create_analyzer(),
        report_path=Path.cwd().joinpath("tmp/report.html"),
    )


def main_report() -> None:
    r"""Define the main function to generate a report."""
    reporter = create_reporter()
    logger.info(f"reporter:\n{reporter}")
    reporter.compute()


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main_report()
