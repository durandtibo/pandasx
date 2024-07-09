# noqa: INP001
r"""Contain a demo example to generate a report."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from grizz.ingestor import Ingestor

from flamme import analyzer as fa
from flamme.reporter import BaseReporter, Reporter
from flamme.transformer.dataframe import (
    BaseDataFrameTransformer,
    SequentialDataFrameTransformer,
    StripString,
    ToDatetime,
    ToNumeric,
)
from flamme.utils.logging import configure_logging

logger = logging.getLogger(__name__)

FIGSIZE = (14, 5)


def create_dataframe(nrows: int = 1000) -> pd.DataFrame:
    r"""Create a DataFrame.

    Args:
        nrows: Specifies the number of rows.

    Returns:
        The generated DayaFrame.
    """
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "bool": rng.integers(low=0, high=2, size=(nrows,), dtype=bool),
            "float": rng.normal(size=(nrows,)) * 3 + 1 + 0.001 * np.arange(nrows),
            "int": rng.integers(low=0, high=10, size=(nrows,)),
            "str": rng.choice(["A", "B", "C"], size=(nrows,), p=[0.6, 0.3, 0.1]),
            "cauchy": rng.standard_cauchy(size=(nrows,)) + 0.01 * np.arange(nrows),
            "half cauchy": np.abs(rng.standard_cauchy(size=(nrows,))),
        }
    )
    mask = rng.choice([True, False], size=frame.shape, p=[0.2, 0.8])
    mask[:, 0] = rng.choice([True, False], size=(mask.shape[0]), p=[0.4, 0.6])
    mask[:, 1] = rng.choice([True, False], size=(mask.shape[0]), p=[0.8, 0.2])
    mask[:, 2] = rng.choice([True, False], size=(mask.shape[0]), p=[0.6, 0.4])
    mask[:, 3] = rng.choice([True, False], size=(mask.shape[0]), p=[0.2, 0.8])
    mask[mask.all(1), -1] = 0
    frame = frame.mask(mask)
    frame["discrete"] = rng.integers(low=0, high=1001, size=(nrows,))
    frame["datetime"] = pd.date_range("2018-01-01", periods=nrows, freq="H")
    frame["datetime_str"] = pd.date_range("2018-01-01", periods=nrows, freq="H").astype(str)
    return frame


def create_null_value_analyzer() -> fa.BaseAnalyzer:
    r"""Instantiate an analyzer about null values.

    Returns:
        The instantiated analyzer.
    """
    return fa.MappingAnalyzer(
        {
            "overall": fa.NullValueAnalyzer(figsize=FIGSIZE),
            "temporal": fa.TemporalNullValueAnalyzer(
                dt_column="datetime", period="M", figsize=FIGSIZE
            ),
            "monthly": fa.ColumnTemporalNullValueAnalyzer(
                dt_column="datetime", period="M", figsize=(7, 4)
            ),
            "weekly": fa.ColumnTemporalNullValueAnalyzer(dt_column="datetime", period="W"),
            "daily": fa.ColumnTemporalNullValueAnalyzer(
                dt_column="datetime", period="D", ncols=1, figsize=FIGSIZE
            ),
        }
    )


def create_discrete_column_analyzer(column: str) -> fa.BaseAnalyzer:
    r"""Instantiate an analyzer about discrete values for a given column.

    Args:
        column: Specifies the column to analyze.

    Returns:
        The instantiated analyzer.
    """
    return fa.MappingAnalyzer(
        {
            "overall": fa.ColumnDiscreteAnalyzer(column=column, figsize=FIGSIZE),
            "monthly": fa.ColumnTemporalDiscreteAnalyzer(
                column=column, dt_column="datetime", period="M", figsize=FIGSIZE
            ),
            # "weekly": ColumnTemporalDiscreteAnalyzer(
            #     column=column, dt_column="datetime", period="W", figsize=FIGSIZE
            # ),
            # "daily": ColumnTemporalDiscreteAnalyzer(
            #     column=column, dt_column="datetime", period="D", figsize=FIGSIZE
            # ),
            "null monthly": fa.ColumnTemporalNullValueAnalyzer(
                columns=[column], dt_column="datetime", period="M", figsize=FIGSIZE
            ),
            "most frequent": fa.MostFrequentValuesAnalyzer(column=column, top=10),
        }
    )


def create_continuous_column_analyzer(
    column: str,
    yscale: str = "linear",
    xmin: float | str | None = None,
    xmax: float | str | None = None,
) -> fa.BaseAnalyzer:
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
    return fa.MappingAnalyzer(
        {
            "overall": fa.ColumnContinuousAnalyzer(
                column=column,
                yscale=yscale,
                nbins=100,
                figsize=FIGSIZE,
                xmin=xmin,
                xmax=xmax,
            ),
            "null monthly": fa.ColumnTemporalNullValueAnalyzer(
                columns=[column], dt_column="datetime", period="M", figsize=FIGSIZE
            ),
            "monthly": fa.ColumnTemporalContinuousAnalyzer(
                column=column,
                dt_column="datetime",
                period="M",
                yscale=yscale,
                figsize=FIGSIZE,
            ),
            "weekly": fa.ColumnTemporalContinuousAnalyzer(
                column=column,
                dt_column="datetime",
                period="W",
                yscale=yscale,
                figsize=FIGSIZE,
            ),
            "daily": fa.ColumnTemporalContinuousAnalyzer(
                column=column,
                dt_column="datetime",
                period="D",
                yscale=yscale,
                figsize=FIGSIZE,
            ),
            "advanced": fa.ColumnContinuousAdvancedAnalyzer(
                column=column, yscale=yscale, nbins=100, figsize=FIGSIZE
            ),
            "most frequent": fa.MostFrequentValuesAnalyzer(column=column, top=10),
            "temporal drift monthly": fa.ColumnContinuousTemporalDriftAnalyzer(
                column=column,
                dt_column="datetime",
                period="M",
                nbins=201,
                figsize=FIGSIZE,
                xmin="q0.01",
                xmax="q0.99",
            ),
            "temporal drift monthly (density)": fa.ColumnContinuousTemporalDriftAnalyzer(
                column=column,
                dt_column="datetime",
                period="M",
                nbins=201,
                figsize=FIGSIZE,
                xmin="q0.01",
                xmax="q0.99",
                density=True,
            ),
        },
        max_toc_depth=1,
    )


def create_analyzer() -> fa.BaseAnalyzer:
    r"""Instantiate an analyzer.

    Returns:
        The instantiated analyzer.
    """
    columns = fa.MappingAnalyzer(
        {
            "str": create_discrete_column_analyzer(column="str"),
            "int": create_discrete_column_analyzer(column="int"),
            "discrete": create_discrete_column_analyzer(column="discrete"),
            "missing": fa.ColumnDiscreteAnalyzer(column="missing"),
            "float": create_continuous_column_analyzer(column="float"),
            "cauchy": create_continuous_column_analyzer(
                column="cauchy", yscale="auto", xmin="g0.001", xmax="g0.999"
            ),
            "half cauchy": create_continuous_column_analyzer(
                column="half cauchy", yscale="log", xmax="q0.99"
            ),
        }
    )
    # columns = fa.MappingAnalyzer({})
    return fa.MappingAnalyzer(
        {
            "query": fa.ContentAnalyzer(content="blablabla..."),
            "summary": fa.DataFrameSummaryAnalyzer(),
            "monthly count": fa.TemporalRowCountAnalyzer(
                dt_column="datetime",
                period="M",
                figsize=FIGSIZE,
            ),
            "duplicate": fa.DuplicatedRowAnalyzer(),
            "column type": fa.DataTypeAnalyzer(),
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
            # ColumnSelection(
            #     columns=["str", "float", "int", "cauchy", "datetime", "datetime_str", "missing"],
            #     ignore_missing=True,
            # ),
            StripString(columns=["str"]),
            ToNumeric(columns=["float", "int", "cauchy"]),
            ToDatetime(columns=["datetime", "datetime_str"]),
        ]
    )


def create_reporter() -> BaseReporter:
    r"""Instantiate a reporter.

    Returns:
        The instantiated reporter.
    """
    return Reporter(
        ingestor=Ingestor(frame=create_dataframe(nrows=10000)),
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
