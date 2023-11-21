from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from coola.utils import str_indent

from flamme.analyzer import (
    BaseAnalyzer,
    ColumnTypeAnalyzer,
    ContinuousDistributionAnalyzer,
    DiscreteDistributionAnalyzer,
    MappingAnalyzer,
    NullValueAnalyzer,
    TemporalContinuousDistributionAnalyzer,
    TemporalDiscreteDistributionAnalyzer,
    TemporalNullValueAnalyzer,
)
from flamme.section import BaseSection
from flamme.utils.io import save_text


def create_dataframe(nrows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "bool": np.random.randint(0, 2, (nrows,), dtype=bool),
            "float": np.random.randn(nrows) * 3 + 1,
            "int": np.random.randint(0, 10, (nrows,)),
            "str": np.random.choice(["A", "B", "C"], size=(nrows,), p=[0.6, 0.3, 0.1]),
            # "discrete": np.random.randint(0, 1000, (nrows,)),
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
    df["discrete"] = np.random.randint(0, 1000, (nrows,))
    df["datetime"] = pd.date_range("2018-01-01", periods=nrows, freq="H")
    return df


def create_report(toc: str, body: str) -> str:
    return f"""
<!doctype html>
<html>
    <head>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-4bw+/aepP/YC94hEpVNVgiZdgIC5+VKNBQNGCHeKRQN+PtmoHDEXuppvnDJzQIu9" crossorigin="anonymous"
    </head>
    <body style="margin: 1.5rem;">
    <div id="toc_container">
    <h2>Table of content</h2>
    {str_indent(toc, num_spaces=4)}
    </div>

    \t{str_indent(body, num_spaces=8)}
    </body>
</html>
"""


def create_analyzer() -> BaseAnalyzer:
    return MappingAnalyzer(
        {
            "column type": ColumnTypeAnalyzer(),
            "null values": MappingAnalyzer(
                {
                    "overall": NullValueAnalyzer(),
                    "monthly": TemporalNullValueAnalyzer(dt_column="datetime", period="M"),
                    "weekly": TemporalNullValueAnalyzer(dt_column="datetime", period="W"),
                    "daily": TemporalNullValueAnalyzer(dt_column="datetime", period="D"),
                }
            ),
            "columns": MappingAnalyzer(
                {
                    "str": DiscreteDistributionAnalyzer(column="str"),
                    "int": MappingAnalyzer(
                        {
                            "overall": DiscreteDistributionAnalyzer(column="int"),
                            "monthly": TemporalDiscreteDistributionAnalyzer(
                                column="int", dt_column="datetime", period="M"
                            ),
                            "weekly": TemporalDiscreteDistributionAnalyzer(
                                column="int", dt_column="datetime", period="W"
                            ),
                            "daily": TemporalDiscreteDistributionAnalyzer(
                                column="int", dt_column="datetime", period="D"
                            ),
                        }
                    ),
                    "discrete": DiscreteDistributionAnalyzer(column="discrete"),
                    "missing": DiscreteDistributionAnalyzer(column="missing"),
                    "float": MappingAnalyzer(
                        {
                            "overall": ContinuousDistributionAnalyzer(column="float"),
                            "monthly": TemporalContinuousDistributionAnalyzer(
                                column="float", dt_column="datetime", period="M"
                            ),
                            "weekly": TemporalContinuousDistributionAnalyzer(
                                column="float", dt_column="datetime", period="W"
                            ),
                            "daily": TemporalContinuousDistributionAnalyzer(
                                column="float", dt_column="datetime", period="D"
                            ),
                        }
                    ),
                    "cauchy": MappingAnalyzer(
                        {
                            "overall": ContinuousDistributionAnalyzer(column="cauchy", log_y=True),
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
        }
    )


def main_report() -> None:
    df = create_dataframe(nrows=10000)
    analyzer = create_analyzer()
    print(analyzer)
    section: BaseSection = analyzer.analyze(df)
    report = create_report(
        toc=section.render_html_toc(max_depth=6), body=section.render_html_body()
    )

    path = Path.cwd().joinpath("tmp/report.html")
    print(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_text(report, path)


if __name__ == "__main__":
    main_report()
