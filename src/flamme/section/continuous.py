from __future__ import annotations

__all__ = ["ContinuousDistributionSection"]

from collections.abc import Sequence

from jinja2 import Template
from pandas import DataFrame, Series

from flamme.section.base import BaseSection
from flamme.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)


class ContinuousDistributionSection(BaseSection):
    r"""Implements a section that analyzes a continuous distribution of
    values.

    Args:
    ----
        df (``pandas.DataFrame``): Specifies the DataFrame to analyze.
        column (str): Specifies the column of the DataFrame to analyze.
    """

    def __init__(self, df: DataFrame, column: str = "N/A") -> None:
        self._df = df
        self._column = column

    def get_statistics(self) -> dict:
        series = self._df[self._column] if self._column in self._df else Series([])
        stats = {
            "count": int(series.shape[0]),
            "num_nulls": int(series.isnull().sum()),
            "nunique": series.nunique(dropna=False),
        }
        stats["num_non_nulls"] = stats["count"] - stats["num_nulls"]
        if stats["num_non_nulls"] > 0:
            stats |= (
                series.dropna()
                .agg(
                    {
                        "mean": "mean",
                        "median": "median",
                        "min": "min",
                        "max": "max",
                        "std": "std",
                        "q01": lambda x: x.quantile(0.01),
                        "q05": lambda x: x.quantile(0.05),
                        "q10": lambda x: x.quantile(0.1),
                        "q25": lambda x: x.quantile(0.25),
                        "q75": lambda x: x.quantile(0.75),
                        "q90": lambda x: x.quantile(0.9),
                        "q95": lambda x: x.quantile(0.95),
                        "q99": lambda x: x.quantile(0.99),
                    }
                )
                .to_dict()
            )
        else:
            stats |= {
                "mean": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "std": float("nan"),
                "q01": float("nan"),
                "q05": float("nan"),
                "q10": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
                "q90": float("nan"),
                "q95": float("nan"),
                "q99": float("nan"),
            }
        return stats

    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        return Template(self._create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "column": self._column,
            }
        )

    def render_html_toc(
        self, number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
    ) -> str:
        return render_html_toc(number=number, tags=tags, depth=depth, max_depth=max_depth)

    def _create_template(self) -> str:
        return """
<h{{depth}} id="{{id}}">{{section}} {{title}} </h{{depth}}>

{{go_to_top}}

<p style="margin-top: 1rem;">
This section analyzes the discrete distribution of values for column {{column}}.

<ul>
  <li> total values: {{total_values}} </li>
  <li> number of unique values: {{unique_values}} </li>
  <li> number of null values: {{null_values}} / {{total_values}} ({{null_values_pct}}%) </li>
</ul>

{{figure}}
{{table}}
<p style="margin-top: 1rem;">
"""
