from __future__ import annotations

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from flamme.ingestor import Ingestor

##############################
#     Tests for Ingestor     #
##############################


def test_ingestor_repr() -> None:
    assert repr(
        Ingestor(
            frame=pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["a", "b", "c", "d", "e"],
                    "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
                }
            )
        )
    ).startswith("Ingestor(")


def test_ingestor_str() -> None:
    assert str(
        Ingestor(
            frame=pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["a", "b", "c", "d", "e"],
                    "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
                }
            )
        )
    ).startswith("Ingestor(")


def test_ingestor_ingest_polars() -> None:
    frame = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    out = Ingestor(frame=frame).ingest()
    assert frame is out
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )


def test_ingestor_ingest_pandas() -> None:
    frame = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    out = Ingestor(frame=frame).ingest()
    assert frame is not out
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
