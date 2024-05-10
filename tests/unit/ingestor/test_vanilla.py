from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from flamme.ingestor import Ingestor

##############################
#     Tests for Ingestor     #
##############################


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


def test_ingestor_ingest() -> None:
    assert_frame_equal(
        Ingestor(
            frame=pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["a", "b", "c", "d", "e"],
                    "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
                }
            )
        ).ingest(),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
            }
        ),
    )
