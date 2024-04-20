from __future__ import annotations

from unittest.mock import Mock

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from flamme.ingestor import ClickHouseIngestor
from flamme.testing import clickhouse_connect_available
from flamme.utils.imports import is_clickhouse_connect_available

if is_clickhouse_connect_available():
    from clickhouse_connect.driver import Client

########################################
#     Tests for ClickHouseIngestor     #
########################################


@clickhouse_connect_available
def test_clickhouse_ingestor_str() -> None:
    assert str(
        ClickHouseIngestor(query="select * from source.dataset", client=Mock(spec=Client))
    ).startswith("ClickHouseIngestor(")


@clickhouse_connect_available
def test_clickhouse_ingestor_ingest() -> None:
    frame = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    client_mock = Mock(spec=Client, query_df=Mock(return_value=frame))
    ingestor = ClickHouseIngestor(query="select * from source.dataset", client=client_mock)
    out = ingestor.ingest()
    assert_frame_equal(out, frame)
    client_mock.query_df.assert_called_once_with(query="select * from source.dataset")
