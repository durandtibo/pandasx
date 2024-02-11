from __future__ import annotations

from unittest.mock import Mock, patch

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from flamme.ingestor import ClickHouseIngestor
from flamme.testing import clickhouse_connect_available

########################################
#     Tests for ClickHouseIngestor     #
########################################


@clickhouse_connect_available
def test_clickhouse_ingestor_str() -> None:
    assert str(
        ClickHouseIngestor(query="select * from source.dataset", client_config={})
    ).startswith("ClickHouseIngestor(")


@clickhouse_connect_available
def test_clickhouse_ingestor_ingest() -> None:
    ingestor = ClickHouseIngestor(
        query="select * from source.dataset",
        client_config={"username": "POLAR"},
    )
    dataframe = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    query_df_mock = Mock(return_value=dataframe)
    clickhouse_mock = Mock(return_value=Mock(query_df=query_df_mock))
    with patch("flamme.ingestor.clickhouse.clickhouse_connect.get_client", clickhouse_mock):
        dataframe = ingestor.ingest()
        assert_frame_equal(
            dataframe,
            DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["a", "b", "c", "d", "e"],
                    "col3": [1.2, 2.2, 3.2, 4.2, 5.2],
                }
            ),
        )
        clickhouse_mock.assert_called_once_with(username="POLAR")
        query_df_mock.assert_called_once_with(query="select * from source.dataset")
