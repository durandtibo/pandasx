from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from flamme.schema.reader import (
    ParquetSchemaReader,
    is_schema_reader_config,
    setup_schema_reader,
)

if TYPE_CHECKING:
    import pytest

#############################################
#     Tests for is_schema_reader_config     #
#############################################


def test_is_schema_reader_config_true() -> None:
    assert is_schema_reader_config(
        {OBJECT_TARGET: "flamme.schema.reader.ParquetSchemaReader", "path": "/path/to/data.parquet"}
    )


def test_is_schema_reader_config_false() -> None:
    assert not is_schema_reader_config({OBJECT_TARGET: "collections.Counter"})


#########################################
#     Tests for setup_schema_reader     #
#########################################


def test_setup_schema_reader_object() -> None:
    reader = ParquetSchemaReader(path="/path/to/data.parquet")
    assert setup_schema_reader(reader) is reader


def test_setup_schema_reader_dict() -> None:
    assert isinstance(
        setup_schema_reader(
            {
                OBJECT_TARGET: "flamme.schema.reader.ParquetSchemaReader",
                "path": "/path/to/data.parquet",
            }
        ),
        ParquetSchemaReader,
    )


def test_setup_schema_reader_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_schema_reader({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
