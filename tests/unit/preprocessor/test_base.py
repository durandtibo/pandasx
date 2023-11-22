from __future__ import annotations

import logging
from collections import Counter

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture

from flamme.preprocessor import (
    ToNumericPreprocessor,
    is_preprocessor_config,
    setup_preprocessor,
)

############################################
#     Tests for is_preprocessor_config     #
############################################


def test_is_preprocessor_config_true() -> None:
    assert is_preprocessor_config(
        {
            OBJECT_TARGET: "flamme.preprocessor.ToNumericPreprocessor",
            "columns": ["col1", "col3"],
        }
    )


def test_is_preprocessor_config_false() -> None:
    assert not is_preprocessor_config({OBJECT_TARGET: "collections.Counter"})


########################################
#     Tests for setup_preprocessor     #
########################################


def test_setup_preprocessor_object() -> None:
    preprocessor = ToNumericPreprocessor(columns=["col1", "col3"])
    assert setup_preprocessor(preprocessor) is preprocessor


def test_setup_preprocessor_dict() -> None:
    assert isinstance(
        setup_preprocessor(
            {
                OBJECT_TARGET: "flamme.preprocessor.ToNumericPreprocessor",
                "columns": ["col1", "col3"],
            }
        ),
        ToNumericPreprocessor,
    )


def test_setup_preprocessor_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_preprocessor({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
