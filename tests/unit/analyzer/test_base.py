from __future__ import annotations

import logging
from collections import Counter

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture

from flamme.analyzer import NullValueAnalyzer, is_analyzer_config, setup_analyzer

########################################
#     Tests for is_analyzer_config     #
########################################


def test_is_analyzer_config_true() -> None:
    assert is_analyzer_config({OBJECT_TARGET: "flamme.analyzer.NullValueAnalyzer"})


def test_is_analyzer_config_false() -> None:
    assert not is_analyzer_config({OBJECT_TARGET: "torch.nn.Identity"})


####################################
#     Tests for setup_analyzer     #
####################################


def test_setup_analyzer_object() -> None:
    generator = NullValueAnalyzer()
    assert setup_analyzer(generator) is generator


def test_setup_analyzer_dict() -> None:
    assert isinstance(
        setup_analyzer({OBJECT_TARGET: "flamme.analyzer.NullValueAnalyzer"}),
        NullValueAnalyzer,
    )


def test_setup_analyzer_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_analyzer({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
