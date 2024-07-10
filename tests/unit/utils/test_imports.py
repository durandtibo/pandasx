from __future__ import annotations

from unittest.mock import patch

import pytest

from flamme.utils.imports import (
    check_clickhouse_connect,
    check_colorlog,
    clickhouse_connect_available,
    colorlog_available,
    is_clickhouse_connect_available,
    is_colorlog_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


##############################
#     clickhouse_connect     #
##############################


def test_check_clickhouse_connect_with_package() -> None:
    with patch("flamme.utils.imports.is_clickhouse_connect_available", lambda: True):
        check_clickhouse_connect()


def test_check_clickhouse_connect_without_package() -> None:
    with (
        patch("flamme.utils.imports.is_clickhouse_connect_available", lambda: False),
        pytest.raises(
            RuntimeError, match="`clickhouse_connect` package is required but not installed."
        ),
    ):
        check_clickhouse_connect()


def test_is_clickhouse_connect_available() -> None:
    assert isinstance(is_clickhouse_connect_available(), bool)


def test_clickhouse_connect_available_with_package() -> None:
    with patch("flamme.utils.imports.is_clickhouse_connect_available", lambda: True):
        fn = clickhouse_connect_available(my_function)
        assert fn(2) == 44


def test_clickhouse_connect_available_without_package() -> None:
    with patch("flamme.utils.imports.is_clickhouse_connect_available", lambda: False):
        fn = clickhouse_connect_available(my_function)
        assert fn(2) is None


def test_clickhouse_connect_available_decorator_with_package() -> None:
    with patch("flamme.utils.imports.is_clickhouse_connect_available", lambda: True):

        @clickhouse_connect_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_clickhouse_connect_available_decorator_without_package() -> None:
    with patch("flamme.utils.imports.is_clickhouse_connect_available", lambda: False):

        @clickhouse_connect_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


####################
#     colorlog     #
####################


def test_check_colorlog_with_package() -> None:
    with patch("flamme.utils.imports.is_colorlog_available", lambda: True):
        check_colorlog()


def test_check_colorlog_without_package() -> None:
    with (
        patch("flamme.utils.imports.is_colorlog_available", lambda: False),
        pytest.raises(RuntimeError, match="`colorlog` package is required but not installed."),
    ):
        check_colorlog()


def test_is_colorlog_available() -> None:
    assert isinstance(is_colorlog_available(), bool)


def test_colorlog_available_with_package() -> None:
    with patch("flamme.utils.imports.is_colorlog_available", lambda: True):
        fn = colorlog_available(my_function)
        assert fn(2) == 44


def test_colorlog_available_without_package() -> None:
    with patch("flamme.utils.imports.is_colorlog_available", lambda: False):
        fn = colorlog_available(my_function)
        assert fn(2) is None


def test_colorlog_available_decorator_with_package() -> None:
    with patch("flamme.utils.imports.is_colorlog_available", lambda: True):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_colorlog_available_decorator_without_package() -> None:
    with patch("flamme.utils.imports.is_colorlog_available", lambda: False):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
