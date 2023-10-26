from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from flamme.utils.path import human_file_size, sanitize_path

#####################################
#     Tests for human_file_size     #
#####################################


def test_human_file_size(tmp_path: Path) -> None:
    assert isinstance(human_file_size(Path(__file__).resolve()), str)


def test_human_file_size_2kb() -> None:
    path = Mock(spec=Path, stat=Mock(return_value=Mock(st_size=2048)))
    sanitize_mock = Mock(return_value=path)
    with patch("flamme.utils.path.sanitize_path", sanitize_mock):
        assert human_file_size(path) == "2.00 KB"
        sanitize_mock.assert_called_once_with(path)


###################################
#     Tests for sanitize_path     #
###################################


def test_sanitize_path_empty_str() -> None:
    assert sanitize_path("") == Path.cwd()


def test_sanitize_path_str() -> None:
    assert sanitize_path("something") == Path.cwd().joinpath("something")


def test_sanitize_path_path(tmp_path: Path) -> None:
    assert sanitize_path(tmp_path) == tmp_path


def test_sanitize_path_resolve() -> None:
    assert sanitize_path(Path("something/./../")) == Path.cwd()


def test_sanitize_path_uri() -> None:
    assert sanitize_path("file:///my/path/something/./../") == Path("/my/path")
