from __future__ import annotations

from typing import TYPE_CHECKING

from flamme.utils.io import load_text, save_text

if TYPE_CHECKING:
    from pathlib import Path


def test_save_load_text(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data", "data.txt")
    save_text("Hello!", file_path)
    assert load_text(file_path) == "Hello!"
