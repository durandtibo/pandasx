from __future__ import annotations

from pathlib import Path

from flamme.utils.io import load_text, save_text


def test_save_load_text(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data", "data.txt")
    save_text("Hello!", file_path)
    assert load_text(file_path) == "Hello!"
