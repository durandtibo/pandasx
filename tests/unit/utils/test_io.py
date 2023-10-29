from __future__ import annotations

from pathlib import Path

from flamme.utils.io import save_text


def test_save_text(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data", "data.txt")
    save_text("Hello!", file_path)
    assert file_path.is_file()
