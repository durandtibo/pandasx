from __future__ import annotations

from unittest.mock import patch

import pytest

from flamme.testing import markdown_available
from flamme.utils.text import markdown_to_html

######################################
#     Tests for markdown_to_html     #
######################################


@markdown_available
def test_markdown_to_html() -> None:
    assert markdown_to_html("- a\n- b\n- c") == "<ul>\n<li>a</li>\n<li>b</li>\n<li>c</li>\n</ul>"


def test_markdown_to_html_no_markdown() -> None:
    with (
        patch("flamme.utils.text.is_markdown_available", lambda: False),
        patch("flamme.utils.imports.is_markdown_available", lambda: False),
        pytest.raises(RuntimeError, match="`markdown` package is required but not installed."),
    ):
        markdown_to_html("- a\n- b\n- c")


def test_markdown_to_html_no_markdown_ignore() -> None:
    with patch("flamme.utils.text.is_markdown_available", lambda: False):
        assert markdown_to_html("- a\n- b\n- c", ignore_error=True) == "- a\n- b\n- c"
