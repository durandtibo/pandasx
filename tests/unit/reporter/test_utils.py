from __future__ import annotations

from jinja2 import Template

from flamme.reporter.utils import create_html_report

########################################
#     Tests for create_html_report     #
########################################


def test_create_html_report() -> None:
    assert isinstance(Template(create_html_report(toc="", body="")).render(), str)
