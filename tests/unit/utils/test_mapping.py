from __future__ import annotations

from flamme.utils.mapping import sort_by_keys

##################################
#     Tests for sort_by_keys     #
##################################


def test_sort_dict_keys_empty() -> None:
    assert sort_by_keys({}) == {}


def test_sort_dict_keys() -> None:
    assert sort_by_keys({"dog": 1, "cat": 5, "fish": 2}) == {"cat": 5, "dog": 1, "fish": 2}
