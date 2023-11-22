from __future__ import annotations

__all__ = [
    "BasePreprocessor",
    "ToNumericPreprocessor",
    "is_preprocessor_config",
    "setup_preprocessor",
]

from flamme.preprocessor.base import (
    BasePreprocessor,
    is_preprocessor_config,
    setup_preprocessor,
)
from flamme.preprocessor.numeric import ToNumericPreprocessor
