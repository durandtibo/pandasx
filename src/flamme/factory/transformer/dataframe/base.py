r"""Contain the base class to implement a DataFrame transformer
factory."""

from __future__ import annotations

__all__ = [
    "BaseDataFrameTransformerFactory",
    "is_dataframe_transformer_factory_config",
    "setup_dataframe_transformer_factory",
]

import logging
from abc import ABC
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    from flamme.transformer.df import BaseDataFrameTransformer

logger = logging.getLogger(__name__)


class BaseDataFrameTransformerFactory(ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a DataFrame ingestor.

    Example usage:

    ```pycon

    >>> from flamme.factory.transformer.dataframe import BaseDataFrameTransformerFactory

    ```
    """

    def create(self) -> BaseDataFrameTransformer:
        r"""Ingest a DataFrame.

        Returns:
            The ingested DataFrame.

        Example usage:

        ```pycon

        >>> from flamme.factory.transformer.dataframe import BaseDataFrameTransformerFactory

        ```
        """


def is_dataframe_transformer_factory_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDataFrameTransformerFactory``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseDataFrameTransformerFactory`` object.

    Example usage:

    ```pycon
    >>> from flamme.ingestor import is_ingestor_config
    >>> is_ingestor_config(
    ...     {"_target_": "flamme.ingestor.CsvTransformerFactory", "path": "/path/to/data.csv"}
    ... )
    True

    ```
    """
    return is_object_config(config, BaseDataFrameTransformerFactory)


def setup_dataframe_transformer_factory(
    transformer_factory: BaseDataFrameTransformerFactory | dict,
) -> BaseDataFrameTransformerFactory:
    r"""Set up a DataFrame transformer factory.

    The ingestor is instantiated from its configuration
    by using the ``BaseDataFrameTransformerFactory`` factory function.

    Args:
        transformer_factory: The DataFrame transformer factory or its
            configuration.

    Returns:
        An instantiated DataFrame transformer factory.

    Example usage:

    ```pycon

    >>> from flamme.ingestor import setup_ingestor
    >>> ingestor = setup_ingestor(
    ...     {"_target_": "flamme.ingestor.CsvTransformerFactory", "path": "/path/to/data.csv"}
    ... )
    >>> ingestor
    CsvTransformerFactory(path=/path/to/data.csv)

    ```
    """
    if isinstance(transformer_factory, dict):
        logger.info("Initializing a DataFrame transformer factory from its configuration... ")
        transformer_factory = BaseDataFrameTransformerFactory.factory(**transformer_factory)
    if not isinstance(transformer_factory, BaseDataFrameTransformerFactory):
        logger.warning(
            "transformer_factory is not a BaseDataFrameTransformerFactory "
            f"(received: {type(transformer_factory)})"
        )
    return transformer_factory
