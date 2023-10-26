from __future__ import annotations

__all__ = ["BaseIngestor", "CsvIngestor", "ParquetIngestor"]

from flamme.ingestor.base import BaseIngestor
from flamme.ingestor.csv import CsvIngestor
from flamme.ingestor.parquet import ParquetIngestor
