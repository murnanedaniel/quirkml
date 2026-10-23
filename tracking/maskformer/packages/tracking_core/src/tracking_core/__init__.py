"""Shared contract for the ML Tracking Review monorepo.

See `docs/predictions_schema.md` and the package README for what each
module is for.
"""

from __future__ import annotations

from . import physics, schema
from .io import FEATURE_COLUMNS, ObjectLevel, Split, load_split
from .schema import (
    PREDICTIONS_COLUMNS,
    RunMeta,
    read_predictions,
    write_predictions,
)

__all__ = [
    "FEATURE_COLUMNS",
    "ObjectLevel",
    "PREDICTIONS_COLUMNS",
    "RunMeta",
    "Split",
    "load_split",
    "physics",
    "read_predictions",
    "schema",
    "write_predictions",
]
