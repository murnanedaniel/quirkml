"""Shared contract for the ML Tracking Review monorepo.

See `docs/predictions_schema.md` and the package README for what each
module is for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

# .io imports `colliderml` + `polars`; .schema (and .physics) are pure
# stdlib. Lazy-load so `from tracking_core.schema import write_predictions`
# -- what a baseline's adapter needs to emit the predictions contract --
# doesn't drag colliderml into environments (e.g. a baseline's Docker
# image) that have no reason to have it installed.
_ATTR_SOURCES = {
    "FEATURE_COLUMNS": "io",
    "ObjectLevel": "io",
    "Split": "io",
    "load_split": "io",
    "PREDICTIONS_COLUMNS": "schema",
    "RunMeta": "schema",
    "read_predictions": "schema",
    "write_predictions": "schema",
}


def __getattr__(name: str) -> Any:
    if name in ("physics", "schema"):
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _ATTR_SOURCES:
        import importlib

        module = importlib.import_module(f".{_ATTR_SOURCES[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from . import physics, schema
    from .io import FEATURE_COLUMNS, ObjectLevel, Split, load_split
    from .schema import (
        PREDICTIONS_COLUMNS,
        RunMeta,
        read_predictions,
        write_predictions,
    )
