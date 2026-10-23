"""Adapters from external tracking outputs to the canonical prediction schema."""

from __future__ import annotations

from pathlib import Path

__all__ = ["convert_acts_colliderml"]


def convert_acts_colliderml(
    dataset_root: Path | str, n_files: int, out_dir: Path | str
) -> Path:
    """Lazily import and run the ColliderML ACTS adapter."""
    from .acts import convert_acts_colliderml as convert

    return convert(dataset_root, n_files, out_dir)
