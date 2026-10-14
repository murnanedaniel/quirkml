"""Toy quirk-pair simulator: relativistic infracolor-string trajectories
through a cylindrical barrel, validated against the For_Daniel reference data.
"""
from .pair import QuirkPair
from .detector import Barrel, LAYER_R_CM, N_LAYERS
from . import units

__all__ = ["QuirkPair", "Barrel", "LAYER_R_CM", "N_LAYERS", "units"]
