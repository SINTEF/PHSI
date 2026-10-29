"""
Port-Hamiltonian system identification from measured trajectories.
"""

from .errors import FitError, SimulationError, SupportNonconvergence
from .estimator import Model, load, model
from .libraries import (
    CustomLibrary,
    DampingLibrary,
    GaussianExtension,
    LogCoshExtension,
    Neural,
    Polynomial,
)

__version__ = "1.0.0a1"
__all__ = [
    "Model",
    "model",
    "load",
    "Polynomial",
    "GaussianExtension",
    "LogCoshExtension",
    "DampingLibrary",
    "Neural",
    "CustomLibrary",
    "FitError",
    "SimulationError",
    "SupportNonconvergence",
]
