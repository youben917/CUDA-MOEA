import torch  # noqa: F401 — must be imported before _C so that libtorch_python
# is initialized before any at::Tensor crosses the Python boundary.

from .algorithm import Algorithm, AlgorithmBuilder, NSGA3, Population, RVEA, Result
from . import problem as _problem
from .problem import *
from ._specs import (
    AdaptiveRVEADirections, CudaConfig, DasDennisDirections,
    NSGA3EnvironmentSelector, NoMutation, PolynomialMutation,
    PythonCrossover, PythonEnvironmentSelector, PythonMating, PythonMutation,
    PythonReferenceDirections, RandomMating, RVEAEnvironmentSelector, SBX,
    TournamentMating, UserDefinedDirections,
)

__version__ = "0.2.0"

__all__ = [
    "NativeProblem",
    "Algorithm", "AlgorithmBuilder", "NSGA3", "RVEA", "Population", "Result",
    "CudaConfig", "SBX", "PolynomialMutation", "NoMutation",
    "RandomMating", "TournamentMating", "DasDennisDirections",
    "AdaptiveRVEADirections", "UserDefinedDirections",
    "NSGA3EnvironmentSelector", "RVEAEnvironmentSelector",
    "PythonProblem", "PythonMating", "PythonCrossover", "PythonMutation",
    "PythonReferenceDirections", "PythonEnvironmentSelector",
] + _problem.__all__


def __getattr__(name):
    # Keep `python -m cuda_moea.native` free of runpy double-import warnings.
    if name == "NativeProblem":
        from .native import NativeProblem
        return NativeProblem
    raise AttributeError(name)
