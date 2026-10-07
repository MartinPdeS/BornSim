"""Class-based API for vector Born scattering from finite dielectric samples."""

from ._version import __version__
from .model import AnalyticalMedium
from .media import Medium, RandomMedium
from .volume import Volume
from .geometry import Layer, Sphere, Ellipsoid, Box, Cylinder, StructuredMedium
from .rotation import Rotation
from .source import Source
from .solver import Solver
from .grid import Grid
from .sampling import AngularSampling
from .results import Result
from .angular_data import AngularData
from .material import Material
from .ensemble_sampling import EnsembleSampling

__all__ = [
    "Material",
    "EnsembleSampling",
    "__version__",
    "Medium",
    "RandomMedium",
    "AnalyticalMedium",
    "StructuredMedium",
    "Volume",
    "Source",
    "Solver",
    "Grid",
    "AngularSampling",
    "AngularData",
    "Result",
    "Layer",
    "Sphere",
    "Ellipsoid",
    "Box",
    "Cylinder",
    "Rotation",
]


def __getattr__(name):
    """Keep deprecated advanced imports available during API migration."""
    from importlib import import_module
    import warnings

    locations = {
        "BornSeries": "series",
        "ensemble_scattering": "ensemble",
        "random_volume": "media",
        "angular_scattering": "model",
        "optical_properties": "model",
    }
    if name not in locations:
        raise AttributeError(f"module 'bornsim' has no attribute {name!r}")
    module = locations[name]
    warnings.warn(
        f"Import {name} from bornsim.{module}; top-level convenience imports are deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module(f".{module}", __name__), name)
