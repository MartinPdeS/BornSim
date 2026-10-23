"""Class-based API for vector Born scattering from finite dielectric samples."""

from ._version import __version__
from .medium import Medium, GaussianMedium, ExponentialMedium, WhittleMaternMedium
from .medium.random_spheres import RandomSphereMedium
from .volume import Volume
from .geometry import Layer, Sphere, Ellipsoid, Box, Cylinder, StructuredMedium
from .rotation import Rotation
from .source import Source
from .solver import Solver
from .grid import Grid
from .directions import Directions
from .sampling import AngularSampling
from .results import Result
from .convergence import ConvergenceReport
from .angular_data import AngularData
from .material import Material
from .ensemble_sampling import EnsembleSampling
from .medium.structured import PeriodicMedium
from .medium.particles import CoreShellMedium, EllipsoidMedium, CylinderMedium

__all__ = [
    "ConvergenceReport",
    "PeriodicMedium",
    "CoreShellMedium",
    "EllipsoidMedium",
    "CylinderMedium",
    "Directions",
    "Material",
    "EnsembleSampling",
    "__version__",
    "Medium",
    "GaussianMedium",
    "ExponentialMedium",
    "WhittleMaternMedium",
    "RandomSphereMedium",
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
