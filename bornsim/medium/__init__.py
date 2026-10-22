"""Finite media for numerical Born calculations."""

from .random_spheres import RandomSphereMedium
from .base import Medium
from .random_medium import GaussianMedium, ExponentialMedium, WhittleMaternMedium

__all__ = ["Medium", "GaussianMedium", "ExponentialMedium", "WhittleMaternMedium", "RandomSphereMedium"]

__all__ += ["StructuredMedium", "PeriodicMedium", "CoreShellMedium", "EllipsoidMedium"]


def __getattr__(name):
    """Resolve composed media after geometry initialization."""
    if name in ("StructuredMedium", "PeriodicMedium"):
        from . import structured

        return getattr(structured, name)

    if name in ("CoreShellMedium", "EllipsoidMedium"):
        from . import particles

        return getattr(particles, name)

    raise AttributeError(name)
