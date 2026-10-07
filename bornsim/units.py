"""TypedUnit's shared registry and SI conversions used by BornSim.

Bare numbers retain the legacy SI convention: lengths in metres, angles in
radians, and refractive indices dimensionless. Quantity inputs are converted
explicitly and incompatible dimensions are rejected before computation.
"""

import numpy as np
from pint.errors import DimensionalityError
from TypedUnit import Angle, Dimensionless, Length, Quantity, RefractiveIndex, ureg

__all__ = ["ureg", "Quantity", "Length", "Angle", "Dimensionless", "RefractiveIndex"]


def _si(*, value, unit, name, scalar=False):
    if isinstance(value, Quantity):
        try:
            value = value.to(unit).magnitude
        except DimensionalityError as error:
            raise ValueError(f"{name} must have units compatible with {unit}.") from error
    data = np.asarray(value, dtype=float)
    if scalar:
        if data.ndim != 0:
            raise ValueError(f"{name} must be a scalar.")
        return float(data)
    return data


def _quantity(*, value, unit, name):
    if isinstance(value, Quantity):
        try:
            value = value.to(unit).magnitude
        except DimensionalityError as error:
            raise ValueError(f"{name} must have units compatible with {unit}.") from error
    return ureg.Quantity(np.array(value, copy=True), unit)
