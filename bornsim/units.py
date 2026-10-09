"""Shared TypedUnit registry and explicit quantity validation.

Validation preserves the supplied units. Dimensional inputs require quantities;
dimensionless inputs may be bare numbers. Numerical kernels, plotting and
serialization explicitly select units when they need magnitudes.
"""

from typing import Any, Literal, overload
import numpy as np
from numpy.typing import NDArray
from TypedUnit import Angle, Dimensionless, Length, Quantity, RefractiveIndex, ureg

__all__ = ["ureg", "Quantity", "Length", "Angle", "Dimensionless", "RefractiveIndex", "validate_units"]


def validate_units(value: object, *, unit: str, name: str, scalar: bool = False) -> None:
    """Check units and shape without converting or stripping the quantity."""

    if not isinstance(value, Quantity):
        raise ValueError(f"{name} requires an explicit quantity with units compatible with {unit}.")

    if unit == "radian" and value.units == ureg.dimensionless:
        raise ValueError(f"{name} requires explicit angular units, such as degree or radian.")

    if not value.is_compatible_with(unit):
        raise ValueError(f"{name} must have units compatible with {unit}.")

    if scalar and np.ndim(value.magnitude) != 0:
        raise ValueError(f"{name} must be a scalar.")


@overload
def _dimensionless(*, value: Any, name: str, scalar: Literal[True]) -> float: ...


@overload
def _dimensionless(*, value: Any, name: str, scalar: Literal[False] = False) -> NDArray[np.float64]: ...


def _dimensionless(*, value: Any, name: str, scalar: bool = False) -> float | NDArray[np.float64]:
    """Validate dimensionless numbers, including scaled units such as percent."""

    if isinstance(value, Quantity):
        validate_units(
            value,
            unit="dimensionless",
            name=name,
            scalar=scalar,
        )

        value = value.to("dimensionless").magnitude

    data = np.asarray(value, dtype=float)

    if scalar:
        if data.ndim != 0:
            raise ValueError(f"{name} must be a scalar.")

        return float(data)

    return data


@overload
def _refractive_index_values(*, value: Any, name: str, scalar: Literal[True]) -> float: ...


@overload
def _refractive_index_values(*, value: Any, name: str, scalar: Literal[False] = False) -> NDArray[np.float64]: ...


def _refractive_index_values(*, value: Any, name: str, scalar: bool = False) -> float | NDArray[np.float64]:
    """Require plain numerical refractive indices and fluctuations, without units."""

    if isinstance(value, Quantity):
        raise ValueError(f"{name} must be unitless; supply plain numbers without units.")

    if scalar:
        return _dimensionless(value=value, name=name, scalar=True)

    return _dimensionless(value=value, name=name)
