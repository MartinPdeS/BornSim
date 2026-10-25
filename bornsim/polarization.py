"""Pure polarization states in a source's local transverse Jones basis."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from .units import Quantity, validate_units


@dataclass(frozen=True, kw_only=True, init=False, eq=False)
class PolarizationState:
    """Define linear polarization by angle or a general pure Jones state.

    Parameters
    ----------
    angle : Quantity, optional
        Linear polarization orientation with explicit angular units. Zero
        aligns with the source's first transverse basis vector; positive
        angles turn toward its second. Cannot be combined with jones_vector.
    jones_vector : array_like, optional
        Two finite, complex, unitless electric-field components. Nonzero
        vectors are normalized to unit incident intensity, retaining relative
        phase. Cannot be combined with angle. Supply exactly one argument.

    Notes
    -----
    Source projects laboratory +x onto the plane normal to its incident
    direction to obtain the first basis vector. For incidence parallel to x,
    laboratory +y is used instead. The second vector is direction cross first,
    so the basis and propagation direction form a right-handed frame.
    For positive-z incidence the Jones basis is exactly laboratory x, y.
    The time convention is exp(-i omega t). Jones vectors describe pure states;
    unpolarized illumination is selected by omitting Source.polarization.
    """

    jones_vector: np.ndarray
    angle: Quantity | None

    def __init__(self, *, angle: Quantity | None = None, jones_vector: ArrayLike | None = None) -> None:
        if (angle is None) == (jones_vector is None):
            raise ValueError("Supply exactly one of angle or jones_vector.")

        if angle is not None:
            validate_units(
                angle,
                unit="radian",
                name="angle",
                scalar=True,
            )

            if np.iscomplexobj(angle.magnitude) or not np.isfinite(angle):
                raise ValueError("angle must be a finite real angular quantity.")

            vector = np.array([np.cos(angle).magnitude, np.sin(angle).magnitude], dtype=complex)
        else:
            if isinstance(jones_vector, Quantity):
                raise TypeError("jones_vector must contain plain unitless components.")

            vector = np.asarray(jones_vector, dtype=complex)

        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError("jones_vector must contain two finite complex components.")

        scale = max(np.max(np.abs(vector.real)), np.max(np.abs(vector.imag)))

        if scale == 0:
            raise ValueError("jones_vector must be nonzero.")

        scaled = vector / scale

        normalized = scaled / np.linalg.norm(scaled)

        immutable = np.frombuffer(normalized.tobytes(), dtype=np.complex128)

        object.__setattr__(self, "jones_vector", immutable)

        object.__setattr__(self, "angle", None if angle is None else angle.copy())


class LeftCircular(PolarizationState):
    """Unit Jones state (1, +i)/sqrt(2), matching PyMieSim's convention."""

    def __init__(self) -> None:
        super().__init__(jones_vector=[1, 1j])


class RightCircular(PolarizationState):
    """Unit Jones state (1, -i)/sqrt(2), matching PyMieSim's convention."""

    def __init__(self) -> None:
        super().__init__(jones_vector=[1, -1j])
