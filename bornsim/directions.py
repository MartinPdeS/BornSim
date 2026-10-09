"""Validated observation directions for numerical scattering."""

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from .units import Quantity, _dimensionless, validate_units, ureg


@dataclass(frozen=True, kw_only=True, init=False, eq=False, repr=False)
class Directions:
    """Define an ordered collection of Cartesian unit observation vectors.

    Parameters
    ----------
    vectors : array_like or Quantity
        Explicit dimensionless vectors of shape (n, 3), with 1 to 16384 rows.
        Every vector must be finite and have unit length. Vectors are copied
        into an immutable array; they are never normalized automatically.

    Notes
    -----
    Coordinates use the global x, y, z axes. The incident wave propagates
    along positive z. Vector order is preserved in scattered amplitudes.
    Use ``vectors`` to access the NumPy array and :meth:`from_angles` to
    construct vectors from explicitly unit-bearing spherical coordinates.
    """

    vectors: NDArray[np.float64]

    def __init__(self, *, vectors: NDArray[np.float64] | Sequence[Sequence[float]] | Quantity) -> None:
        coordinate_values = vectors.magnitude if isinstance(vectors, Quantity) else vectors

        if np.iscomplexobj(coordinate_values):
            raise ValueError("directions.vectors must contain real Cartesian coordinates.")

        vectors = _dimensionless(value=vectors, name="directions.vectors")

        if vectors.ndim != 2 or vectors.shape[1] != 3 or not 1 <= len(vectors) <= 16384:
            raise ValueError("directions.vectors must contain 1–16384 unit 3-vectors.")

        invalid_vectors = not np.all(np.isfinite(vectors)) or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1, atol=1e-10, rtol=0
        )

        if invalid_vectors:
            raise ValueError("directions.vectors must be finite unit vectors.")

        # Immutable backing storage prevents changing a cached solver configuration.
        immutable_vectors = np.frombuffer(vectors.tobytes(), dtype=np.float64).reshape(vectors.shape)

        object.__setattr__(self, "vectors", immutable_vectors)

    @classmethod
    def from_angles(cls, *, polar_angles: Quantity, azimuth_angles: Quantity) -> "Directions":
        """Convert paired or broadcastable spherical angles into unit vectors.

        Polar angles are measured from positive z and lie in [0, pi]. Azimuth
        angles are measured from positive x toward positive y and are periodic.
        Both inputs require explicit angular units. Broadcast coordinates are
        flattened in row-major order, preserving polar/azimuth grid ordering.
        """

        validate_units(
            polar_angles,
            unit="radian",
            name="polar_angles",
        )

        validate_units(
            azimuth_angles,
            unit="radian",
            name="azimuth_angles",
        )

        invalid_polar_angles = np.any(~np.isfinite(polar_angles)) or np.any(
            (polar_angles < 0) | (polar_angles > np.pi * ureg.radian)
        )

        if invalid_polar_angles:
            raise ValueError("polar_angles must be finite and between zero and pi.")

        if np.any(~np.isfinite(azimuth_angles)):
            raise ValueError("azimuth_angles must be finite.")

        # Trigonometric quantities become dimensionless Cartesian coordinates.
        cartesian_components = np.broadcast_arrays(
            (np.sin(polar_angles) * np.cos(azimuth_angles)).magnitude,
            (np.sin(polar_angles) * np.sin(azimuth_angles)).magnitude,
            np.cos(polar_angles).magnitude,
        )

        return cls(vectors=np.stack(cartesian_components, axis=-1).reshape(-1, 3))

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return f"Directions(n_vectors={len(self)})"
