"""Proper three-dimensional rotations for immutable material shapes."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, validate_units, _dimensionless


@dataclass(frozen=True, kw_only=True)
class Rotation:
    """Define an active right-handed rotation about a dimensionless axis.

    angle accepts angular quantities; angular values require explicit units. The axis is
    normalized. matrix maps local column coordinates into global coordinates.
    No numerical scattering or voxel resampling is performed by this class.
    """

    axis: tuple[float, float, float]
    angle: Quantity

    def __post_init__(self) -> None:
        axis = _dimensionless(value=self.axis, name="axis")

        validate_units(
            self.angle,
            unit="radian",
            name="angle",
            scalar=True,
        )

        if axis.shape != (3,) or np.any(~np.isfinite(axis)) or np.linalg.norm(axis) == 0:
            raise ValueError("axis must be a finite, nonzero three-vector.")

        if not np.isfinite(self.angle):
            raise ValueError("angle must be finite.")

        object.__setattr__(self, "axis", tuple(axis / np.linalg.norm(axis)))

    @property
    def matrix(self) -> np.ndarray:
        """Return Rodrigues' active rotation matrix, shape (3, 3)."""

        axis = np.asarray(self.axis)

        x, y, z = axis

        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

        cosine = float(np.cos(self.angle))

        sine = float(np.sin(self.angle))

        return cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * skew

    @staticmethod
    def _matrix(*, rotation):
        matrix = rotation.matrix if isinstance(rotation, Rotation) else _dimensionless(value=rotation, name="rotation")

        invalid = (
            matrix.shape != (3, 3)
            or np.any(~np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(matrix), 1, rtol=0, atol=1e-10)
        )

        if invalid:
            raise ValueError("rotation must be a proper orthonormal 3-by-3 matrix or Rotation.")

        return matrix
