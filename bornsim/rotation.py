"""Proper three-dimensional rotations for immutable material shapes."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, _si


@dataclass(frozen=True, kw_only=True)
class Rotation:
    """Define an active right-handed rotation about a dimensionless axis.

    angle accepts angular quantities; bare values mean radians. The axis is
    normalized. matrix maps local column coordinates into global coordinates.
    No numerical scattering or voxel resampling is performed by this class.
    """

    axis: tuple = (0.0, 0.0, 1.0)
    angle: Quantity | float = 0.0

    def __post_init__(self):
        axis = _si(value=self.axis, unit="dimensionless", name="axis")
        angle = _si(value=self.angle, unit="radian", name="angle", scalar=True)
        if axis.shape != (3,) or np.any(~np.isfinite(axis)) or np.linalg.norm(axis) == 0:
            raise ValueError("axis must be a finite, nonzero three-vector.")
        if not np.isfinite(angle):
            raise ValueError("angle must be finite.")
        object.__setattr__(self, "axis", tuple(axis / np.linalg.norm(axis)))
        object.__setattr__(self, "angle", angle)

    @property
    def matrix(self):
        """Return Rodrigues' active rotation matrix, shape (3, 3)."""
        axis = np.asarray(self.axis)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return (
            np.cos(self.angle) * np.eye(3) + (1 - np.cos(self.angle)) * np.outer(axis, axis) + np.sin(self.angle) * skew
        )

    @staticmethod
    def _matrix(*, rotation):
        matrix = (
            rotation.matrix
            if isinstance(rotation, Rotation)
            else _si(value=rotation, unit="dimensionless", name="rotation")
        )
        invalid = (
            matrix.shape != (3, 3)
            or np.any(~np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(matrix), 1, rtol=0, atol=1e-10)
        )
        if invalid:
            raise ValueError("rotation must be a proper orthonormal 3-by-3 matrix or Rotation.")
        return matrix
