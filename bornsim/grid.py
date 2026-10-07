"""Reusable centred cubic voxel grids with numeric SI coordinates."""

from dataclasses import dataclass
import numpy as np
import warnings
from ._validation import _integer
from .units import Quantity, _si


@dataclass(frozen=True, kw_only=True)
class Grid:
    """Define the spatial discretization once for generation and scattering.

    Parameters
    ----------
    shape : tuple of int, optional
        Three voxel counts, each from 2 to 32. Default is (12, 12, 12).
    spacing : float or Quantity, optional
        Positive, finite cubic voxel width, stored in metres. Default is 50 nm.

    Notes
    -----
    Voxel centres are ``(i - (N - 1)/2) * spacing`` on each axis. The box
    extends half a voxel beyond its outer centres. It describes the sampled
    domain, not an additional dielectric boundary. Background index belongs
    to the medium or volume, rather than the spatial grid.
    """

    shape: tuple = (12, 12, 12)
    spacing: Quantity | float = 50e-9

    def __post_init__(self):
        if not isinstance(self.shape, (tuple, list)) or len(self.shape) != 3:
            raise ValueError("shape must have three axes.")
        shape = tuple(_integer(value=n, name="axis size", low=2, high=32) for n in self.shape)
        spacing = _si(
            value=self.spacing,
            unit="meter",
            name="spacing",
            scalar=True,
        )
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive.")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing", spacing)

    def __repr__(self):
        return f"Grid(shape={self.shape}, spacing={self.spacing:g} m)"

    @property
    def positions(self):
        """Voxel-centre coordinates in metres, shape ``(*shape, 3)``."""
        axes = [(np.arange(n) - (n - 1) / 2) * self.spacing for n in self.shape]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    @property
    def volume(self):
        """Physical voxel-box volume in cubic metres."""
        return float(np.prod(self.shape) * self.spacing**3)

    @property
    def metadata(self):
        """Fresh JSON-compatible grid settings in SI units."""
        return {"shape": list(self.shape), "spacing_m": self.spacing}

    @classmethod
    def _resolve(cls, *, grid=None, shape=None, spacing=None):
        if grid is not None:
            if not isinstance(grid, cls):
                raise TypeError("grid must be a Grid.")
            if shape is not None or spacing is not None:
                raise ValueError("Supply grid or shape/spacing, not both.")
            return grid
        if shape is not None or spacing is not None:
            warnings.warn("Use Grid instead of shape/spacing keywords.", DeprecationWarning, stacklevel=3)
        return cls(
            shape=(12, 12, 12) if shape is None else shape,
            spacing=50e-9 if spacing is None else spacing,
        )
