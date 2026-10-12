"""Reusable centred cubic voxel grids with unit-bearing coordinates."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, _integer, validate_units


@dataclass(frozen=True, kw_only=True)
class Grid:
    """Define the spatial discretization once for generation and scattering.

    Parameters
    ----------
    shape : tuple of int
        Three explicitly chosen voxel counts, each from 2 to 32.
    spacing : Quantity
        Positive, finite cubic voxel width, with its supplied length units. A spacing must be supplied.


    Use :meth:`from_extent` when the full physical voxel-box extent is known
    instead of the voxel spacing.

    Notes
    -----
    Voxel centres are ``(i - (N - 1)/2) * spacing`` on each axis. The box
    extends half a voxel beyond its outer centres. It describes the sampled
    domain, not an additional dielectric boundary. Background refractive index belongs
    to the medium or volume, rather than the spatial grid.
    """

    shape: tuple[int, int, int]
    spacing: Quantity

    @classmethod
    def from_extent(cls, *, shape: tuple[int, int, int], extent: Quantity) -> "Grid":
        """Create an isotropic voxel grid from its full physical extent.

        ``extent`` gives the full voxel-box width along x, y, and z. Each
        axis extent must imply the same cubic voxel spacing.
        """

        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError("shape must have three axes.")

        normalized_shape = (
            _integer(value=shape[0], name="axis size", low=2, high=32),
            _integer(value=shape[1], name="axis size", low=2, high=32),
            _integer(value=shape[2], name="axis size", low=2, high=32),
        )

        validate_units(
            extent,
            unit="meter",
            name="extent",
            scalar=False,
        )

        if np.shape(extent) != (3,):
            raise ValueError("extent must contain three axis lengths.")

        if np.any(~np.isfinite(extent)) or np.any(extent <= 0):
            raise ValueError("extent must be finite and positive.")

        spacing = extent / np.asarray(normalized_shape)

        spacing_magnitudes = spacing.to("meter").magnitude

        if not np.allclose(spacing_magnitudes, spacing_magnitudes[0], rtol=1e-12, atol=0):
            raise ValueError("extent and shape must imply the same cubic voxel spacing on every axis.")

        return cls(
            shape=normalized_shape,
            spacing=spacing[0],
        )

    def __post_init__(self) -> None:
        if not isinstance(self.shape, (tuple, list)) or len(self.shape) != 3:
            raise ValueError("shape must have three axes.")

        shape = tuple(_integer(value=n, name="axis size", low=2, high=32) for n in self.shape)

        validate_units(
            self.spacing,
            unit="meter",
            name="spacing",
            scalar=True,
        )

        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive.")

        object.__setattr__(self, "shape", shape)

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape}, spacing={self.spacing:~g})"

    @property
    def positions(self) -> Quantity:
        """Unit-bearing voxel-centre coordinates, shape ``(*shape, 3)``."""

        axes = [np.arange(n) - (n - 1) / 2 for n in self.shape]

        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1) * self.spacing

    @property
    def volume(self) -> Quantity:
        """Physical voxel-box volume in cubic metres."""

        return np.prod(self.shape) * self.spacing**3

    @property
    def metadata(self) -> dict[str, object]:
        """Fresh JSON-compatible grid settings in SI units."""

        return {"shape": list(self.shape), "spacing_m": float(self.spacing.to("meter").magnitude)}

    @classmethod
    def _resolve(cls, *, grid: "Grid") -> "Grid":
        if not isinstance(grid, cls):
            raise TypeError("grid must be a Grid.")

        return grid
