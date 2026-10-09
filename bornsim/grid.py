"""Reusable centred cubic voxel grids with unit-bearing coordinates."""

from dataclasses import dataclass
import numpy as np
import warnings
from ._validation import _integer
from .units import Quantity, validate_units


@dataclass(frozen=True, kw_only=True)
class Grid:
    """Define the spatial discretization once for generation and scattering.

    Parameters
    ----------
    shape : tuple of int
        Three explicitly chosen voxel counts, each from 2 to 32.
    spacing : Quantity
        Positive, finite cubic voxel width, with its supplied length units. A spacing must be supplied.

    Notes
    -----
    Voxel centres are ``(i - (N - 1)/2) * spacing`` on each axis. The box
    extends half a voxel beyond its outer centres. It describes the sampled
    domain, not an additional dielectric boundary. Background refractive index belongs
    to the medium or volume, rather than the spatial grid.
    """

    shape: tuple[int, int, int]
    spacing: Quantity

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
    def _resolve(
        cls,
        *,
        grid: "Grid | None" = None,
        shape: tuple[int, int, int] | None = None,
        spacing: Quantity | None = None,
    ) -> "Grid":
        if grid is not None:
            if not isinstance(grid, cls):
                raise TypeError("grid must be a Grid.")

            if shape is not None or spacing is not None:
                raise ValueError("Supply grid or shape/spacing, not both.")

            return grid

        if shape is None or spacing is None:
            raise ValueError("Supply grid or both shape and spacing; physical grid settings have no defaults.")

        warnings.warn("Use Grid instead of shape/spacing keywords.", DeprecationWarning, stacklevel=3)

        return cls(
            shape=shape,
            spacing=spacing,
        )
