"""Reusable centred cubic voxel grids with unit-bearing coordinates."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, validate_quantity


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

        Parameters
        ----------
        shape : tuple of int
            Voxel counts along x, y, and z; each must be between 2 and 32.
        extent : Quantity
            Full voxel-box lengths along x, y, and z, with explicit length
            units. All three lengths must imply the same voxel spacing.

        Returns
        -------
        Grid
            Grid with spacing derived from the first extent component.

        Raises
        ------
        ValueError
            If shape, units, extent values, or implied spacings are invalid.
        """

        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError("shape must have three axes.")

        normalized_shape_values: list[int] = []

        for axis_size in shape:
            if isinstance(axis_size, bool) or not isinstance(axis_size, (int, np.integer)) or not 2 <= axis_size <= 32:
                raise ValueError("axis size must be an integer between 2 and 32.")

            normalized_shape_values.append(int(axis_size))

        normalized_shape = (
            normalized_shape_values[0],
            normalized_shape_values[1],
            normalized_shape_values[2],
        )

        validate_quantity(
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
        """Validate grid dimensions and explicit positive voxel spacing.

        Raises
        ------
        ValueError
            If shape is not three valid voxel counts or spacing is invalid.
        """
        if not isinstance(self.shape, (tuple, list)) or len(self.shape) != 3:
            raise ValueError("shape must have three axes.")

        shape_values: list[int] = []

        for axis_size in self.shape:
            if isinstance(axis_size, bool) or not isinstance(axis_size, (int, np.integer)) or not 2 <= axis_size <= 32:
                raise ValueError("axis size must be an integer between 2 and 32.")

            shape_values.append(int(axis_size))

        shape = (shape_values[0], shape_values[1], shape_values[2])

        validate_quantity(
            self.spacing,
            unit="meter",
            name="spacing",
            scalar=True,
        )

        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive.")

        object.__setattr__(self, "shape", shape)

    def __repr__(self) -> str:
        """Return a concise representation including shape and spacing."""
        return f"Grid(shape={self.shape}, spacing={self.spacing:~g})"

    @property
    def positions(self) -> Quantity:
        """Return unit-bearing voxel-centre coordinates.

        Returns
        -------
        Quantity
            Cartesian coordinates with shape ``(*shape, 3)`` and the spacing's
            length units. The returned array is newly computed.
        """

        axes = [np.arange(n) - (n - 1) / 2 for n in self.shape]

        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1) * self.spacing

    @property
    def volume(self) -> Quantity:
        """Return the physical voxel-box volume.

        Returns
        -------
        Quantity
            Box volume in cubic metres, including one full voxel per grid
            element.
        """

        return np.prod(self.shape) * self.spacing**3

    @property
    def metadata(self) -> dict[str, object]:
        """Return fresh JSON-compatible grid settings in SI units.

        Returns
        -------
        dict of str to object
            Voxel shape and spacing in metres.
        """

        return {"shape": list(self.shape), "spacing_m": float(self.spacing.to("meter").magnitude)}

    @classmethod
    def _resolve(cls, *, grid: "Grid") -> "Grid":
        """Validate and return an existing Grid.

        Parameters
        ----------
        grid : Grid
            Grid instance to validate.

        Returns
        -------
        Grid
            The same instance.

        Raises
        ------
        TypeError
            If ``grid`` is not a Grid.
        """
        if not isinstance(grid, cls):
            raise TypeError("grid must be a Grid.")

        return grid
