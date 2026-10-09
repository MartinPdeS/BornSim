"""Reusable finite-volume vector Born-series engine.

Time convention exp(-i omega t). Relative dielectric contrast is linearized
as 2 n0 delta_n at every order. Amplitudes interfere coherently before the
unpolarized intensity is evaluated.
"""

from dataclasses import dataclass, field
import numpy as np
from ._validation import _integer
from .green import GreenOperator
from .grid import Grid
from .directions import Directions
from .results import BornResult
from .units import Quantity, validate_units
from .volume import Volume


@dataclass(frozen=True, kw_only=True, eq=False)
class BornSeries:
    """Own the fixed geometry and propagation operator for numerical Born orders.

    Parameters
    ----------
    grid : Grid
        Shared spatial configuration, also used by the input Volume.
        Required for generation and scattering.
    background_refractive_index : float
        Positive uniform background refractive index, dimensionless.
    wavelength : Quantity
        Positive vacuum wavelength; explicit length units are required.
    directions : Directions
        Validated, immutable Cartesian unit observation vectors.
    order : int, optional
        Highest cumulative Born order, from 1 to 12; default 3.

    Attributes
    ----------
    operator : bornsim.green.GreenOperator
        One open-boundary FFT Green operator reused for every compatible volume.

    Notes
    -----
    The cached incident plane wave travels along +z with x and y polarizations.
    For each call, ``E[0] = E_inc`` and ``E[m] = K(chi*E[m-1])``, where
    ``chi = 2*n0*delta_refractive_index`` and K integrates the background dyadic Green tensor
    multiplied by ``k0**2``. Order-m far-field amplitude is computed from
    ``chi*E[m-1]``. Cumulative intensity is ``|sum_m f[m]|**2``, averaged over the two
    polarizations and divided by the entire voxel-box volume.

    Geometry, incident field, and the Green tensor are reused; contrast,
    propagated fields, amplitudes, and diagnostics are local to each solve.
    Compatible volumes must have exactly the configured grid and background.
    Configuration is frozen to prevent stale propagation settings. Decreasing
    field terms do not establish universal convergence; the dielectric
    linearization is unchanged at higher orders.
    """

    grid: Grid
    shape: tuple = field(init=False)
    spacing: Quantity = field(init=False)
    background_refractive_index: float
    wavelength: Quantity
    directions: Directions
    order: int = 3
    operator: GreenOperator = field(init=False, repr=False)
    _positions: Quantity = field(init=False, repr=False)
    _incident_field: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grid = Grid._resolve(grid=self.grid)

        object.__setattr__(self, "grid", grid)

        volume = Volume(
            delta_refractive_index=np.zeros(grid.shape),
            grid=grid,
            background_refractive_index=self.background_refractive_index,
        )

        shape = grid.shape

        wavelength = self.wavelength

        validate_units(
            self.wavelength,
            unit="meter",
            name="wavelength",
            scalar=True,
        )

        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be finite and positive.")

        order = _integer(
            value=self.order,
            name="order",
            low=1,
            high=12,
        )

        if not isinstance(self.directions, Directions):
            raise TypeError("directions must be a Directions instance.")

        directions = self.directions

        positions = grid.positions

        positions.magnitude.setflags(write=False)

        phase = np.exp(1j * 2 * np.pi / wavelength * volume.background_refractive_index * positions[..., 2])

        incident = phase.to("dimensionless").magnitude[..., None, None] * np.eye(3)[:2]

        incident.setflags(write=False)

        operator = GreenOperator(
            shape=shape,
            spacing=grid.spacing,
            wavelength=wavelength,
            background_refractive_index=volume.background_refractive_index,
        )

        for name, value in (
            ("shape", shape),
            ("spacing", grid.spacing),
            ("background_refractive_index", volume.background_refractive_index),
            ("wavelength", wavelength.copy()),
            ("directions", directions),
            ("order", order),
            ("operator", operator),
            ("_positions", positions),
            ("_incident_field", incident),
        ):
            object.__setattr__(self, name, value)

    def _far_field(self, *, source: np.ndarray) -> np.ndarray:
        """Project the voxel source onto transverse outgoing amplitudes in metres."""

        # Fourier projection uses numeric SI coordinates and amplitudes in metres.
        k0 = 2 * np.pi / float(self.wavelength.to("meter").magnitude)

        k = k0 * self.background_refractive_index

        positions = self._positions.to("meter").magnitude.reshape(-1, 3)

        voxel_spacing_m = float(self.spacing.to("meter").magnitude)

        values = source.reshape(-1, 2, 3)

        amplitude = np.empty((len(self.directions), 2, 3), dtype=complex)

        for start in range(0, len(self.directions), 64):
            direction = self.directions.vectors[start : start + 64]

            phase = np.exp(-1j * k * (direction @ positions.T))

            integral = np.einsum("dn,npv->dpv", phase, values) * k0**2 * voxel_spacing_m**3 / (4 * np.pi)

            amplitude[start : start + 64] = (
                integral - direction[:, None, :] * np.einsum("dv,dpv->dp", direction, integral)[..., None]
            )

        return amplitude

    def solve(self, *, volume: Volume) -> BornResult:
        """Compute coherent scattering from one compatible finite refractive index volume.

        Parameters
        ----------
        volume : Volume
            Fixed fluctuation field with positive linearized permittivity,
            matching this engine's grid spacing, dimensions and background.

        Returns
        -------
        result : bornsim.results.BornResult
            Fresh numeric SI amplitudes, cumulative and isolated intensities,
            relative field norms, and diagnostic messages.

        Raises
        ------
        TypeError
            If volume is not a Volume.
        ValueError
            If grid or background differs from the engine configuration, or
            iteration produces nonfinite fields.

        Notes
        -----
        Each solve starts with the incident field. Amplitudes are summed
        before squaring, preserving interference. Intensities are finite-sample
        cross sections divided by the entire voxel-box volume, not intrinsic
        infinite-medium transport coefficients. The dielectric contrast is
        ``2*n0*delta_refractive_index`` at every order. Growing field terms generate a
        diagnostic; decreasing terms do not certify convergence.

        Examples
        --------
        >>> from bornsim.units import ureg
        >>> from bornsim import Directions, Grid, GaussianMedium
        >>> from bornsim.series import BornSeries
        ...
        ...
        >>> medium = GaussianMedium(
        ...     background_refractive_index=1.33,
        ...     refractive_index_std=0.01,
        ...     correlation_length=100e-9 * ureg.meter,
        ... )

        ...
        >>> grid = Grid(
        ...     shape=(2, 2, 2),
        ...     spacing=50 * ureg.nanometer,
        ... )
        >>> volume = medium.to_volume(
        ...     grid=grid,
        ...     seed=42,
        ... )
        >>> directions = Directions(vectors=[[0.0, 0.0, 1.0]])
        >>> engine = BornSeries(
        ...     grid=grid,
        ...     background_refractive_index=volume.background_refractive_index,
        ...     wavelength=633 * ureg.nanometer,
        ...     directions=directions,
        ...     order=2,
        ... )
        >>> result = engine.solve(volume=volume)
        >>> result.amplitudes.shape
        (2, 1, 2, 3)
        """

        if not isinstance(volume, Volume):
            raise TypeError("volume must be a Volume.")

        incompatible_volume = (
            volume.delta_refractive_index.shape != self.shape
            or volume.spacing != self.spacing
            or volume.background_refractive_index != self.background_refractive_index
        )

        if incompatible_volume:
            raise ValueError("volume grid and background_refractive_index must match the BornSeries configuration.")

        field = self._incident_field

        incident_norm = np.linalg.norm(field)

        contrast = 2 * volume.background_refractive_index * volume.delta_refractive_index

        amplitude_terms, norms = [], []

        for _ in range(self.order):
            source = contrast[..., None, None] * field

            amplitude_terms.append(self._far_field(source=source))

            field = self.operator.apply(source=source)

            if not np.all(np.isfinite(field)):
                raise ValueError("Born terms overflowed; reduce contrast or order.")

            norms.append(float(np.linalg.norm(field) / incident_norm))

        amplitudes = np.asarray(amplitude_terms)

        cumulative = np.cumsum(amplitudes, axis=0)

        differential = np.sum(np.abs(cumulative) ** 2, axis=(-1, -2)) / (
            2 * float(volume.volume.to("meter**3").magnitude)
        )

        separate = np.sum(np.abs(amplitudes) ** 2, axis=(-1, -2)) / (2 * float(volume.volume.to("meter**3").magnitude))

        warnings = []

        if volume.spacing > self.wavelength / volume.background_refractive_index / 10:
            warnings.append("Voxel spacing exceeds one tenth of the background wavelength; refine the grid.")

        if len(norms) > 1 and any(b >= a and b > 0 for a, b in zip(norms, norms[1:])):
            warnings.append("Successive field terms are not decreasing; Born convergence is not established.")

        return BornResult(
            directions=self.directions.vectors.copy(),
            amplitudes=amplitudes,
            differential=differential,
            term_differential=separate,
            field_norms=np.array(norms),
            warnings=tuple(warnings),
        )
