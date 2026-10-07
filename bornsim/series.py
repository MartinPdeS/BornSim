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
from .results import BornResult
from .units import Quantity, _si
from .volume import Volume


@dataclass(frozen=True, kw_only=True, eq=False)
class BornSeries:
    """Own the fixed geometry and propagation operator for numerical Born orders.

    Parameters
    ----------
    grid : Grid, optional
        Shared spatial configuration, also used by the input Volume.
        Cannot be combined with legacy shape or spacing keywords.
    shape : tuple of int, optional
        Three grid dimensions, each from 2 to 32.
    spacing : float or Quantity, optional
        Positive cubic voxel width; bare lengths are metres.
    background_index : float or Quantity
        Positive uniform background index, dimensionless.
    wavelength : float or Quantity
        Positive vacuum wavelength; bare lengths are metres.
    directions : array_like or Quantity
        Dimensionless unit observation vectors, shape (observation, 3), with
        1 to 16384 observations. Copied and made read-only.
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
    ``chi = 2*n0*delta_index`` and K integrates the background dyadic Green tensor
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

    shape: tuple | None = None
    spacing: Quantity | float = None
    grid: Grid | None = None
    background_index: Quantity | float
    wavelength: Quantity | float
    directions: Quantity | np.ndarray
    order: int = 3
    operator: GreenOperator = field(init=False, repr=False)
    _positions: np.ndarray = field(init=False, repr=False)
    _incident_field: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        grid = Grid._resolve(
            grid=self.grid,
            shape=self.shape,
            spacing=self.spacing,
        )
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "shape", grid.shape)
        object.__setattr__(self, "spacing", grid.spacing)
        volume = Volume(
            delta_index=np.zeros(grid.shape),
            grid=grid,
            background_index=self.background_index,
        )
        shape = grid.shape
        wavelength = _si(
            value=self.wavelength,
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
        directions = np.array(
            _si(
                value=self.directions,
                unit="dimensionless",
                name="directions",
            ),
            copy=True,
        )
        if directions.ndim != 2 or directions.shape[1] != 3 or not 1 <= len(directions) <= 16384:
            raise ValueError("directions must contain 1–16384 unit 3-vectors.")
        invalid_directions = not np.all(np.isfinite(directions)) or not np.allclose(
            np.linalg.norm(directions, axis=1), 1, atol=1e-10, rtol=0
        )
        if invalid_directions:
            raise ValueError("directions must be finite unit vectors.")
        directions.setflags(write=False)
        positions = grid.positions
        positions.setflags(write=False)
        phase = np.exp(1j * 2 * np.pi / wavelength * volume.background_index * positions[..., 2])
        incident = phase[..., None, None] * np.eye(3)[:2]
        incident.setflags(write=False)
        operator = GreenOperator(
            shape=shape,
            spacing=grid.spacing,
            wavelength=wavelength,
            background_index=volume.background_index,
        )
        for name, value in (
            ("shape", shape),
            ("spacing", grid.spacing),
            ("background_index", volume.background_index),
            ("wavelength", wavelength),
            ("directions", directions),
            ("order", order),
            ("operator", operator),
            ("_positions", positions),
            ("_incident_field", incident),
        ):
            object.__setattr__(self, name, value)

    def _far_field(self, *, source):
        """Project the voxel source onto transverse outgoing amplitudes in metres."""
        k0 = 2 * np.pi / self.wavelength
        k = k0 * self.background_index
        positions = self._positions.reshape(-1, 3)
        values = source.reshape(-1, 2, 3)
        amplitude = np.empty((len(self.directions), 2, 3), dtype=complex)
        for start in range(0, len(self.directions), 64):
            direction = self.directions[start : start + 64]
            phase = np.exp(-1j * k * (direction @ positions.T))
            integral = np.einsum("dn,npv->dpv", phase, values) * k0**2 * self.spacing**3 / (4 * np.pi)
            amplitude[start : start + 64] = (
                integral - direction[:, None, :] * np.einsum("dv,dpv->dp", direction, integral)[..., None]
            )
        return amplitude

    def solve(self, *, volume: Volume):
        """Compute coherent scattering from one compatible finite index volume.

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
        ``2*n0*delta_index`` at every order. Growing field terms generate a
        diagnostic; decreasing terms do not certify convergence.

        Examples
        --------
        >>> from bornsim import BornSeries, RandomMedium
        >>> medium = RandomMedium(correlation="gaussian")

        >>> volume = medium.to_volume(
        ...     shape=(2, 2, 2),
        ...     seed=42,
        ... )
        >>> engine = BornSeries(
        ...     shape=volume.delta_index.shape,
        ...     spacing=volume.spacing,
        ...     background_index=volume.background_index,
        ...     wavelength=633e-9,
        ...     directions=[[0.0, 0.0, 1.0]],
        ...     order=2,
        ... )
        >>> result = engine.solve(volume=volume)
        >>> result.amplitudes.shape
        (2, 1, 2, 3)
        """
        if not isinstance(volume, Volume):
            raise TypeError("volume must be a Volume.")
        incompatible_volume = (
            volume.delta_index.shape != self.shape
            or volume.spacing != self.spacing
            or volume.background_index != self.background_index
        )
        if incompatible_volume:
            raise ValueError("volume grid and background_index must match the BornSeries configuration.")
        field = self._incident_field
        incident_norm = np.linalg.norm(field)
        contrast = 2 * volume.background_index * volume.delta_index
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
        differential = np.sum(np.abs(cumulative) ** 2, axis=(-1, -2)) / (2 * volume.volume)
        separate = np.sum(np.abs(amplitudes) ** 2, axis=(-1, -2)) / (2 * volume.volume)
        warnings = []
        if volume.spacing > self.wavelength / volume.background_index / 10:
            warnings.append("Voxel spacing exceeds one tenth of the background wavelength; refine the grid.")
        if len(norms) > 1 and any(b >= a and b > 0 for a, b in zip(norms, norms[1:])):
            warnings.append("Successive field terms are not decreasing; Born convergence is not established.")
        return BornResult(
            directions=self.directions.copy(),
            amplitudes=amplitudes,
            differential=differential,
            term_differential=separate,
            field_norms=np.array(norms),
            warnings=tuple(warnings),
        )
