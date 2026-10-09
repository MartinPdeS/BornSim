"""Numerical random-medium statistics, independent of analytical solutions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from .units import _refractive_index_values, Quantity, validate_units, ureg, _dimensionless
from ._validation import _integer
from .grid import Grid

if TYPE_CHECKING:
    from .volume import Volume


class Medium(ABC):
    """Abstract interface for media that can be sampled into a finite volume.

    Instantiate :class:`RandomMedium` or :class:`bornsim.StructuredMedium`,
    not this base class. Concrete media specify a uniform ``background_refractive_index``
    and implement :meth:`to_volume` using cubic voxels, SI spacing, and centred
    coordinates. The resulting field stores ``delta_refractive_index = n(r) - n0``;
    propagation retains ``epsilon_r = n0**2 + 2*n0*delta_refractive_index``.
    """

    background_refractive_index: float | None

    @property
    def is_random(self):
        """Whether independent generation seeds represent random realizations."""

        return False

    @property
    def metadata(self):
        """Return a fresh JSON-compatible description with numeric SI values.

        Concrete media extend this dictionary with their own statistics or
        geometry. Generation seeds and grids belong to individual volumes.
        """

        return {"background_refractive_index": self.background_refractive_index}

    def add_background(self, *, refractive_index=None, medium=None, material=None):
        """Configure a composable medium's background in place.

        StructuredMedium accepts exactly one of a positive uniform ``refractive_index``
        or a RandomMedium supplied as ``medium``. Statistical media describe
        homogeneous distributions; use StructuredMedium to compose material.
        """

        raise TypeError("Use StructuredMedium to compose backgrounds and structures.")

    def add_structures(self, *structures):
        """Append positional material shapes to a composable medium in place.

        StructuredMedium supports Layer, Sphere, Ellipsoid, Box, and Cylinder.
        Later structures replace earlier material wherever their masks overlap.
        """

        raise TypeError("Use StructuredMedium to compose backgrounds and structures.")

    @abstractmethod
    def to_volume(
        self,
        *,
        grid: Grid | None = None,
        shape: tuple[int, int, int] | None = None,
        spacing: Quantity | None = None,
        seed: int = 0,
    ) -> "Volume":
        """Return a finite voxel sample; concrete classes define generation."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class RandomMedium(Medium):
    r"""Define a Gaussian random refractive index field by its isotropic spatial spectrum.

    Parameters
    ----------
    background_refractive_index : float
        Positive uniform background refractive index (required).
    refractive_index_std : float
        Nonnegative ensemble standard deviation, not variance (required).
    correlation_length : Quantity
        Positive quantity with explicit length units (required).
    correlation : {'gaussian', 'exponential', 'matern'}
        Explicitly chosen spatial covariance family. This does not change
        the Gaussian probability distribution of the refractive index field.
    smoothness : float or Quantity
        Positive finite Matérn parameter nu; required for 'matern'.
        Other covariance families do not require this parameter. Larger values give smoother continuum fields.

    Notes
    -----
    With ell = correlation_length and x = sqrt(2*nu)*r/ell, Whittle–Matérn
    covariance is ``C_n(r) = sigma_n**2 * 2**(1-nu)/Gamma(nu) * x**nu*K_nu(x)``.
    Its three-dimensional spectrum is proportional to
    ``(1 + (q*ell)**2/(2*nu))**(-nu-3/2)``. Thus nu = 0.5 reproduces exponential
    covariance with the same ell. Gaussian covariance uses
    ``exp(-r**2/(2*ell**2))``; exponential uses ``exp(-r/ell)``.

    :func:`bornsim.random_volume` normalizes the discrete spectrum to preserve
    expected point variance, retains the DC mode, and crops a doubled synthesis
    box. Neither individual sample means nor variances are forced to match the
    ensemble statistics. Grid resolution and box size affect sampled covariance.
    This class is for numerical generation and ensembles, not analytical solve.
    """

    background_refractive_index: float
    refractive_index_std: float
    correlation_length: Quantity
    correlation: str
    smoothness: Quantity | float | None = None

    def __post_init__(self) -> None:
        validate_units(
            self.correlation_length,
            unit="meter",
            name="correlation_length",
            scalar=True,
        )

        if not np.isfinite(self.correlation_length) or self.correlation_length <= 0:
            raise ValueError("correlation_length must be finite and positive.")

        for name in ("background_refractive_index", "refractive_index_std"):
            value = _refractive_index_values(
                value=getattr(self, name),
                name=name,
                scalar=True,
            )

            if not np.isfinite(value) or value < 0 or (name == "background_refractive_index" and value == 0):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if name == 'refractive_index_std' else 'positive'}."
                )

        if self.correlation not in ("gaussian", "exponential", "matern"):
            raise ValueError("correlation must be gaussian, exponential, or matern.")

        if self.smoothness is None:
            if self.correlation == "matern":
                raise ValueError("smoothness must be explicitly supplied for matern correlation.")
        else:
            smoothness = _dimensionless(
                value=self.smoothness,
                name="smoothness",
                scalar=True,
            )

            if not np.isfinite(smoothness) or smoothness <= 0:
                raise ValueError("smoothness must be finite and positive.")

            object.__setattr__(self, "smoothness", smoothness)

    @property
    def is_random(self):
        """Statistical media support independent seeded realizations."""

        return True

    @property
    def metadata(self):
        """Return independent SI statistics for result provenance.

        Lengths use the ``_m`` suffix. Smoothness is retained for all covariance
        families, including inherited analytical media, preserving the saved
        result metadata convention. This describes ensemble statistics rather
        than a particular seed or a sample's measured mean and variance.
        """

        return {
            **super().metadata,
            "refractive_index_std": self.refractive_index_std,
            "correlation_length_m": float(self.correlation_length.to("meter").magnitude),
            "correlation": self.correlation,
            "smoothness": self.smoothness,
        }

    def spectral_weight(self, *, q_squared: Quantity) -> np.ndarray:
        """Return dimensionless spectrum shape at squared wavenumbers in m^-2.

        The zero-wavenumber weight is one. This is not the normalized continuum
        power spectral density; the synthesis code supplies discrete variance
        normalization. Inputs must be finite and nonnegative.
        """

        validate_units(
            q_squared,
            unit="1 / meter**2",
            name="q_squared",
        )

        if np.any(~np.isfinite(q_squared)) or np.any(q_squared < 0):
            raise ValueError("q_squared must be finite and nonnegative.")

        scaled = (q_squared * self.correlation_length**2).to("dimensionless").magnitude

        if self.correlation == "gaussian":
            return np.exp(-scaled / 2)

        if self.correlation == "exponential":
            return (1 + scaled) ** -2

        assert self.smoothness is not None

        # log1p avoids loss of precision for small q and large smoothness.
        return np.exp(-(self.smoothness + 1.5) * np.log1p(scaled / (2 * self.smoothness)))

    def to_volume(
        self,
        *,
        grid: Grid | None = None,
        shape: tuple[int, int, int] | None = None,
        spacing: Quantity | None = None,
        seed: int = 0,
    ) -> "Volume":
        """Generate a seeded Gaussian random refractive index field with a chosen covariance.

        Parameters
        ----------
        grid : Grid, optional
            Explicit shared spatial configuration. Cannot be
            combined with the legacy shape and spacing keywords.
        shape : tuple of int, optional
            Three sample dimensions, each from 2 to 32. Required with spacing when grid is omitted.
        spacing : Quantity, optional
            Positive, finite cubic voxel spacing; length values require explicit units.
            Must be supplied with shape when grid is omitted.
        seed : int, optional
            Reproducible random seed, from 0 to 2**32 - 1. Default is 0.

        Returns
        -------
        volume : Volume
            Cropped fluctuation field with unit-bearing spacing and dimensionless background refractive index.

        Raises
        ------
        ValueError
            If shape, spacing, units, or seed are invalid, or the generated
            linearized relative permittivity is nonpositive.

        Notes
        -----
        The probability distribution is Gaussian for every spatial covariance
        choice. Spectral synthesis uses a periodic box doubled along every axis,
        then crops its first half. The discrete spectrum approximates the continuum
        covariance and normalizes the expected point variance. The DC component is
        retained; individual samples are never recentered or rescaled. Resolution
        and finite synthesis-box size can alter the sampled covariance.

        Examples
        --------
        >>> from bornsim import RandomMedium
        ...
        >>> from bornsim.units import ureg
        ...
        >>> medium = RandomMedium(
        ...     correlation="gaussian",
        ...     background_refractive_index=1.33,
        ...     refractive_index_std=0.01,
        ...     correlation_length=100e-9 * ureg.meter,
        ... )

        ...
        >>> volume = medium.to_volume(
        ...     shape=(4, 4, 4),
        ...     seed=42,
        ...     spacing=50e-9 * ureg.meter,
        ... )
        >>> volume.delta_refractive_index.shape
        (4, 4, 4)
        """

        from .volume import Volume

        grid = Grid._resolve(
            grid=grid,
            shape=shape,
            spacing=spacing,
        )

        # FFT frequencies require numeric spacing in a chosen length unit.
        sample_grid_shape = grid.shape

        voxel_spacing_m = float(grid.spacing.to("meter").magnitude)

        seed = _integer(
            value=seed,
            name="seed",
            low=0,
            high=2**32 - 1,
        )

        synthesis_grid_shape = tuple(2 * voxel_count for voxel_count in sample_grid_shape)

        axis_wavenumbers_per_meter = [
            2 * np.pi * np.fft.fftfreq(voxel_count, voxel_spacing_m) for voxel_count in synthesis_grid_shape
        ]

        squared_wavenumbers_per_meter_squared = np.zeros(synthesis_grid_shape)

        for wavenumber_component_per_meter in np.meshgrid(*axis_wavenumbers_per_meter, indexing="ij"):
            squared_wavenumbers_per_meter_squared += wavenumber_component_per_meter**2

        normalized_spectral_weights = self.spectral_weight(
            q_squared=squared_wavenumbers_per_meter_squared * (1 / ureg.meter**2)
        )

        # Normalize expected point variance, not each sample's spatial variance.
        normalized_spectral_weights /= normalized_spectral_weights.mean()

        uncorrelated_gaussian_noise = np.random.default_rng(seed).normal(size=synthesis_grid_shape)

        synthesized_refractive_index_fluctuations = (
            np.fft.ifftn(np.fft.fftn(uncorrelated_gaussian_noise) * np.sqrt(normalized_spectral_weights)).real
            * self.refractive_index_std
        )

        sample_crop = tuple(slice(0, voxel_count) for voxel_count in sample_grid_shape)

        return Volume(
            delta_refractive_index=synthesized_refractive_index_fluctuations[sample_crop],
            grid=grid,
            background_refractive_index=self.background_refractive_index,
            medium=self,
            seed=seed,
        )


def random_volume(
    *,
    medium: RandomMedium,
    grid: Grid | None = None,
    shape: tuple[int, int, int] | None = None,
    spacing: Quantity | None = None,
    seed: int = 0,
) -> "Volume":
    """Generate a seeded Gaussian refractive index field via :meth:`RandomMedium.to_volume`.

    Medium specifies the covariance and ensemble statistics. Shape contains
    three integers from 2 to 32; spacing is a positive SI length or quantity.
    Seed lies in 0 to 2**32-1. Expected point variance is preserved without
    forcing individual sample means or variances. See the medium method for
    the spectral synthesis equations and finite-box conventions.
    """

    if not isinstance(medium, RandomMedium):
        raise TypeError("medium must be a RandomMedium.")

    return medium.to_volume(
        grid=grid,
        shape=shape,
        spacing=spacing,
        seed=seed,
    )
