"""Numerical random-medium statistics, independent of analytical solutions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from .units import Quantity, _si
from ._validation import _integer
from .grid import Grid


class Medium(ABC):
    """Abstract interface for media that can be sampled into a finite volume.

    Instantiate :class:`RandomMedium` or :class:`bornsim.StructuredMedium`,
    not this base class. Concrete media specify a uniform ``background_index``
    and implement :meth:`to_volume` using cubic voxels, SI spacing, and centred
    coordinates. The resulting field stores ``delta_index = n(r) - n0``;
    propagation retains ``epsilon_r = n0**2 + 2*n0*delta_index``.
    """

    background_index: Quantity | float

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
        return {"background_index": self.background_index}

    def add_background(self, *, index=None, medium=None, material=None):
        """Configure a composable medium's background in place.

        StructuredMedium accepts exactly one of a positive uniform ``index``
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
    def to_volume(self, *, grid=None, shape=None, spacing=None, seed=0):
        """Return a finite voxel sample; concrete classes define generation."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class RandomMedium(Medium):
    r"""Define a Gaussian random index field by its isotropic spatial spectrum.

    Parameters
    ----------
    background_index : float or Quantity
        Positive uniform background index (default 1.33).
    index_std : float or Quantity
        Nonnegative ensemble standard deviation, not variance (default 0.01).
    correlation_length : float or Quantity
        Positive length in metres, or a length quantity (default 100 nm).
    correlation : {'gaussian', 'exponential', 'matern'}
        Spatial covariance family; default is 'matern'. This does not change
        the Gaussian probability distribution of the index field.
    smoothness : float or Quantity
        Positive finite Matérn parameter nu (default 1.5); used only for
        'matern'. Larger values give smoother continuum fields.

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

    background_index: Quantity | float = 1.33
    index_std: Quantity | float = 0.01
    correlation_length: Quantity | float = 100e-9
    correlation: str = "matern"
    smoothness: Quantity | float = 1.5

    def __post_init__(self):
        for name in ("background_index", "index_std", "correlation_length"):
            unit = "meter" if name == "correlation_length" else "dimensionless"
            value = _si(
                value=getattr(self, name),
                unit=unit,
                name=name,
                scalar=True,
            )
            if not np.isfinite(value) or value < 0 or (name != "index_std" and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'index_std' else 'positive'}.")
            object.__setattr__(self, name, value)
        if self.correlation not in ("gaussian", "exponential", "matern"):
            raise ValueError("correlation must be gaussian, exponential, or matern.")
        nu = _si(
            value=self.smoothness,
            unit="dimensionless",
            name="smoothness",
            scalar=True,
        )
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError("smoothness must be finite and positive.")
        object.__setattr__(self, "smoothness", nu)

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
            "index_std": self.index_std,
            "correlation_length_m": self.correlation_length,
            "correlation": self.correlation,
            "smoothness": self.smoothness,
        }

    def spectral_weight(self, *, q_squared):
        """Return dimensionless spectrum shape at squared wavenumbers in m^-2.

        The zero-wavenumber weight is one. This is not the normalized continuum
        power spectral density; the synthesis code supplies discrete variance
        normalization. Inputs must be finite and nonnegative.
        """
        q_squared = np.asarray(q_squared, dtype=float)
        if np.any(~np.isfinite(q_squared)) or np.any(q_squared < 0):
            raise ValueError("q_squared must be finite and nonnegative.")
        scaled = q_squared * self.correlation_length**2
        if self.correlation == "gaussian":
            return np.exp(-scaled / 2)
        if self.correlation == "exponential":
            return (1 + scaled) ** -2
        # log1p avoids loss of precision for small q and large smoothness.
        return np.exp(-(self.smoothness + 1.5) * np.log1p(scaled / (2 * self.smoothness)))

    def to_volume(self, *, grid=None, shape=None, spacing=None, seed=0):
        """Generate a seeded Gaussian random index field with a chosen covariance.

        Parameters
        ----------
        grid : Grid, optional
            Shared spatial configuration. Defaults to Grid(). Cannot be
            combined with the legacy shape and spacing keywords.
        shape : tuple of int, optional
            Three sample dimensions, each from 2 to 32. Default is (12, 12, 12).
        spacing : float or Quantity, optional
            Positive, finite cubic voxel spacing; bare values mean metres.
            Default is 50 nm.
        seed : int, optional
            Reproducible random seed, from 0 to 2**32 - 1. Default is 0.

        Returns
        -------
        volume : Volume
            Cropped fluctuation field with numeric SI spacing and background index.

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
        >>> medium = RandomMedium(correlation="gaussian")

        >>> volume = medium.to_volume(
        ...     shape=(4, 4, 4),
        ...     seed=42,
        ... )
        >>> volume.delta_index.shape
        (4, 4, 4)
        """
        from .volume import Volume

        grid = Grid._resolve(
            grid=grid,
            shape=shape,
            spacing=spacing,
        )
        shape, spacing = grid.shape, grid.spacing
        seed = _integer(
            value=seed,
            name="seed",
            low=0,
            high=2**32 - 1,
        )
        extended = tuple(2 * n for n in shape)
        axes = [2 * np.pi * np.fft.fftfreq(n, spacing) for n in extended]
        q2 = np.zeros(extended)
        for component in np.meshgrid(*axes, indexing="ij"):
            q2 += component**2
        spectrum = self.spectral_weight(q_squared=q2)
        # Normalize expected point variance, not each sample's spatial variance.
        spectrum /= spectrum.mean()
        white = np.random.default_rng(seed).normal(size=extended)
        field = np.fft.ifftn(np.fft.fftn(white) * np.sqrt(spectrum)).real * self.index_std
        crop = tuple(slice(0, n) for n in shape)
        return Volume(
            delta_index=field[crop],
            grid=grid,
            background_index=self.background_index,
            medium=self,
            seed=seed,
        )


def random_volume(*, medium: RandomMedium, grid=None, shape=None, spacing=None, seed=0):
    """Generate a seeded Gaussian index field via :meth:`RandomMedium.to_volume`.

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
