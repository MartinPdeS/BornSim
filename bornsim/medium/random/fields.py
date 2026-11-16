"""Seeded random fields with explicitly chosen spatial covariance."""

from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from ..base import Medium
from ...units import NoUnit, Quantity, Scalar, validate_annotation, validate_quantity, ureg
from ...grid import Grid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...volume import Volume


@dataclass(frozen=True, kw_only=True)
class RandomFieldMedium(Medium):
    """Abstract spectral synthesis shared by covariance-specific random fields.

    Fields have Gaussian one-point statistics. Synthesis preserves expected
    variance and the DC mode, using a doubled box followed by cropping.
    Individual realizations are never recentered or rescaled.
    """

    background_refractive_index: float
    refractive_index_std: float
    correlation_length: Quantity
    extinction_coefficient: float = 0.0

    @property
    @abstractmethod
    def correlation(self) -> str:
        """Return the spatial covariance family.

        Returns
        -------
        str
            Covariance-family identifier stored in provenance.
        """

        raise NotImplementedError

    def __post_init__(self) -> None:
        validate_quantity(
            self.correlation_length,
            unit="meter",
            name="correlation_length",
            scalar=True,
        )

        if not np.isfinite(self.correlation_length) or self.correlation_length <= 0:
            raise ValueError("correlation_length must be finite and positive.")

        for name in ("background_refractive_index", "refractive_index_std", "extinction_coefficient"):
            value = getattr(self, name)

            validate_annotation(
                value,
                annotation=NoUnit[Scalar],
                name=name,
            )

            value = float(value)

            if not np.isfinite(value) or value < 0 or (name == "background_refractive_index" and value == 0):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if name == 'refractive_index_std' else 'positive'}."
                )

    @property
    def is_random(self) -> bool:
        """Whether independent seeds generate distinct realizations.

        Returns
        -------
        bool
            Always True for statistical random-field media.
        """

        return True

    @property
    def metadata(self) -> dict[str, object]:
        """Return independent SI statistics for result provenance.

        Lengths use the ``_m`` suffix. This describes ensemble statistics rather
        than a particular seed or a sample's measured mean and variance.

        Returns
        -------
        dict of str to object
            Independent metadata with physical lengths represented in metres.
        """

        return {
            **super().metadata,
            "refractive_index_std": self.refractive_index_std,
            "correlation_length_m": float(self.correlation_length.to("meter").magnitude),
            "correlation": self.correlation,
            "smoothness": getattr(self, "smoothness", None),
            **({"extinction_coefficient": self.extinction_coefficient} if self.extinction_coefficient else {}),
        }

    def spectral_weight(self, *, q_squared: Quantity) -> np.ndarray:
        """Return dimensionless spectrum shape at squared wavenumbers in m^-2.

        The zero-wavenumber weight is one. This is not the normalized continuum
        power spectral density; the synthesis code supplies discrete variance
        normalization. Inputs must be finite and nonnegative.

        Parameters
        ----------
        q_squared : Quantity
            Nonnegative squared wavenumbers with explicit inverse-square-length
            units.

        Returns
        -------
        numpy.ndarray
            Dimensionless spectral weights with the input shape.

        Raises
        ------
        ValueError
            If units or values are invalid.
        """

        validate_quantity(
            q_squared,
            unit="1 / meter**2",
            name="q_squared",
        )

        if np.any(~np.isfinite(q_squared)) or np.any(q_squared < 0):
            raise ValueError("q_squared must be finite and nonnegative.")

        scaled = (q_squared * self.correlation_length**2).to("dimensionless").magnitude

        return self._spectral_weight(scaled_squared_wavenumber=scaled)

    @abstractmethod
    def _spectral_weight(self, *, scaled_squared_wavenumber: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_volume(self, *, grid: Grid, seed: int = 0) -> "Volume":
        """Generate a seeded random refractive-index Volume on an explicit Grid.

        The field has a Gaussian one-point probability distribution for every spatial
        covariance model. Spectral synthesis uses a doubled periodic box and crops
        its first half. Expected point variance is normalized; individual samples
        are never recentered or rescaled. Grid resolution and synthesis-box size
        can alter sampled covariance. The seed defaults to zero and remains explicit
        in the generated volume's provenance.

        Parameters
        ----------
        grid : Grid
            Explicit cubic voxel grid.
        seed : int, default=0
            Nonnegative 32-bit seed for the random realization.

        Returns
        -------
        Volume
            Sampled refractive-index contrast and generation metadata.

        Raises
        ------
        ValueError
            If the seed or sampling configuration is invalid.
        """

        from ...volume import Volume

        grid = Grid._resolve(grid=grid)

        # FFT frequencies require numeric spacing in a chosen length unit.
        sample_grid_shape = grid.shape

        voxel_spacing_m = float(grid.spacing.to("meter").magnitude)

        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer between 0 and 4294967295.")

        seed = int(seed)

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
            extinction_coefficient=self.extinction_coefficient,
            grid=grid,
            background_refractive_index=self.background_refractive_index,
            medium=self,
            seed=seed,
        )


@dataclass(frozen=True, kw_only=True)
class GaussianMedium(RandomFieldMedium):
    """Gaussian spatial covariance: C(r) = refractive_index_std² exp(-r²/(2 ell²))."""

    correlation = "gaussian"

    def _spectral_weight(self, *, scaled_squared_wavenumber: np.ndarray) -> np.ndarray:
        return np.exp(-scaled_squared_wavenumber / 2)


@dataclass(frozen=True, kw_only=True)
class ExponentialMedium(RandomFieldMedium):
    """Exponential spatial covariance: C(r) = refractive_index_std² exp(-r/ell)."""

    correlation = "exponential"

    def _spectral_weight(self, *, scaled_squared_wavenumber: np.ndarray) -> np.ndarray:
        return (1 + scaled_squared_wavenumber) ** -2


@dataclass(frozen=True, kw_only=True)
class WhittleMaternMedium(RandomFieldMedium):
    """Whittle–Matérn covariance with explicitly chosen positive smoothness.

    The spectrum is proportional to (1 + (q ell)²/(2 smoothness))**(-smoothness-3/2).
    Smoothness 0.5 reproduces ExponentialMedium with the same correlation length.
    """

    smoothness: float
    correlation = "matern"

    def _spectral_weight(self, *, scaled_squared_wavenumber: np.ndarray) -> np.ndarray:
        # log1p preserves precision for small wavenumbers and large smoothness.
        return np.exp(-(self.smoothness + 1.5) * np.log1p(scaled_squared_wavenumber / (2 * self.smoothness)))

    def __post_init__(self) -> None:
        super().__post_init__()

        if isinstance(self.smoothness, Quantity) or np.ndim(self.smoothness) != 0:
            raise ValueError("smoothness must be a plain scalar dimensionless number.")

        if not np.isfinite(self.smoothness) or self.smoothness <= 0:
            raise ValueError("smoothness must be finite and positive.")
