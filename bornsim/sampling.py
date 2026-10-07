"""Full directional output sampling and independent solid-angle quadrature."""

from dataclasses import dataclass, field
import numpy as np
import warnings
from ._validation import _integer
from .units import Quantity, _si


@dataclass(frozen=True, kw_only=True)
class AngularSampling:
    """Configure directional plots and their solid-angle normalization.

    Parameters
    ----------
    angles : array_like or Quantity, optional
        Output polar angles theta, from 0 to pi, with 1 to 181 entries.
        Bare values mean radians. Default is 121 evenly spaced angles.
    polar_samples : int, optional
        Gauss-Legendre integration nodes in cos(theta), from 16 to 128.
        Default is 32. These are independent of the output angles.
    azimuth_samples : int, optional
        Uniform azimuth directions in [0, 2*pi), from 4 to 32. Default is 8.
        Used for both directional output and solid-angle integration.

    Notes
    -----
    Intensities are retained at every output (theta, phi). One-dimensional
    curves select sampled meridians; azimuth averages require an explicit
    azimuth_average() call. 3D phase surfaces use directional data.
    Integration uses ``dOmega = dphi * d(cos(theta))``. Each Born order's
    phase density is its intensity divided by its integrated coefficient.
    Refine both quadratures independently of output plot resolution.
    """

    angles: Quantity | np.ndarray = field(default_factory=lambda: np.linspace(0, np.pi, 121))
    polar_samples: int = 32
    azimuth_samples: int = 8

    def __post_init__(self):
        angles = _si(value=self.angles, unit="radian", name="angles").copy()
        invalid_angles = (
            angles.ndim != 1
            or not 1 <= angles.size <= 181
            or np.any(~np.isfinite(angles))
            or np.any((angles < 0) | (angles > np.pi))
        )
        if invalid_angles:
            raise ValueError("angles must contain 1–181 finite angles between 0 and pi.")
        angles.setflags(write=False)
        object.__setattr__(self, "angles", angles)
        for name, low, high in (("polar_samples", 16, 128), ("azimuth_samples", 4, 32)):
            object.__setattr__(self, name, _integer(value=getattr(self, name), name=name, low=low, high=high))

    def __repr__(self):
        angles = np.asarray(self.angles)
        low, high = np.rad2deg([angles.min(), angles.max()])
        return (
            f"AngularSampling(angles={len(angles)} in [{low:g}, {high:g}] deg, "
            f"polar_samples={self.polar_samples}, azimuth_samples={self.azimuth_samples})"
        )

    @property
    def azimuths(self):
        """Uniform output and integration azimuths in radians."""
        return np.arange(self.azimuth_samples) * 2 * np.pi / self.azimuth_samples

    @property
    def directions(self):
        """Unit directions: output grid first, then the integration grid."""
        cosine, _ = np.polynomial.legendre.leggauss(self.polar_samples)
        theta = np.concatenate([self.angles, np.arccos(cosine)])
        tt, pp = np.meshgrid(theta, self.azimuths, indexing="ij")
        return np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1).reshape(-1, 3)

    @property
    def metadata(self):
        """Fresh SI sampling settings for reproducible result archives."""
        return {
            "angles_rad": np.asarray(self.angles).tolist(),
            "polar_samples": self.polar_samples,
            "azimuth_samples": self.azimuth_samples,
        }

    def check_work(self, *, grid, order, realizations=1):
        """Reject synchronous workloads above 100 million voxel-direction-orders."""
        directions = (len(self.angles) + self.polar_samples) * self.azimuth_samples
        if realizations * np.prod(grid.shape) * directions * order > 100_000_000:
            raise ValueError("Requested scattering is too large; reduce grid, angles, order, or realizations.")

    def summarize(self, *, result):
        """Reduce numeric Born data while retaining all directional intensities.

        Output amplitudes retain (order, theta, phi, incident polarization,
        vector component). Integrated coefficients are finite-sample cross
        sections divided by voxel-box volume, not intrinsic material values.
        """
        count = len(self.angles)
        order = len(result.differential)
        shape = (order, count + self.polar_samples, self.azimuth_samples)
        directional = result.differential.reshape(shape)
        averaged = directional.mean(axis=-1)
        directional_terms = result.term_differential.reshape(shape)
        terms = directional_terms.mean(axis=-1)
        cosine, weights = np.polynomial.legendre.leggauss(self.polar_samples)
        quadrature = averaged[:, count:]
        mu = 2 * np.pi * np.sum(quadrature * weights, axis=-1)
        moment = 2 * np.pi * np.sum(quadrature * weights * cosine, axis=-1)
        reduced = 2 * np.pi * np.sum(quadrature * weights * (1 - cosine), axis=-1)
        return {
            "mean": averaged[:, :count],
            "terms": terms[:, :count],
            "directional_terms": directional_terms[:, :count],
            "directional_differential": directional[:, :count],
            "directional_amplitudes": result.amplitudes.reshape(*shape, 2, 3)[:, :count],
            "integrals": np.stack([mu, moment, reduced], axis=-1),
        }

    @classmethod
    def _resolve(cls, *, sampling=None, angles=None, polar_samples=None, azimuth_samples=None):
        if sampling is not None:
            if not isinstance(sampling, cls):
                raise TypeError("sampling must be an AngularSampling.")
            if any(value is not None for value in (angles, polar_samples, azimuth_samples)):
                raise ValueError("Supply sampling or individual angular settings, not both.")
            return sampling
        if any(value is not None for value in (angles, polar_samples, azimuth_samples)):
            warnings.warn(
                "Use AngularSampling instead of individual angular keywords.", DeprecationWarning, stacklevel=3
            )
        return cls(
            angles=np.linspace(0, np.pi, 121) if angles is None else angles,
            polar_samples=32 if polar_samples is None else polar_samples,
            azimuth_samples=8 if azimuth_samples is None else azimuth_samples,
        )
