"""Full directional output sampling and independent solid-angle quadrature."""

from dataclasses import dataclass
from typing import cast
from numpy.typing import ArrayLike
import numpy as np
from .directions import Directions
from .grid import Grid
from .results import BornResult
from .units import Quantity, _integer, validate_units, ureg


@dataclass(frozen=True, kw_only=True, init=False)
class AngularSampling:
    """Configure directional plots and their solid-angle normalization.

    Parameters
    ----------
    angles : Quantity, optional
        One-dimensional output polar angles theta, from 0 to pi, with 1 to
        181 entries. Angular values require explicit units. Cannot be combined
        with start, end or n_points. Omit to generate an evenly spaced range.
    start, end : Quantity, optional
        Inclusive range endpoints between 0 and pi. Angular values require explicit units.
        Defaults are 0 and pi; descending ranges are supported.
    n_points : int, optional
        Number of evenly spaced output angles, from 1 to 181. Default is 121.
        A single point samples start.
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

    angles: Quantity
    polar_samples: int = 32
    azimuth_samples: int = 8

    def __init__(
        self,
        *,
        angles: Quantity | None = None,
        start: Quantity | None = None,
        end: Quantity | None = None,
        n_points: int | None = None,
        polar_samples: int = 32,
        azimuth_samples: int = 8,
    ) -> None:
        """Initialize an angular output grid and its integration quadrature.

        Parameters
        ----------
        angles : Quantity, optional
            Explicit one-dimensional polar angles between 0 and pi. Requires
            explicit angular units and cannot be combined with range settings.
        start, end : Quantity, optional
            Inclusive endpoints for an evenly spaced polar-angle range.
            Defaults are 0 and pi radians.
        n_points : int, optional
            Number of output polar angles, from 1 to 181; defaults to 121.
        polar_samples : int, optional
            Number of Gauss-Legendre nodes, from 16 to 128; defaults to 32.
        azimuth_samples : int, optional
            Number of azimuth directions, from 4 to 32; defaults to 8.

        Returns
        -------
        None
            Initializes the immutable sampling configuration.

        Raises
        ------
        ValueError
            If the angle quantities, angle range, output count, or quadrature
            sample counts are invalid.
        """

        if angles is not None:
            if any(value is not None for value in (start, end, n_points)):
                raise ValueError("Supply angles or start/end/n_points, not both.")
        else:
            start = 0 * ureg.radian if start is None else start

            validate_units(
                start,
                unit="radian",
                name="start",
                scalar=True,
            )

            end = np.pi * ureg.radian if end is None else end

            validate_units(
                end,
                unit="radian",
                name="end",
                scalar=True,
            )

            invalid_endpoints = not (
                np.isfinite(start)
                and np.isfinite(end)
                and 0 <= start <= np.pi * ureg.radian
                and 0 <= end <= np.pi * ureg.radian
            )

            if invalid_endpoints:
                raise ValueError("start and end must be finite angles between 0 and pi.")

            count = _integer(value=121 if n_points is None else n_points, name="n_points", low=1, high=181)

            angles = cast(Quantity, np.linspace(start, end, count))

        validate_units(
            angles,
            unit="radian",
            name="angles",
        )

        angles = angles.copy()

        invalid_angles = (
            angles.ndim != 1
            or not 1 <= angles.size <= 181
            or np.any(~np.isfinite(angles))
            or np.any((angles < 0) | (angles > np.pi * ureg.radian))
        )

        if invalid_angles:
            raise ValueError("angles must contain 1–181 finite angles between 0 and pi.")

        angles.magnitude.setflags(write=False)

        object.__setattr__(self, "angles", angles)

        for name, value, low, high in (
            ("polar_samples", polar_samples, 16, 128),
            ("azimuth_samples", azimuth_samples, 4, 32),
        ):
            object.__setattr__(self, name, _integer(value=value, name=name, low=low, high=high))

    def __repr__(self) -> str:
        """Return a compact summary of the angle range and quadrature sizes.

        Returns
        -------
        str
            The number and range of output angles, in degrees, and the
            integration sample counts.
        """

        angles = self.angles.to("degree").magnitude

        low, high = angles.min(), angles.max()

        return (
            f"AngularSampling(angles={len(angles)} in [{low:g}, {high:g}] deg, "
            f"polar_samples={self.polar_samples}, azimuth_samples={self.azimuth_samples})"
        )

    @property
    def azimuths(self) -> Quantity:
        """Return uniform output and integration azimuths.

        Returns
        -------
        Quantity
            Azimuth directions spanning one full turn, in radians.
        """

        return np.arange(self.azimuth_samples) * 2 * np.pi / self.azimuth_samples * ureg.radian

    @property
    def directions(self) -> Directions:
        """Return unit directions for output and integration grids.

        Returns
        -------
        Directions
            Directions with output polar angles first, followed by the
            Gauss-Legendre integration nodes, for each azimuth.
        """

        cosine, _ = np.polynomial.legendre.leggauss(self.polar_samples)

        theta = np.concatenate([self.angles.to("radian").magnitude, np.arccos(cosine)]) * ureg.radian

        return Directions.from_angles(
            polar_angles=theta[:, None],
            azimuth_angles=self.azimuths[None, :],
        )

    @property
    def metadata(self) -> dict[str, list[int | float] | int]:
        """Return fresh SI sampling settings for reproducible result archives.

        Returns
        -------
        dict[str, list[int | float] | int]
            Output angles in radians and the polar and azimuthal sample counts.
        """

        return {
            "angles_rad": self.angles.to("radian").magnitude.tolist(),
            "polar_samples": self.polar_samples,
            "azimuth_samples": self.azimuth_samples,
        }

    def check_work(self, *, grid: Grid, order: int, realizations: int = 1) -> None:
        """Reject synchronous workloads above 100 million voxel-direction-orders.

        Parameters
        ----------
        grid : Grid
            Spatial grid used to estimate the number of voxel-direction pairs.
        order : int
            Highest cumulative Born order in the requested calculation.
        realizations : int, optional
            Number of independent volumes; defaults to 1.

        Returns
        -------
        None
            Returns normally when the estimated workload is within the limit.

        Raises
        ------
        ValueError
            If the estimate exceeds 100 million voxel-direction-orders.
        """

        directions = (len(self.angles) + self.polar_samples) * self.azimuth_samples

        if realizations * np.prod(grid.shape) * directions * order > 100_000_000:
            raise ValueError("Requested scattering is too large; reduce grid, angles, order, or realizations.")

    def summarize(
        self, *, result: BornResult, incident_direction: ArrayLike = (0.0, 0.0, 1.0)
    ) -> dict[str, np.ndarray]:
        """Reduce numeric Born data while retaining all directional intensities.

        Parameters
        ----------
        incident_direction : array_like, optional
            Cartesian unit beam direction used for anisotropy and reduced
            scattering moments. Defaults to positive z; output angles stay
            in laboratory coordinates.
        result : BornResult
            Numeric Born amplitudes, cumulative differential data, and
            isolated-term differential data for one volume.

        Returns
        -------
        dict[str, numpy.ndarray]
            Numeric arrays with these keys: ``mean`` and ``terms`` have shape
            (order, output angle); ``directional_terms`` and
            ``directional_differential`` have shape (order, output angle,
            azimuth); ``directional_amplitudes`` adds incident polarization
            and vector-component axes of sizes one or two and three; ``integrals``
            has shape (order, 3), for total, first-moment, and reduced
            coefficients.

        Notes
        -----
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

        _, weights = np.polynomial.legendre.leggauss(self.polar_samples)

        quadrature = averaged[:, count:]

        mu = 2 * np.pi * np.sum(quadrature * weights, axis=-1)

        # Moments use the physical scattering cosine, not the laboratory z axis.
        direction = Directions(vectors=np.asarray(incident_direction)[None, :]).vectors[0]

        integration_directions = self.directions.vectors.reshape(-1, self.azimuth_samples, 3)[count:]

        scattering_cosine = np.einsum("ijc,c->ij", integration_directions, direction)

        integrated_differential = directional[:, count:]

        moment = (
            2
            * np.pi
            * np.sum(
                np.mean(integrated_differential * scattering_cosine, axis=-1) * weights,
                axis=-1,
            )
        )

        reduced = (
            2
            * np.pi
            * np.sum(
                np.mean(integrated_differential * (1 - scattering_cosine), axis=-1) * weights,
                axis=-1,
            )
        )

        return {
            "mean": averaged[:, :count],
            "terms": terms[:, :count],
            "directional_terms": directional_terms[:, :count],
            "directional_differential": directional[:, :count],
            "directional_amplitudes": result.amplitudes.reshape(*shape, result.amplitudes.shape[-2], 3)[:, :count],
            "integrals": np.stack([mu, moment, reduced], axis=-1),
        }

    @classmethod
    def _resolve(cls, *, sampling: "AngularSampling | None" = None) -> "AngularSampling":
        """Return the supplied configuration or construct the default.

        Parameters
        ----------
        sampling : AngularSampling, optional
            Configuration to validate and return. If omitted, construct the
            default angular sampling configuration.

        Returns
        -------
        AngularSampling
            The validated supplied configuration or a default instance.

        Raises
        ------
        TypeError
            If ``sampling`` is not an AngularSampling instance.
        """

        if sampling is None:
            return cls()

        if not isinstance(sampling, cls):
            raise TypeError("sampling must be an AngularSampling.")

        return sampling
