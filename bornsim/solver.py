"""Solver configuration and analytical, volume, and ensemble calculations."""

from dataclasses import dataclass, field
import hashlib
import numpy as np
from ._version import __version__
from .media import Medium
from .model import AnalyticalMedium, angular_scattering, optical_properties
from .results import Result
from ._validation import _integer
from .volume import Volume
from .series import BornSeries
from .ensemble import ensemble_scattering
from .grid import Grid
from .directions import Directions
from .sampling import AngularSampling
from .ensemble_sampling import EnsembleSampling
from .source import Source
from .units import Quantity, validate_units, ureg


def _provenance(*, order, **settings):
    return {
        "bornsim_version": __version__,
        "numpy_version": np.__version__,
        "order": int(order),
        "dielectric_contrast": "2 * background_refractive_index * delta_refractive_index",
        "green_self_cell": "equal-volume sphere with longitudinal contact term",
        **settings,
    }


@dataclass(frozen=True, kw_only=True)
class Solver:
    """Configure analytical and finite-volume Born scattering calculations.

    Parameters
    ----------
    source : Source
        Incident vacuum wavelength and unpolarized illumination.
    sampling : AngularSampling, optional
        Shared output angles and solid-angle quadrature. Defaults to
        AngularSampling(). Used for full-volume solves and ensembles.
    order : int, optional
        Highest cumulative numerical Born order, from 1 to 12. Default is 3.
        Analytical calculations always use first order.
    quadrature_order : int, optional
        Gauss-Legendre order for analytical integrated coefficients. Must be
        at least 16; default is 256. Numerical integration uses sampling's
        polar_samples and azimuth_samples instead.

    Attributes
    ----------
    source : Source
        Incident source shared by all calculations.
    order : int
        Highest numerical Born order.
    quadrature_order : int
        Analytical integration order.

    Raises
    ------
    TypeError
        If ``source`` is not a Source.
    ValueError
        If either order parameter is outside its supported integer range.

    Notes
    -----
    Numerical calculations retain the linearized dielectric contrast
    ``2 * n0 * delta_n``. Green-tensor self interactions use an equal-volume
    sphere including the longitudinal contact term. Increasing Born order
    does not restore the omitted quadratic constitutive term. Decreasing
    successive terms do not establish universal Born convergence.

    Examples
    --------
    >>> from bornsim import AnalyticalMedium, Solver, Source
    ...
    >>> from bornsim.units import ureg
    ...
    >>> solver = Solver(
    ...     source=Source(
    ...         wavelength=633e-9 * ureg.meter,
    ...     ),
    ...     order=3,
    ... )
    >>> result = solver.solve(
    ...     target=AnalyticalMedium(
    ...         background_refractive_index=1.33,
    ...         refractive_index_std=0.01,
    ...         correlation_length=100e-9 * ureg.meter,
    ...         correlation="gaussian",
    ...     )
    ... )
    >>> result.differential.shape
    (1, 121, 8)
    """

    source: Source
    sampling: AngularSampling = field(default_factory=AngularSampling)
    order: int = 3
    quadrature_order: int = 256

    def __post_init__(self):
        if not isinstance(self.sampling, AngularSampling):
            raise TypeError("sampling must be an AngularSampling.")

        if not isinstance(self.source, Source):
            raise TypeError("source must be a Source.")

        _integer(
            value=self.order,
            name="order",
            low=1,
            high=12,
        )

        invalid_quadrature_order = (
            isinstance(self.quadrature_order, bool)
            or not isinstance(self.quadrature_order, int)
            or self.quadrature_order < 16
        )

        if invalid_quadrature_order:
            raise ValueError("quadrature_order must be an integer of at least 16.")

    def __repr__(self):
        wavelength = getattr(self.source, "wavelength").to("nanometer")

        return f"Solver(wavelength={wavelength.magnitude:g} nm, order={self.order}, sampling={self.sampling!r})"

    def solve(
        self,
        *,
        target: Volume | AnalyticalMedium,
        sampling: AngularSampling | None = None,
        angles: Quantity | None = None,
        directions: Directions | None = None,
    ) -> Result:
        """Compute a full angular scattering distribution with normalization.

        Parameters
        ----------
        target : Volume or AnalyticalMedium
            Fixed numerical sample or analytical first-order statistics.
        sampling : AngularSampling, optional
            Override the solver's shared angular settings for this call.
        angles : array_like or Quantity, optional
            Deprecated analytical output-angle keyword. Use AngularSampling.
            Numerical angular cuts use solve_cut instead.
        directions : Directions, optional
            Rejected here; arbitrary numerical observations use solve_cut.

        Returns
        -------
        result : Result
            Directional intensities and normalized phase densities, with
            integrated coefficients. Numerical amplitudes are retained.
            Coefficients for finite samples are cross sections divided by
            voxel-box volume, not intrinsic infinite-medium properties.
        """

        if not isinstance(target, (AnalyticalMedium, Volume)):
            raise TypeError("target must be an AnalyticalMedium or Volume.")

        if directions is not None or (isinstance(target, Volume) and angles is not None):
            raise ValueError("Use solve_cut for explicit angles or directions; solve always computes full scattering.")

        if sampling is not None and angles is not None:
            raise ValueError("Supply sampling or analytical angles, not both.")

        sampling = self.sampling if sampling is None else AngularSampling._resolve(sampling=sampling)

        if isinstance(target, Volume):
            return self._solve_volume(volume=target, sampling=sampling)

        if angles is not None:
            validate_units(
                angles,
                unit="radian",
                name="angles",
            )

        theta = sampling.angles.copy() if angles is None else angles.copy()

        invalid_angles = (
            theta.ndim != 1
            or theta.size == 0
            or np.any(~np.isfinite(theta))
            or np.any((theta < 0) | (theta > np.pi * ureg.radian))
        )

        if invalid_angles:
            raise ValueError("angles must be a nonempty 1D array of finite angles between 0 and pi.")

        if angles is not None:
            import warnings

            warnings.warn("Use AngularSampling for analytical output angles.", DeprecationWarning, stacklevel=2)

        wavelength = self.source.wavelength

        differential = angular_scattering(
            medium=target,
            wavelength=wavelength,
            theta=theta,
        )

        coefficients = optical_properties(
            medium=target,
            wavelength=wavelength,
            quadrature_order=self.quadrature_order,
        )

        directional = np.broadcast_to(differential[None, :, None], (1, len(theta), sampling.azimuth_samples)).copy()

        return Result(
            source=self.source,
            kind="analytical",
            angles=theta if theta is not None else None,
            azimuths=sampling.azimuths,
            differential=directional,
            mu_s=coefficients["mu_s"].reshape(1),
            g=np.array([np.nan if coefficients["g"] is None else coefficients["g"]]),
            mu_s_prime=coefficients["mu_s_prime"].reshape(1),
            provenance=_provenance(
                order=1,
                medium=target.metadata,
                quadrature_order=self.quadrature_order,
                sampling={**sampling.metadata, "angles_rad": theta.to("radian").magnitude.tolist()},
                coefficient_scope="infinite-medium",
            ),
        )

    def solve_cut(
        self,
        *,
        target: Volume,
        angles: Quantity | None = None,
        directions: Directions | None = None,
    ) -> Result:
        """Compute an unnormalized angular cut through a fixed Volume.

        Supply polar angles on the x-z meridian or a Directions configuration,
        but not both. Explicit angular units are required. Default angles span 0 to pi
        with 121 samples. Cut amplitudes and intensities retain every requested
        direction. A cut alone cannot determine a solid-angle integral;
        integrated coefficients and normalized phase functions are unavailable.
        """

        if not isinstance(target, Volume):
            raise TypeError("target must be a Volume.")

        if angles is not None and directions is not None:
            raise ValueError("Supply angles or directions, not both.")

        if angles is not None:
            validate_units(
                angles,
                unit="radian",
                name="angles",
            )

        theta = None

        if directions is None:
            theta = np.linspace(0, np.pi, 121) * ureg.radian if angles is None else angles.copy()

            invalid_angles = (
                theta.ndim != 1
                or theta.size == 0
                or np.any(~np.isfinite(theta))
                or np.any((theta < 0) | (theta > np.pi * ureg.radian))
            )

            if invalid_angles:
                raise ValueError("angles must be a nonempty 1D array of finite angles between 0 and pi.")

        if directions is None:
            assert theta is not None

            directions = Directions.from_angles(
                polar_angles=theta,
                azimuth_angles=0 * ureg.radian,
            )

        engine = BornSeries(
            grid=target.grid,
            background_refractive_index=target.background_refractive_index,
            wavelength=self.source.wavelength,
            directions=directions,
            order=self.order,
        )

        result = engine.solve(volume=target)

        return Result(
            source=self.source,
            kind="volume",
            sample_volume=target.volume,
            angles=theta if theta is not None else None,
            directions=result.directions,
            differential=result.differential * (1 / ureg.meter / ureg.steradian),
            amplitudes=result.amplitudes * ureg.meter,
            term_differential=result.term_differential * (1 / ureg.meter / ureg.steradian),
            field_norms=result.field_norms,
            warnings=result.warnings,
            provenance=_provenance(
                order=self.order,
                grid={
                    "shape": list(target.delta_refractive_index.shape),
                    "spacing_m": float(target.spacing.to("meter").magnitude),
                    "background_refractive_index": target.background_refractive_index,
                    "delta_refractive_index_sha256": hashlib.sha256(
                        np.asarray(target.delta_refractive_index, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                },
                medium=None if target.medium is None else target.medium.metadata,
                seed=target.seed,
                coefficient_scope="finite-sample",
            ),
        )

    def _solve_volume(self, *, volume, sampling):
        sampling.check_work(grid=volume.grid, order=self.order)

        engine = BornSeries(
            grid=volume.grid,
            background_refractive_index=volume.background_refractive_index,
            wavelength=self.source.wavelength,
            directions=sampling.directions,
            order=self.order,
        )

        born = engine.solve(volume=volume)

        sampled = sampling.summarize(result=born)

        integrated = sampled["integrals"]

        mu = integrated[:, 0]

        g = np.divide(integrated[:, 1], mu, out=np.full_like(mu, np.nan), where=mu != 0)

        return Result(
            source=self.source,
            kind="volume",
            sample_volume=volume.volume,
            angles=sampling.angles,
            azimuths=sampling.azimuths,
            differential=sampled["directional_differential"] * (1 / ureg.meter / ureg.steradian),
            amplitudes=sampled["directional_amplitudes"] * ureg.meter,
            term_differential=sampled["directional_terms"] * (1 / ureg.meter / ureg.steradian),
            mu_s=mu * (1 / ureg.meter),
            g=g,
            mu_s_prime=integrated[:, 2] * (1 / ureg.meter),
            field_norms=born.field_norms,
            warnings=born.warnings,
            provenance=_provenance(
                order=self.order,
                grid={
                    **volume.grid.metadata,
                    "background_refractive_index": volume.background_refractive_index,
                    "delta_refractive_index_sha256": hashlib.sha256(
                        np.asarray(volume.delta_refractive_index, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                },
                sampling=sampling.metadata,
                medium=None if volume.medium is None else volume.medium.metadata,
                seed=volume.seed,
                coefficient_scope="finite-sample",
            ),
        )

    def ensemble(
        self,
        *,
        medium: Medium,
        grid=None,
        sampling=None,
        shape=None,
        spacing=None,
        ensemble_sampling=None,
        realizations=None,
        seed=None,
        angles=None,
        azimuth_samples=None,
        polar_samples=None,
    ):
        """Average sampled-volume intensities and integrate over solid angle.

        Parameters
        ----------
        medium : Medium
            Random medium or structured medium with a random background.
            Deterministic structures use solve on their generated Volume.
        grid : Grid, optional
            Explicit spatial configuration for every realization. Cannot be
            combined with shape or spacing.
        sampling : AngularSampling, optional
            Override shared angular settings. Defaults to the solver's sampling
            when individual angular keywords are omitted. Cannot be combined
            with angles, polar_samples, or azimuth_samples.
        shape : tuple of int, optional
            Three grid dimensions, each from 2 to 32. Required with spacing when grid is omitted.
        spacing : Quantity, optional
            Positive, finite cubic voxel spacing. Explicit length units are required;
            must be supplied with shape when grid is omitted.
        ensemble_sampling : EnsembleSampling, optional
            Realization counts and ordered independent seeds. Defaults to
            four consecutive seeds starting at zero.
        realizations : int, optional
            Deprecated; use EnsembleSampling.
            Independent sample count, from 1 to 32. Default is 4.
        seed : int, optional
            Deprecated; use EnsembleSampling.
            First sample seed, from 0 to 2**32 - 1. Default is 0. Samples use
            consecutive seeds; the entire seed range must remain valid.
        angles : array_like or Quantity, optional
            One-dimensional plot angles in [0, pi], with 1 to 181 observations.
            Angular values require explicit units. Default is 121 evenly spaced angles.
        azimuth_samples : int, optional
            Uniform azimuth samples for both curves and quadrature, from 4 to 32.
            Default is 8.
        polar_samples : int, optional
            Gauss-Legendre nodes for integration, from 16 to 128. Default is 32.
            These nodes are independent of the plot angles.

        Returns
        -------
        result : Result
            Directional cumulative and isolated-term intensities through
            ``self.order``, standard errors, finite-sample effective coefficients,
            per-realization field norms, and numerical diagnostics, all with units.
            Directional intensities retain azimuth for 3D phase plotting.
            Complex amplitudes are not retained in the ensemble result.

        Raises
        ------
        TypeError
            If ``medium`` is not a Medium.
        ValueError
            If a grid, sampling, seed, unit, or order constraint is violated,
            the ensemble exceeds the synchronous work limit, or a generated
            volume has nonpositive linearized permittivity.

        See Also
        --------
        bornsim.ensemble.ensemble_scattering : Numeric SI ensemble function.

        Notes
        -----
        Intensities are averaged after coherent summation within each realization;
        random amplitudes are never averaged together. Standard errors describe
        realization sampling, not spatial or angular discretization error. One
        realization gives NaN errors. Numerical integrated coefficients are
        finite-sample cross sections divided by volume. Check voxel refinement,
        angular quadrature, sample size, and realization count independently.
        Deterministic structures have no realization sampling and must use
        solve. A single random realization reports unknown sampling errors
        (NaN), rather than inferred uncertainty. Individual shape/spacing, angular and realization keywords remain
        available with deprecation warnings.

        Examples
        --------
        >>> from bornsim import RandomMedium, Solver, Source
        >>> from bornsim.units import ureg
        ...
        ...
        >>> ensemble_solver = Solver(
        ...     source=Source(
        ...         wavelength=633e-9 * ureg.meter,
        ...     ),
        ...     order=2,
        ... )

        ...
        >>> result = ensemble_solver.ensemble(
        ...     medium=RandomMedium(
        ...         correlation="gaussian",
        ...         background_refractive_index=1.33,
        ...         refractive_index_std=0.01,
        ...         correlation_length=100e-9 * ureg.meter,
        ...     ),
        ...     shape=(2, 2, 2),
        ...     realizations=2,
        ...     seed=42,
        ...     angles=[0, 1] * ureg.radian,
        ...     azimuth_samples=4,
        ...     polar_samples=16,
        ...     spacing=50e-9 * ureg.meter,
        ... )
        >>> result.stderr.shape
        (2, 2, 4)
        """

        if not isinstance(medium, Medium):
            raise TypeError("medium must be a Medium.")

        grid = Grid._resolve(
            grid=grid,
            shape=shape,
            spacing=spacing,
        )

        if sampling is None and all(value is None for value in (angles, azimuth_samples, polar_samples)):
            sampling = self.sampling

        sampling = AngularSampling._resolve(
            sampling=sampling,
            angles=angles,
            polar_samples=polar_samples,
            azimuth_samples=azimuth_samples,
        )

        ensemble_sampling = EnsembleSampling._resolve(
            ensemble_sampling=ensemble_sampling,
            realizations=realizations,
            seed=seed,
        )

        result = ensemble_scattering(
            medium=medium,
            wavelength=self.source.wavelength,
            grid=grid,
            sampling=sampling,
            order=self.order,
            ensemble_sampling=ensemble_sampling,
        )

        return Result(
            source=self.source,
            kind="ensemble",
            sample_volume=grid.volume,
            angles=result["angles"].copy(),
            differential=result["directional_differential"] * (1 / ureg.meter / ureg.steradian),
            azimuths=result["azimuths"],
            term_differential=result["directional_terms"] * (1 / ureg.meter / ureg.steradian),
            stderr=result["directional_stderr"] * (1 / ureg.meter / ureg.steradian),
            azimuth_stderr=result["stderr"] * (1 / ureg.meter / ureg.steradian),
            mu_s=result["mu_s"] * (1 / ureg.meter),
            g=result["g"],
            mu_s_prime=result["mu_s_prime"] * (1 / ureg.meter),
            field_norms=result["field_norms"],
            warnings=result["warnings"],
            realizations=result["realizations"],
            provenance=_provenance(
                order=self.order,
                medium=medium.metadata,
                grid=grid.metadata,
                sampling=sampling.metadata,
                seed=ensemble_sampling.seed,
                seeds=list(ensemble_sampling.seeds),
                ensemble_sampling=ensemble_sampling.metadata,
                realizations=result["realizations"],
                polar_samples=sampling.polar_samples,
                azimuth_samples=sampling.azimuth_samples,
                coefficient_scope="finite-sample",
            ),
        )
