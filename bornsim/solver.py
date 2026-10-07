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
from .sampling import AngularSampling
from .ensemble_sampling import EnsembleSampling
from .source import Source
from .units import _si


def _provenance(*, order, **settings):
    return {
        "bornsim_version": __version__,
        "numpy_version": np.__version__,
        "order": int(order),
        "dielectric_contrast": "2 * background_index * delta_index",
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
    >>> solver = Solver(
    ...     source=Source(),
    ...     order=3,
    ... )
    >>> result = solver.solve(target=AnalyticalMedium())
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

    def solve(self, *, target, sampling=None, angles=None, directions=None):
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
        directions : array_like or Quantity, optional
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
        theta = (
            np.asarray(sampling.angles).copy()
            if angles is None
            else _si(value=angles, unit="radian", name="angles").copy()
        )
        invalid_angles = (
            theta.ndim != 1 or theta.size == 0 or np.any(~np.isfinite(theta)) or np.any((theta < 0) | (theta > np.pi))
        )
        if invalid_angles:
            raise ValueError("angles must be a nonempty 1D array of finite angles between 0 and pi.")
        if angles is not None:
            import warnings

            warnings.warn("Use AngularSampling for analytical output angles.", DeprecationWarning, stacklevel=2)
        wavelength = _si(value=self.source.wavelength, unit="meter", name="wavelength", scalar=True)
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
            angles=theta,
            azimuths=sampling.azimuths,
            differential=directional,
            mu_s=np.array([coefficients["mu_s"]]),
            g=np.array([np.nan if coefficients["g"] is None else coefficients["g"]]),
            mu_s_prime=np.array([coefficients["mu_s_prime"]]),
            provenance=_provenance(
                order=1,
                medium=target.metadata,
                quadrature_order=self.quadrature_order,
                sampling={**sampling.metadata, "angles_rad": theta.tolist()},
                coefficient_scope="infinite-medium",
            ),
        )

    def solve_cut(self, *, target, angles=None, directions=None):
        """Compute an unnormalized angular cut through a fixed Volume.

        Supply polar angles on the x-z meridian or arbitrary unit directions,
        but not both. Bare angles mean radians. Default angles span 0 to pi
        with 121 samples. Cut amplitudes and intensities retain every requested
        direction. A cut alone cannot determine a solid-angle integral;
        integrated coefficients and normalized phase functions are unavailable.
        """
        if not isinstance(target, Volume):
            raise TypeError("target must be a Volume.")
        if angles is not None and directions is not None:
            raise ValueError("Supply angles or directions, not both.")
        theta = None
        if directions is None:
            theta = (
                np.linspace(0, np.pi, 121) if angles is None else _si(value=angles, unit="radian", name="angles").copy()
            )
            invalid_angles = (
                theta.ndim != 1
                or theta.size == 0
                or np.any(~np.isfinite(theta))
                or np.any((theta < 0) | (theta > np.pi))
            )
            if invalid_angles:
                raise ValueError("angles must be a nonempty 1D array of finite angles between 0 and pi.")
        if directions is None:
            assert theta is not None
            directions = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)
        engine = BornSeries(
            grid=target.grid,
            background_index=target.background_index,
            wavelength=_si(
                value=self.source.wavelength,
                unit="meter",
                name="wavelength",
                scalar=True,
            ),
            directions=directions,
            order=self.order,
        )

        result = engine.solve(volume=target)
        return Result(
            source=self.source,
            kind="volume",
            sample_volume=target.volume,
            angles=theta,
            directions=result.directions,
            differential=result.differential,
            amplitudes=result.amplitudes,
            term_differential=result.term_differential,
            field_norms=result.field_norms,
            warnings=result.warnings,
            provenance=_provenance(
                order=self.order,
                grid={
                    "shape": list(target.delta_index.shape),
                    "spacing_m": target.spacing,
                    "background_index": target.background_index,
                    "delta_index_sha256": hashlib.sha256(
                        np.asarray(target.delta_index, dtype="<f8").tobytes(order="C")
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
            background_index=volume.background_index,
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
            differential=sampled["directional_differential"],
            amplitudes=sampled["directional_amplitudes"],
            term_differential=sampled["directional_terms"],
            mu_s=mu,
            g=g,
            mu_s_prime=integrated[:, 2],
            field_norms=born.field_norms,
            warnings=born.warnings,
            provenance=_provenance(
                order=self.order,
                grid={
                    **volume.grid.metadata,
                    "background_index": volume.background_index,
                    "delta_index_sha256": hashlib.sha256(
                        np.asarray(volume.delta_index, dtype="<f8").tobytes(order="C")
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
            Shared spatial configuration for every realization. Defaults to
            Grid(). Cannot be combined with shape or spacing.
        sampling : AngularSampling, optional
            Override shared angular settings. Defaults to the solver's sampling
            when individual angular keywords are omitted. Cannot be combined
            with angles, polar_samples, or azimuth_samples.
        shape : tuple of int, optional
            Three grid dimensions, each from 2 to 32. Default is (12, 12, 12).
        spacing : float or Quantity, optional
            Positive, finite cubic voxel spacing. Bare values mean metres;
            default is 50 nm.
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
            Bare values mean radians. Default is 121 evenly spaced angles.
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
        >>> ensemble_solver = Solver(
        ...     source=Source(),
        ...     order=2,
        ... )

        >>> result = ensemble_solver.ensemble(
        ...     medium=RandomMedium(correlation="gaussian"),
        ...     shape=(2, 2, 2),
        ...     realizations=2,
        ...     seed=42,
        ...     angles=[0, 1],
        ...     azimuth_samples=4,
        ...     polar_samples=16,
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
            wavelength=_si(
                value=self.source.wavelength,
                unit="meter",
                name="wavelength",
                scalar=True,
            ),
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
            differential=result["directional_differential"],
            azimuths=result["azimuths"],
            term_differential=result["directional_terms"],
            stderr=result["directional_stderr"],
            azimuth_stderr=result["stderr"],
            mu_s=result["mu_s"],
            g=result["g"],
            mu_s_prime=result["mu_s_prime"],
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
