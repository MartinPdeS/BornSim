"""Solver configuration for finite-volume and ensemble Born calculations."""

from dataclasses import dataclass, field
import hashlib
import numpy as np
from ._version import __version__
from .medium import Medium
from .results import Result
from .volume import Volume
from .series import BornSeries
from .ensemble import ensemble_scattering
from .grid import Grid
from .directions import Directions
from .sampling import AngularSampling
from .ensemble_sampling import EnsembleSampling
from .source import Source
from .units import Quantity, _integer, validate_units, ureg


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
    """Configure numerical Born scattering from explicit finite samples.

    Source chooses wavelength, incident direction, and polarization. AngularSampling supplies output directions
    and integration quadrature; order sets the highest cumulative Born order.
    All orders retain the linearized dielectric contrast and the equal-volume
    Green self cell. Decreasing successive terms do not certify convergence.
    """

    source: Source
    sampling: AngularSampling = field(default_factory=AngularSampling)
    order: int = 3

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

    def __repr__(self):
        wavelength = getattr(self.source, "wavelength").to("nanometer")

        return f"Solver(wavelength={wavelength.magnitude:g} nm, order={self.order}, sampling={self.sampling!r})"

    def solve(
        self,
        *,
        target: Volume,
        sampling: AngularSampling | Directions | None = None,
    ) -> Result:
        """Compute scattering from a volume using angular or Cartesian sampling.

        Pass AngularSampling for solid-angle integration, or Directions for an
        unnormalized set of observation directions. The target is an explicit
        voxelized Volume; random statistics use ensemble instead.
        """

        if not isinstance(target, Volume):
            raise TypeError("target must be a Volume.")

        if isinstance(sampling, Directions):
            return self.solve_cut(
                target=target,
                directions=sampling,
            )

        sampling = self.sampling if sampling is None else AngularSampling._resolve(sampling=sampling)

        return self._solve_volume(volume=target, sampling=sampling)

    def solve_cut(
        self,
        *,
        target: Volume,
        angles: Quantity | None = None,
        directions: Directions | None = None,
    ) -> Result:
        """Compute an unnormalized angular cut through a fixed Volume.

        Supply laboratory polar angles on the x-z meridian or a Directions configuration,
        but not both. This is a convenience wrapper for ``solve(target=...,
        sampling=Directions(...))``. Explicit angular units are required.
        Default angles span 0 to pi with 121 samples. Cut amplitudes and
        intensities retain every requested direction. A cut alone cannot
        determine a solid-angle integral; integrated coefficients and normalized
        phase functions are unavailable.
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
            incident_direction=self.source.incident_direction,
            polarization=self.source.polarization,
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
            mu_a=result.mu_a * (1 / ureg.meter),
            residual_norms=result.residual_norms,
            warnings=result.warnings,
            provenance=_provenance(
                order=self.order,
                source=self.source.metadata,
                grid={
                    "shape": list(target.delta_refractive_index.shape),
                    "spacing_m": float(target.spacing.to("meter").magnitude),
                    "background_refractive_index": target.background_refractive_index,
                    "delta_refractive_index_sha256": hashlib.sha256(
                        np.asarray(target.delta_refractive_index, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                },
                medium=target.medium_metadata,
                seed=target.seed,
                coefficient_scope="finite-sample",
                absorption={
                    "contrast": "2*n0*(delta_n+i*kappa)",
                    "field_order": "M-1 for absorption reported at scattering order M",
                    "extinction_coefficient_sha256": hashlib.sha256(
                        np.asarray(target.extinction_coefficient, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                },
            ),
        )

    def _solve_volume(self, *, volume, sampling):
        sampling.check_work(
            grid=volume.grid,
            order=self.order,
        )

        engine = BornSeries(
            grid=volume.grid,
            background_refractive_index=volume.background_refractive_index,
            wavelength=self.source.wavelength,
            incident_direction=self.source.incident_direction,
            polarization=self.source.polarization,
            directions=sampling.directions,
            order=self.order,
        )

        born = engine.solve(volume=volume)

        sampled = sampling.summarize(
            result=born,
            incident_direction=self.source.incident_direction,
        )

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
            mu_a=born.mu_a * (1 / ureg.meter),
            residual_norms=born.residual_norms,
            warnings=born.warnings,
            provenance=_provenance(
                order=self.order,
                source=self.source.metadata,
                grid={
                    **volume.grid.metadata,
                    "background_refractive_index": volume.background_refractive_index,
                    "delta_refractive_index_sha256": hashlib.sha256(
                        np.asarray(volume.delta_refractive_index, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                },
                sampling=sampling.metadata,
                medium=volume.medium_metadata,
                seed=volume.seed,
                coefficient_scope="finite-sample",
                absorption={
                    "contrast": "2*n0*(delta_n+i*kappa)",
                    "field_order": "M-1 for absorption reported at scattering order M",
                    "extinction_coefficient_sha256": hashlib.sha256(
                        np.asarray(volume.extinction_coefficient, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                },
            ),
        )

    def ensemble(
        self,
        *,
        medium: Medium,
        grid: Grid,
        sampling: AngularSampling | None = None,
        ensemble_sampling: EnsembleSampling | None = None,
    ) -> Result:
        """Average numerical scattering over independently generated finite samples.

        Grid selects the voxel domain. AngularSampling selects output directions
        and integration quadrature; the solver's sampling is used when omitted.
        EnsembleSampling selects ordered independent seeds and defaults to four
        consecutive seeds starting at zero. Deterministic structures use solve.

        Intensities are averaged after coherent summation within each realization.
        Standard errors quantify realization uncertainty, not discretization or
        finite-size error. One realization has unknown (NaN) uncertainty. Integrated
        coefficients are finite-sample cross sections divided by voxel-box volume."""

        if not isinstance(medium, Medium):
            raise TypeError("medium must be a Medium.")

        grid = Grid._resolve(grid=grid)

        sampling = self.sampling if sampling is None else sampling

        sampling = AngularSampling._resolve(sampling=sampling)

        ensemble_sampling = EnsembleSampling._resolve(ensemble_sampling=ensemble_sampling)

        result = ensemble_scattering(
            medium=medium,
            wavelength=self.source.wavelength,
            incident_direction=self.source.incident_direction,
            polarization=self.source.polarization,
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
            mu_a=result["mu_a"] * (1 / ureg.meter),
            mu_a_stderr=result["mu_a_stderr"] * (1 / ureg.meter),
            residual_norms=result["residual_norms"],
            coherent_differential=result["coherent_differential"] * (1 / ureg.meter / ureg.steradian),
            warnings=result["warnings"],
            realizations=result["realizations"],
            provenance=_provenance(
                order=self.order,
                source=self.source.metadata,
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
