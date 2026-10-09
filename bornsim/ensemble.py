"""Numerical ensemble quadrature and realization sampling uncertainty."""

from typing import Any
import numpy as np
from ._validation import _integer
from .medium import Medium
from .grid import Grid
from .sampling import AngularSampling
from .ensemble_sampling import EnsembleSampling
from .series import BornSeries
from .units import Quantity, validate_units, ureg


def ensemble_scattering(
    *,
    medium: Medium,
    wavelength: Quantity,
    grid: Grid,
    sampling: AngularSampling | None = None,
    order: int = 3,
    ensemble_sampling: EnsembleSampling | None = None,
) -> dict[str, Any]:
    """Run numerical Born ensembles on an explicit Grid.

    A random Medium generates one Volume per ordered seed in EnsembleSampling.
    AngularSampling defines output observations and independent integration
    quadrature. Complex amplitudes interfere within each realization; intensities
    are averaged across realizations. The returned dictionary contains numeric
    SI directional intensities, isolated-term intensities, integrated moments,
    standard errors, per-realization field norms, and diagnostic warnings.
    Angles and azimuths retain quantities from the sampling configuration.

    Coefficients are finite-sample cross sections divided by voxel-box volume.
    Standard errors describe realization sampling; one realization produces NaN.
    Check spatial, angular, finite-size, and ensemble convergence separately."""

    if not isinstance(medium, Medium):
        raise TypeError("medium must be a Medium.")

    if not medium.is_random:
        raise ValueError("Ensembles require a random medium or background; use Solver.solve for a fixed volume.")

    validate_units(
        wavelength,
        unit="meter",
        name="wavelength",
        scalar=True,
    )

    grid = Grid._resolve(grid=grid)

    sampling = AngularSampling._resolve(sampling=sampling)

    shape, spacing = grid.shape, grid.spacing

    ensemble_sampling = EnsembleSampling._resolve(ensemble_sampling=ensemble_sampling)

    realizations = ensemble_sampling.realizations

    order = _integer(value=order, name="order", low=1, high=12)

    sampling.check_work(
        grid=grid,
        order=order,
        realizations=realizations,
    )

    if medium.background_refractive_index is None:
        raise ValueError("Set an explicit background refractive index before solving an ensemble.")

    engine = BornSeries(
        grid=grid,
        background_refractive_index=medium.background_refractive_index,
        wavelength=wavelength,
        directions=sampling.directions,
        order=order,
    )

    notices: set[str]

    curve_samples, directional_samples, directional_terms, terms, integrals, norms, notices = (
        [],
        [],
        [],
        [],
        [],
        [],
        set(),
    )

    for sample_seed in ensemble_sampling.seeds:
        volume = medium.to_volume(
            grid=grid,
            seed=sample_seed,
        )

        result = engine.solve(volume=volume)

        sampled = sampling.summarize(result=result)

        directional_samples.append(sampled["directional_differential"])

        curve_samples.append(sampled["mean"])

        terms.append(sampled["terms"])

        directional_terms.append(sampled["directional_terms"])

        integrals.append(sampled["integrals"])

        norms.append(result.field_norms)

        notices.update(result.warnings)

    curves = np.asarray(curve_samples)

    integrated = np.mean(integrals, axis=0)

    mu = integrated[:, 0]

    anisotropy = np.divide(integrated[:, 1], mu, out=np.full_like(mu, np.nan), where=mu != 0)

    metadata = medium.metadata

    statistics = metadata.get("background", metadata)

    correlation_length_m = statistics.get("correlation_length_m") if isinstance(statistics, dict) else None

    correlation_length = None if correlation_length_m is None else correlation_length_m * ureg.meter

    if correlation_length is not None:
        if spacing > correlation_length / 2:
            notices.add("Correlation length is poorly resolved; refine voxel spacing.")

        if min(shape) * spacing < 6 * correlation_length:
            notices.add("Sample is smaller than six correlation lengths; check finite-size and synthesis-box effects.")

    stderr = curves.std(axis=0, ddof=1) / np.sqrt(realizations) if realizations > 1 else np.full_like(curves[0], np.nan)

    directional = np.asarray(directional_samples)

    directional_stderr = (
        directional.std(axis=0, ddof=1) / np.sqrt(realizations)
        if realizations > 1
        else np.full_like(directional[0], np.nan)
    )

    return {
        "angles": sampling.angles.copy(),
        "azimuths": sampling.azimuths,
        "directional_differential": directional.mean(axis=0),
        "directional_terms": np.mean(directional_terms, axis=0),
        "directional_stderr": directional_stderr,
        "mean": curves.mean(axis=0),
        "stderr": stderr,
        "terms": np.mean(terms, axis=0),
        "mu_s": mu,
        "g": anisotropy,
        "mu_s_prime": integrated[:, 2],
        "field_norms": np.asarray(norms),
        "warnings": tuple(sorted(notices)),
        "realizations": realizations,
        "seeds": ensemble_sampling.seeds,
    }
