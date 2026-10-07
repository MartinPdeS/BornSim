"""Numerical ensemble quadrature and realization sampling uncertainty."""

import numpy as np
from ._validation import _integer
from .media import Medium
from .grid import Grid
from .sampling import AngularSampling
from .ensemble_sampling import EnsembleSampling
from .series import BornSeries
from .units import Quantity, _si


def ensemble_scattering(
    *,
    medium: Medium,
    wavelength: Quantity | float,
    grid=None,
    sampling=None,
    shape=None,
    spacing=None,
    order=3,
    ensemble_sampling=None,
    realizations=None,
    seed=None,
    angles=None,
    azimuth_samples=None,
    polar_samples=None,
):
    """Retain directional intensities and integrate sampled-volume scattering.

    Parameters
    ----------
    medium : Medium
        Random medium or structure with a random background, sampled at
        consecutive seeds. Deterministic volumes use Solver.solve.
    grid : Grid, optional
        Shared spatial configuration. Cannot be combined with shape or spacing.
    sampling : AngularSampling, optional
        Shared output and integration settings. Cannot be combined with
        individual angular keywords.
    wavelength : float or Quantity
        Positive, finite vacuum wavelength; bare values mean metres.
    shape : tuple of int, optional
        Three grid dimensions, each from 2 to 32. Default is (12, 12, 12).
    spacing : float or Quantity, optional
        Positive, finite cubic voxel spacing; bare values mean metres.
        Default is 50 nm.
    order : int, optional
        Highest cumulative Born order, from 1 to 12. Default is 3.
    realizations : int, optional
        Independent sample count, from 1 to 32. Default is 4.
    seed : int, optional
        First seed, from 0 to 2**32 - 1. Default is 0. Consecutive seeds are
        used, and the last seed must also lie in this range.
    angles : array_like or Quantity, optional
        One-dimensional plot angles in [0, pi], with 1 to 181 observations.
        Bare values mean radians. Default is 121 evenly spaced angles.
    azimuth_samples : int, optional
        Uniform azimuth sample count, from 4 to 32. Default is 8.
    polar_samples : int, optional
        Gauss-Legendre node count, from 16 to 128. Default is 32. Integration
        nodes are independent of the supplied plot angles.

    Returns
    -------
    ensemble : dict
        Numeric SI arrays and metadata with the following keys:

        * ``angles`` : radians, shape (observation,).
        * ``azimuths`` : radians, shape (azimuth,), uniform in [0, 2*pi).
        * ``directional_differential`` : cumulative intensities in
          m^-1 sr^-1, shape (order, observation, azimuth), averaged only
          over realizations, preserving directional asymmetry.
        * ``mean`` : azimuth-averaged cumulative scattering in m^-1 sr^-1, shape
          (order, observation).
        * ``stderr`` : standard error of mean curves, same shape and units;
          NaN for one realization.
        * ``terms`` : mean isolated-term curves, same shape and units.
        * ``mu_s`` and ``mu_s_prime`` : finite-sample effective total and
          reduced scattering coefficients in m^-1, shape (order,).
        * ``g`` : dimensionless anisotropy, shape (order,); NaN for zero scattering.
        * ``field_norms`` : relative norms, shape (realization, order).
        * ``warnings`` : sorted tuple of numerical diagnostic messages.
        * ``realizations`` : independent sample count.

    Raises
    ------
    ValueError
        If grid, wavelength, units, angles, order, sample counts, or seed range
        are invalid, generated linearized permittivity is nonpositive, or the
        synchronous work estimate exceeds 100 million voxel-direction-order
        operations across all realizations.

    See Also
    --------
    bornsim.media.Medium.to_volume : Generate each seeded or deterministic sample.
    bornsim.series.BornSeries : Compute per-sample coherent amplitudes and intensities.
    bornsim.solver.Solver.ensemble : Return unitful ensemble data with plotting.

    Notes
    -----
    Interference is preserved within each realization. Intensities, not random
    amplitudes, are then averaged over realizations and azimuth. Standard
    errors use the sample standard deviation divided by the square root of
    the realization count; they exclude discretization and finite-size bias.
    Anisotropy is computed from averaged angular moments.
    Directional data retain each azimuth for the 3D phase surface;
    one-dimensional curves remain azimuth averages. With one random sample,
    sampling errors are unknown (NaN). A deterministic structure has no
    realization sampling and must use Solver.solve.

    The coefficients are finite-sample cross sections divided by volume,
    not automatically infinite-medium transport coefficients. Check grid,
    angular integration, sample size, synthesis box, and realization count.
    """
    if not isinstance(medium, Medium):
        raise TypeError("medium must be a Medium.")
    if not medium.is_random:
        raise ValueError("Ensembles require a random medium or background; use Solver.solve for a fixed volume.")
    wavelength = _si(
        value=wavelength,
        unit="meter",
        name="wavelength",
        scalar=True,
    )
    grid = Grid._resolve(
        grid=grid,
        shape=shape,
        spacing=spacing,
    )
    sampling = AngularSampling._resolve(
        sampling=sampling,
        angles=angles,
        polar_samples=polar_samples,
        azimuth_samples=azimuth_samples,
    )
    shape, spacing = grid.shape, grid.spacing
    ensemble_sampling = EnsembleSampling._resolve(
        ensemble_sampling=ensemble_sampling,
        realizations=realizations,
        seed=seed,
    )
    realizations = ensemble_sampling.realizations
    order = _integer(value=order, name="order", low=1, high=12)
    sampling.check_work(grid=grid, order=order, realizations=realizations)
    engine = BornSeries(
        grid=grid,
        background_index=medium.background_index,
        wavelength=wavelength,
        directions=sampling.directions,
        order=order,
    )
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
    correlation_length = statistics.get("correlation_length_m")
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
        "angles": np.asarray(sampling.angles).copy(),
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
