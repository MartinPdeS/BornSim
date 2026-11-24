from bornsim.medium.random_medium import GaussianMedium, ExponentialMedium
from bornsim import EnsembleSampling
from bornsim import AngularSampling
import numpy as np
import pytest
from bornsim import Volume
from bornsim.series import BornSeries
from bornsim import Grid
from bornsim.ensemble import ensemble_scattering
from bornsim.green import GreenOperator
from bornsim.units import ureg
from bornsim import Directions


def directions():
    return np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def independent_matrix(volume, wavelength):
    """Direct all-pairs Green tensor from derivatives of exp(ikr)/(4pi r)."""

    xyz = volume.positions.to("meter").magnitude.reshape(-1, 3)

    k0 = 2 * np.pi / wavelength

    k = k0 * volume.background_refractive_index

    n = len(xyz)

    blocks = np.empty((n, 3, n, 3), complex)

    for i in range(n):
        for j in range(n):
            displacement = xyz[i] - xyz[j]

            r = np.linalg.norm(displacement)

            if r:
                unit = displacement / r

                g = np.exp(1j * k * r) / (4 * np.pi * r)

                first = g * (1j * k - 1 / r)

                second = g * (-k * k - 2j * k / r + 2 / r**2)

                tensor = (
                    g * np.eye(3)
                    + (first / r * (np.eye(3) - np.outer(unit, unit)) + second * np.outer(unit, unit)) / k**2
                )

                blocks[i, :, j, :] = k0**2 * volume.spacing.to("meter").magnitude ** 3 * tensor
            else:
                a = (3 * volume.spacing.to("meter").magnitude ** 3 / (4 * np.pi)) ** (1 / 3)

                nodes, weights = np.polynomial.legendre.leggauss(40)

                radial = (nodes + 1) * a / 2

                integrated = np.sum(weights * radial * np.exp(1j * k * radial)) * a / 2

                blocks[i, :, j, :] = k0**2 * (2 / 3 * integrated - 1 / (3 * k * k)) * np.eye(3)

    return blocks.reshape(3 * n, 3 * n)


def test_fft_matches_independent_nonperiodic_matrix():
    volume = Volume(
        delta_refractive_index=np.zeros((2, 3, 2)),
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(np.zeros((2, 3, 2))),
            spacing=4.5e-08 * ureg.meter,
        ),
    )

    operator = GreenOperator(
        shape=volume.delta_refractive_index.shape,
        spacing=volume.spacing.to("meter"),
        wavelength=6.33e-07 * ureg.meter,
        background_refractive_index=volume.background_refractive_index,
    )

    source = np.random.default_rng(3).normal(size=(2, 3, 2, 2, 3)) + 1j

    reference = independent_matrix(volume, 6.33e-07) @ source.reshape(-1, 2, 3).transpose(0, 2, 1).reshape(-1, 2)

    actual = operator.apply(source=source).reshape(-1, 2, 3).transpose(0, 2, 1).reshape(-1, 2)

    np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=2e-12)


def test_series_converges_toward_direct_linear_solve():
    delta = np.arange(8).reshape(2, 2, 2) * 0.004 + 0.01

    volume = Volume(
        delta_refractive_index=delta,
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(delta),
            spacing=4e-08 * ureg.meter,
        ),
    )

    xyz = volume.positions.to("meter").magnitude.reshape(-1, 3)

    incident = (
        np.exp(1j * 2 * np.pi / 6.33e-07 * volume.background_refractive_index * xyz[:, 2])[:, None, None]
        * np.eye(3)[:2]
    )

    contrast = 2 * volume.background_refractive_index * delta.ravel()

    matrix = independent_matrix(volume, 6.33e-07) * np.repeat(contrast, 3)[None, :]

    incident_flat = incident.transpose(0, 2, 1).reshape(-1, 2)

    exact = (
        np.linalg.solve(
            np.eye(matrix.shape[0]) - matrix,
            incident_flat,
        )
        .reshape(-1, 3, 2)
        .transpose(0, 2, 1)
    )

    observation = directions()

    phase = np.exp(-1j * 2 * np.pi / 6.33e-07 * volume.background_refractive_index * (observation @ xyz.T))

    amplitude = (
        np.einsum("dn,npv,n->dpv", phase, exact, contrast)
        * (2 * np.pi / 6.33e-07) ** 2
        * volume.spacing.to("meter").magnitude ** 3
        / (4 * np.pi)
    )

    amplitude -= observation[:, None, :] * np.einsum("dv,dpv->dp", observation, amplitude)[..., None]

    result = BornSeries(
        background_refractive_index=volume.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=observation),
        order=3,
        grid=Grid(
            shape=volume.delta_refractive_index.shape,
            spacing=volume.spacing.to("meter"),
        ),
    ).solve(volume=volume)

    errors = np.linalg.norm((np.cumsum(result.amplitudes, axis=0) - amplitude).reshape(3, -1), axis=1)

    assert errors[2] < errors[1] < errors[0]

    assert errors[2] / np.linalg.norm(amplitude) < 0.0001


def test_each_amplitude_scales_with_its_order_and_interferes():
    volume = Volume(
        delta_refractive_index=np.ones((2, 2, 2)) * 0.08,
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(np.ones((2, 2, 2)) * 0.08),
            spacing=4.5e-08 * ureg.meter,
        ),
    )

    first = BornSeries(
        background_refractive_index=volume.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions()),
        order=3,
        grid=Grid(
            shape=volume.delta_refractive_index.shape,
            spacing=volume.spacing.to("meter"),
        ),
    ).solve(volume=volume)

    second = BornSeries(
        background_refractive_index=1.33,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions()),
        order=3,
        grid=Grid(
            shape=volume.delta_refractive_index.shape,
            spacing=volume.spacing.to("meter"),
        ),
    ).solve(
        volume=Volume(
            delta_refractive_index=volume.delta_refractive_index * 2,
            background_refractive_index=1.33,
            grid=Grid(
                shape=np.shape(volume.delta_refractive_index * 2),
                spacing=volume.spacing.to("meter"),
            ),
        )
    )

    for index in range(3):
        np.testing.assert_allclose(second.amplitudes[index], first.amplitudes[index] * 2 ** (index + 1), atol=1e-22)

    assert not np.allclose(first.differential[-1], first.term_differential.sum(axis=0), rtol=1e-05, atol=0)

    np.testing.assert_allclose(first.differential[0], first.term_differential[0])


def test_first_order_forward_amplitude_normalization_and_transversality():
    volume = Volume(
        delta_refractive_index=np.ones((2, 3, 2)) * 0.01,
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(np.ones((2, 3, 2)) * 0.01),
            spacing=3e-08 * ureg.meter,
        ),
    )

    result = BornSeries(
        background_refractive_index=volume.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions()),
        order=1,
        grid=Grid(
            shape=volume.delta_refractive_index.shape,
            spacing=volume.spacing.to("meter"),
        ),
    ).solve(volume=volume)

    expected = (
        (2 * np.pi / 6.33e-07) ** 2 * (2 * volume.background_refractive_index * 0.01) * volume.volume / (4 * np.pi)
    )

    np.testing.assert_allclose(result.amplitudes[0, 0], expected * np.eye(3)[:2], atol=1e-22)

    np.testing.assert_allclose(np.einsum("dv,odpv->odp", directions(), result.amplitudes), 0, atol=1e-22)


def test_zero_contrast_and_seed_reproducibility():
    medium = GaussianMedium(
        refractive_index_std=0,
        background_refractive_index=1.33,
        correlation_length=1e-07 * ureg.meter,
    )

    volume = medium.to_volume(
        seed=4,
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        ),
    )

    result = BornSeries(
        background_refractive_index=volume.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions()),
        order=3,
        grid=Grid(
            shape=volume.delta_refractive_index.shape,
            spacing=volume.spacing.to("meter"),
        ),
    ).solve(volume=volume)

    assert not np.any(result.amplitudes)

    assert not np.any(result.field_norms)

    a = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    ).to_volume(
        seed=10,
        grid=Grid(
            shape=(4, 4, 4),
            spacing=5e-08 * ureg.meter,
        ),
    )

    b = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    ).to_volume(
        seed=10,
        grid=Grid(
            shape=(4, 4, 4),
            spacing=5e-08 * ureg.meter,
        ),
    )

    np.testing.assert_array_equal(a.delta_refractive_index, b.delta_refractive_index)

    assert not a.delta_refractive_index.flags.writeable

    assert not np.array_equal(
        a.delta_refractive_index,
        GaussianMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=1e-07 * ureg.meter,
        )
        .to_volume(
            seed=11,
            grid=Grid(
                shape=(4, 4, 4),
                spacing=5e-08 * ureg.meter,
            ),
        )
        .delta_refractive_index,
    )


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_random_field_expected_variance_without_sample_normalization(correlation):
    medium = {
        "gaussian": GaussianMedium,
        "exponential": ExponentialMedium,
    }[correlation](
        refractive_index_std=0.02,
        correlation_length=5e-08 * ureg.meter,
        background_refractive_index=1.33,
    )

    fields = np.array(
        [
            medium.to_volume(
                seed=seed,
                grid=Grid(
                    shape=(8, 8, 8),
                    spacing=5e-08 * ureg.meter,
                ),
            ).delta_refractive_index
            for seed in range(100)
        ]
    )

    assert np.mean(fields**2) == pytest.approx(medium.refractive_index_std**2, rel=0.06)

    assert np.std(fields.mean(axis=(1, 2, 3))) > 0.0001


def test_ensemble_mean_and_error_against_manual_realizations():
    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    )

    angles = np.array([0.0, 0.7, np.pi])

    ensemble = ensemble_scattering(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
        order=3,
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        ),
        sampling=AngularSampling(
            angles=angles * ureg.radian,
            azimuth_samples=4,
        ),
        ensemble_sampling=EnsembleSampling(
            realizations=3,
            seed=12,
        ),
    )

    phi = np.arange(4) * np.pi / 2

    tt, pp = np.meshgrid(angles, phi, indexing="ij")

    observe = np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1).reshape(-1, 3)

    curves = np.array(
        [
            BornSeries(
                background_refractive_index=medium.background_refractive_index,
                wavelength=6.33e-07 * ureg.meter,
                directions=Directions(vectors=observe),
                order=3,
                grid=Grid(
                    shape=(3, 3, 3),
                    spacing=5e-08 * ureg.meter,
                ),
            )
            .solve(
                volume=medium.to_volume(
                    seed=seed,
                    grid=Grid(
                        shape=(3, 3, 3),
                        spacing=5e-08 * ureg.meter,
                    ),
                )
            )
            .differential.reshape(3, 3, 4)
            .mean(axis=-1)
            for seed in range(12, 15)
        ]
    )

    np.testing.assert_allclose(ensemble["mean"], curves.mean(axis=0))

    np.testing.assert_allclose(ensemble["stderr"], curves.std(axis=0, ddof=1) / np.sqrt(3))

    np.testing.assert_allclose(ensemble["mu_s_prime"], ensemble["mu_s"] * (1 - ensemble["g"]))


@pytest.mark.parametrize("order", [0, 13, 1.2, True])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        BornSeries(
            background_refractive_index=1.33,
            wavelength=6.33e-07 * ureg.meter,
            directions=Directions(vectors=directions()),
            order=order,
            grid=Grid(
                shape=(2, 2, 2),
                spacing=5e-08 * ureg.meter,
            ),
        ).solve(
            volume=Volume(
                delta_refractive_index=np.zeros((2, 2, 2)),
                background_refractive_index=1.33,
                grid=Grid(
                    shape=np.shape(np.zeros((2, 2, 2))),
                    spacing=5e-08 * ureg.meter,
                ),
            )
        )


def test_invalid_shapes_directions_and_work_limits():
    with pytest.raises(ValueError):
        Volume(
            delta_refractive_index=np.zeros((1, 2, 3)),
            background_refractive_index=1.33,
            grid=Grid(
                shape=np.shape(np.zeros((1, 2, 3))),
                spacing=5e-08 * ureg.meter,
            ),
        )

    volume = Volume(
        delta_refractive_index=np.zeros((2, 2, 2)),
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(np.zeros((2, 2, 2))),
            spacing=5e-08 * ureg.meter,
        ),
    )

    with pytest.raises(ValueError):
        BornSeries(
            background_refractive_index=volume.background_refractive_index,
            wavelength=6.33e-07 * ureg.meter,
            directions=Directions(vectors=[[0, 0, 2]]),
            grid=Grid(
                shape=volume.delta_refractive_index.shape,
                spacing=volume.spacing.to("meter"),
            ),
        ).solve(volume=volume)

    with pytest.raises(ValueError, match="too large"):
        ensemble_scattering(
            medium=GaussianMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=1e-07 * ureg.meter,
            ),
            wavelength=6.33e-07 * ureg.meter,
            order=12,
            grid=Grid(
                shape=(32, 32, 32),
                spacing=5e-08 * ureg.meter,
            ),
            ensemble_sampling=EnsembleSampling(realizations=32),
        )


def test_single_realization_uncertainty_is_unknown():
    result = ensemble_scattering(
        medium=GaussianMedium(
            refractive_index_std=0,
            background_refractive_index=1.33,
            correlation_length=1e-07 * ureg.meter,
        ),
        wavelength=6.33e-07 * ureg.meter,
        grid=Grid(
            shape=(2, 2, 2),
            spacing=5e-08 * ureg.meter,
        ),
        sampling=AngularSampling(angles=[0, 1] * ureg.radian),
        ensemble_sampling=EnsembleSampling(realizations=1),
    )

    assert np.all(np.isnan(result["stderr"]))

    assert np.all(np.isnan(result["g"]))


def test_first_order_grid_refinement_toward_uniform_cube_integral():
    length = 1.8e-07

    wavelength = 6.33e-07

    background = 1.33

    contrast = 0.03

    observe = np.array([[1.0, 0.0, 0.0]])

    q = 2 * np.pi / wavelength * background * np.array([-1.0, 0.0, 1.0])

    integral = length**3 * np.prod(np.sinc(q * length / (2 * np.pi)))

    expected = (2 * np.pi / wavelength) ** 2 * contrast * integral / (4 * np.pi)

    errors = []

    for cells in (2, 4, 8):
        volume = Volume(
            delta_refractive_index=np.full((cells,) * 3, contrast / (2 * background)),
            background_refractive_index=background,
            grid=Grid(
                shape=np.shape(np.full((cells,) * 3, contrast / (2 * background))),
                spacing=length / cells * ureg.meter,
            ),
        )

        result = BornSeries(
            background_refractive_index=volume.background_refractive_index,
            wavelength=wavelength * ureg.meter,
            directions=Directions(vectors=observe),
            order=1,
            grid=Grid(
                shape=volume.delta_refractive_index.shape,
                spacing=volume.spacing.to("meter"),
            ),
        ).solve(volume=volume)

        errors.append(abs(result.amplitudes[0, 0, 1, 1] - expected))

    assert errors[2] < errors[1] < errors[0]


def test_gaussian_neighbor_covariance():
    medium = GaussianMedium(
        refractive_index_std=0.02,
        correlation_length=1e-07 * ureg.meter,
        background_refractive_index=1.33,
    )

    fields = np.array(
        [
            medium.to_volume(
                seed=seed,
                grid=Grid(
                    shape=(12,) * 3,
                    spacing=5e-08 * ureg.meter,
                ),
            ).delta_refractive_index
            for seed in range(100)
        ]
    )

    covariance = np.mean(fields[:, :-1] * fields[:, 1:])

    expected = medium.refractive_index_std**2 * np.exp(
        -0.5 * (5e-08 / medium.correlation_length.to("meter").magnitude) ** 2
    )

    assert covariance == pytest.approx(expected, rel=0.08)


def test_warning_when_higher_terms_grow():
    volume = Volume(
        delta_refractive_index=np.full((2,) * 3, 3.0),
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(np.full((2,) * 3, 3.0)),
            spacing=4e-08 * ureg.meter,
        ),
    )

    result = BornSeries(
        background_refractive_index=volume.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions()),
        order=3,
        grid=Grid(
            shape=volume.delta_refractive_index.shape,
            spacing=volume.spacing.to("meter"),
        ),
    ).solve(volume=volume)

    assert any(("not decreasing" in message for message in result.warnings))


def test_integrated_results_converge_with_angular_resolution():
    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    )

    coarse = ensemble_scattering(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
        grid=Grid(
            shape=(3,) * 3,
            spacing=5e-08 * ureg.meter,
        ),
        sampling=AngularSampling(
            angles=[0] * ureg.radian,
            polar_samples=16,
            azimuth_samples=8,
        ),
        ensemble_sampling=EnsembleSampling(realizations=2),
    )

    fine = ensemble_scattering(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
        grid=Grid(
            shape=(3,) * 3,
            spacing=5e-08 * ureg.meter,
        ),
        sampling=AngularSampling(
            angles=[0] * ureg.radian,
            polar_samples=64,
            azimuth_samples=16,
        ),
        ensemble_sampling=EnsembleSampling(realizations=2),
    )

    reference = ensemble_scattering(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
        grid=Grid(
            shape=(3,) * 3,
            spacing=5e-08 * ureg.meter,
        ),
        sampling=AngularSampling(
            angles=[0] * ureg.radian,
            polar_samples=128,
            azimuth_samples=32,
        ),
        ensemble_sampling=EnsembleSampling(realizations=2),
    )

    coarse_error = np.linalg.norm(coarse["mu_s"] - reference["mu_s"])

    fine_error = np.linalg.norm(fine["mu_s"] - reference["mu_s"])

    assert fine_error < coarse_error

    np.testing.assert_allclose(fine["mu_s"], reference["mu_s"], rtol=1e-10)

    np.testing.assert_allclose(fine["g"], reference["g"], atol=1e-10)
