from bornsim.medium.random_medium import GaussianMedium, ExponentialMedium, WhittleMaternMedium
from bornsim import EnsembleSampling
from bornsim import AngularSampling
import numpy as np
import pytest
from bornsim import Medium, Result, Solver, Source, StructuredMedium
from bornsim import Grid
from bornsim.units import ureg


def test_matern_half_matches_exponential_with_same_seed():
    old = ExponentialMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    ).to_volume(
        seed=31,
        grid=Grid(
            shape=(5, 6, 4),
            spacing=5e-08 * ureg.meter,
        ),
    )

    new = WhittleMaternMedium(
        smoothness=0.5,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    ).to_volume(
        seed=31,
        grid=Grid(
            shape=(5, 6, 4),
            spacing=5e-08 * ureg.meter,
        ),
    )

    np.testing.assert_allclose(new.delta_refractive_index, old.delta_refractive_index, rtol=1e-13, atol=1e-17)

    assert new.medium.smoothness == 0.5

    assert new.seed == 31

    np.testing.assert_array_equal(
        new.delta_refractive_index,
        new.medium.to_volume(
            seed=31,
            grid=Grid(
                shape=(5, 6, 4),
                spacing=5e-08 * ureg.meter,
            ),
        ).delta_refractive_index,
    )


@pytest.mark.parametrize(
    "correlation, expected",
    [
        (
            "gaussian",
            [
                -0.002432456172346587,
                -0.003137874159262009,
                -0.004395382760754141,
                -0.0049952577199897765,
                -0.005114875090077147,
                -0.0057935739306334035,
                -0.006599986035676387,
                -0.007259161529999261,
            ],
        ),
        (
            "exponential",
            [
                -0.0037862638930110906,
                -0.0030812660129967686,
                -0.008542626484436341,
                -0.005225418915227462,
                -0.008937552131854258,
                -0.0035014394298047338,
                -0.006999859531067541,
                -0.0065343039176108935,
            ],
        ),
    ],
)
def test_existing_covariance_seeds_preserved(correlation, expected):
    new = {
        "gaussian": GaussianMedium,
        "exponential": ExponentialMedium,
    }[correlation](
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    ).to_volume(
        seed=4,
        grid=Grid(
            shape=(2, 2, 2),
            spacing=5e-08 * ureg.meter,
        ),
    )

    np.testing.assert_allclose(new.delta_refractive_index.ravel(), expected, rtol=1e-13, atol=1e-17)


def test_matern_three_halves_covariance_against_closed_form():
    ell, spacing = (6e-08, 2e-08)

    medium = WhittleMaternMedium(
        refractive_index_std=0.02,
        correlation_length=ell * ureg.meter,
        smoothness=1.5,
        background_refractive_index=1.33,
    )

    fields = np.array(
        [
            medium.to_volume(
                seed=seed,
                grid=Grid(
                    shape=(16, 16, 16),
                    spacing=spacing * ureg.meter,
                ),
            ).delta_refractive_index
            for seed in range(180)
        ]
    )

    assert np.mean(fields**2) == pytest.approx(medium.refractive_index_std**2, rel=0.06)

    for lag in (1, 3, 6):
        measured = np.mean(fields[:, :-lag] * fields[:, lag:]) / medium.refractive_index_std**2

        x = np.sqrt(3) * lag * spacing / ell

        assert measured == pytest.approx((1 + x) * np.exp(-x), abs=0.04)

    assert np.std(fields.mean(axis=(1, 2, 3))) > 0.001

    assert np.std(fields.std(axis=(1, 2, 3))) > 0.0001


def test_numerical_ensemble_and_archive_preserve_matern_metadata(tmp_path):
    medium = WhittleMaternMedium(
        correlation_length=40 * ureg.nanometer,
        smoothness=2.5,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
    )

    solver = Solver(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        order=2,
    )

    volume = medium.to_volume(
        seed=9,
        grid=Grid(
            shape=(2, 2, 2),
            spacing=5e-08 * ureg.meter,
        ),
    )

    single = solver.solve_cut(
        target=volume,
        angles=[0, 1] * ureg.radian,
    )

    ensemble = solver.ensemble(
        medium=medium,
        grid=Grid(
            shape=(2, 2, 2),
            spacing=5e-08 * ureg.meter,
        ),
        sampling=AngularSampling(
            angles=[0, 1] * ureg.radian,
            azimuth_samples=4,
        ),
        ensemble_sampling=EnsembleSampling(
            realizations=2,
            seed=9,
        ),
    )

    for result in (single, ensemble):
        assert result.provenance["medium"]["smoothness"] == 2.5

        assert result.provenance["medium"]["correlation"] == "matern"

        result.save(path=tmp_path / "result.npz")

        restored = Result.load(path=tmp_path / "result.npz")

        assert restored.provenance == result.provenance

        np.testing.assert_array_equal(restored.differential.magnitude, result.differential.magnitude)

    with pytest.raises(TypeError):
        solver.solve(target=medium)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothness": 0},
        {"smoothness": -1},
        {"smoothness": np.inf},
        {"smoothness": [1, 2]},
        {"smoothness": 1 * ureg.meter},
        {"refractive_index_std": -1},
        {"correlation_length": 0 * ureg.meter},
    ],
)
def test_invalid_random_statistics(kwargs):
    with pytest.raises(ValueError):
        WhittleMaternMedium(
            **{
                "background_refractive_index": 1.33,
                "refractive_index_std": 0.01,
                "correlation_length": 1e-07 * ureg.meter,
                "smoothness": 1.5,
                **kwargs,
            }
        )


def test_spectrum_input_validation_and_units():
    medium = WhittleMaternMedium(
        correlation_length=50 * ureg.nanometer,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        smoothness=1.5,
    )

    assert medium.correlation_length.to("meter").magnitude == pytest.approx(5e-08)

    assert medium.spectral_weight(q_squared=0 * (1 / ureg.meter**2)) == 1

    for invalid in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            medium.spectral_weight(q_squared=invalid * (1 / ureg.meter**2))


def test_medium_is_abstract_and_concrete_classes_share_volume_interface():
    import inspect

    assert inspect.isabstract(Medium)

    with pytest.raises(TypeError, match="abstract"):
        Medium()

    for medium in (
        WhittleMaternMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=1e-07 * ureg.meter,
            smoothness=1.5,
        ),
        StructuredMedium(background_refractive_index=1.33),
    ):
        assert isinstance(medium, Medium)

        assert not inspect.isabstract(type(medium))

        volume = medium.to_volume(
            grid=Grid(
                shape=(2, 3, 2),
                spacing=30 * ureg.nanometer,
            )
        )

        assert volume.delta_refractive_index.shape == (2, 3, 2)

        assert volume.spacing.to("meter").magnitude == pytest.approx(3e-08)

        assert volume.background_refractive_index == medium.background_refractive_index

    medium = WhittleMaternMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
        smoothness=1.5,
    )

    seeded = medium.to_volume(
        seed=24,
        grid=Grid(
            shape=(2, 2, 2),
            spacing=5e-08 * ureg.meter,
        ),
    )

    assert seeded.medium is medium and seeded.seed == 24

    np.testing.assert_array_equal(
        seeded.delta_refractive_index,
        medium.to_volume(
            seed=24,
            grid=Grid(
                shape=(2, 2, 2),
                spacing=5e-08 * ureg.meter,
            ),
        ).delta_refractive_index,
    )

    with pytest.raises(TypeError, match="Volume"):
        Solver(source=Source(wavelength=6.33e-07 * ureg.meter)).solve(
            target=StructuredMedium(background_refractive_index=1.33)
        )
