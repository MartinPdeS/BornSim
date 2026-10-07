import numpy as np
import pytest

from bornsim import AnalyticalMedium, Medium, RandomMedium, Result, Solver, Source, StructuredMedium
from bornsim.media import random_volume
from bornsim.units import ureg


def test_matern_half_matches_exponential_with_same_seed():
    old = random_volume(
        medium=RandomMedium(correlation="exponential"),
        shape=(5, 6, 4),
        seed=31,
    )
    new = random_volume(
        medium=RandomMedium(smoothness=0.5),
        shape=(5, 6, 4),
        seed=31,
    )
    np.testing.assert_allclose(new.delta_index, old.delta_index, rtol=1e-13, atol=1e-17)
    assert new.medium.smoothness == 0.5
    assert new.seed == 31
    np.testing.assert_array_equal(
        new.delta_index,
        random_volume(
            medium=new.medium,
            shape=(5, 6, 4),
            seed=31,
        ).delta_index,
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
    # Frozen reference generated before the Medium hierarchy refactor.
    new = random_volume(
        medium=RandomMedium(correlation=correlation),
        shape=(2, 2, 2),
        seed=4,
    )
    np.testing.assert_allclose(new.delta_index.ravel(), expected, rtol=1e-13, atol=1e-17)


def test_matern_three_halves_covariance_against_closed_form():
    # Independent real-space reference: C(r)/sigma² = (1+x)exp(-x),
    # x=sqrt(3)r/ell. Finite spectral sampling introduces a small bias.
    ell, spacing = 60e-9, 20e-9
    medium = RandomMedium(
        index_std=0.02,
        correlation_length=ell,
        smoothness=1.5,
    )
    fields = np.array(
        [
            random_volume(
                medium=medium,
                shape=(16, 16, 16),
                spacing=spacing,
                seed=seed,
            ).delta_index
            for seed in range(180)
        ]
    )
    assert np.mean(fields**2) == pytest.approx(medium.index_std**2, rel=0.06)
    for lag in (1, 3, 6):
        measured = np.mean(fields[:, :-lag] * fields[:, lag:]) / medium.index_std**2
        x = np.sqrt(3) * lag * spacing / ell
        assert measured == pytest.approx((1 + x) * np.exp(-x), abs=0.04)
    assert np.std(fields.mean(axis=(1, 2, 3))) > 0.001
    assert np.std(fields.std(axis=(1, 2, 3))) > 0.0001


def test_numerical_ensemble_and_archive_preserve_matern_metadata(tmp_path):
    medium = RandomMedium(
        correlation_length=40 * ureg.nanometer,
        smoothness=2.5,
    )
    solver = Solver(
        source=Source(),
        order=2,
    )
    volume = random_volume(
        medium=medium,
        shape=(2, 2, 2),
        seed=9,
    )
    single = solver.solve_cut(
        target=volume,
        angles=[0, 1],
    )
    ensemble = solver.ensemble(
        medium=medium,
        shape=(2, 2, 2),
        realizations=2,
        seed=9,
        angles=[0, 1],
        azimuth_samples=4,
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
        {"correlation": "unsupported"},
        {"index_std": -1},
        {"correlation_length": 0},
    ],
)
def test_invalid_random_statistics(kwargs):
    with pytest.raises(ValueError):
        RandomMedium(**kwargs)


def test_spectrum_input_validation_and_units():
    medium = RandomMedium(correlation_length=50 * ureg.nanometer)
    assert medium.correlation_length == pytest.approx(50e-9)
    assert medium.spectral_weight(q_squared=0) == 1
    for invalid in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            medium.spectral_weight(q_squared=invalid)
    with pytest.raises(TypeError):
        random_volume(medium=object())


def test_medium_is_abstract_and_concrete_classes_share_volume_interface():
    import inspect

    assert inspect.isabstract(Medium)
    with pytest.raises(TypeError, match="abstract"):
        Medium()
    for medium in (RandomMedium(), StructuredMedium(), AnalyticalMedium()):
        assert isinstance(medium, Medium)
        assert not inspect.isabstract(type(medium))
        volume = medium.to_volume(
            shape=(2, 3, 2),
            spacing=30 * ureg.nanometer,
        )
        assert volume.delta_index.shape == (2, 3, 2)
        assert volume.spacing == pytest.approx(30e-9)
        assert volume.background_index == medium.background_index
    medium = RandomMedium()
    seeded = medium.to_volume(
        shape=(2, 2, 2),
        seed=24,
    )
    assert seeded.medium is medium and seeded.seed == 24
    np.testing.assert_array_equal(
        seeded.delta_index,
        medium.to_volume(
            shape=(2, 2, 2),
            seed=24,
        ).delta_index,
    )
    with pytest.raises(TypeError, match="RandomMedium"):
        random_volume(medium=StructuredMedium())
    with pytest.raises(TypeError, match="AnalyticalMedium"):
        Solver(source=Source()).solve(target=StructuredMedium())


def test_analytical_path_requires_explicit_analytical_medium():
    from bornsim.model import angular_scattering, optical_properties

    for medium in (RandomMedium(correlation="gaussian"), StructuredMedium()):
        with pytest.raises(TypeError, match="AnalyticalMedium"):
            angular_scattering(
                medium=medium,
                wavelength=633e-9,
                theta=[0],
            )
        with pytest.raises(TypeError, match="AnalyticalMedium"):
            optical_properties(
                medium=medium,
                wavelength=633e-9,
            )
    with pytest.raises(ValueError, match="gaussian or exponential"):
        AnalyticalMedium(correlation="matern")
