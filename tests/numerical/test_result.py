"""Preserve numerical provenance, interference and uncertainty in archives."""

from bornsim.medium.random_medium import GaussianMedium

from bornsim import EnsembleSampling
from bornsim import AngularSampling
import hashlib
import numpy as np
import pytest
from bornsim import Result, Solver, Source, Volume
from bornsim import Grid
from bornsim.units import ureg


def test_cross_section_curves_and_errors_preserve_units_and_sample_normalization():
    volume = Volume(
        delta_refractive_index=np.zeros((2, 3, 4)),
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(np.zeros((2, 3, 4))),
            spacing=2e-08 * ureg.meter,
        ),
    )

    result = Result(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        kind="ensemble",
        differential=[[1, 3]] * (1 / ureg.meter / ureg.steradian),
        stderr=[[0.1, 0.2]] * (1 / ureg.meter / ureg.steradian),
        angles=[0, np.pi] * ureg.radian,
        realizations=3,
    )

    figure = result.plot_cross_section(volume=volume, area_unit="nanometer**2", title="Sample")

    axis = figure.axes[0]

    scale = 24 * 2e-08**3 * 1e18

    np.testing.assert_allclose(axis.lines[0].get_ydata(), np.array([1, 3]) * scale)

    segments = axis.containers[0].lines[2][0].get_segments()

    np.testing.assert_allclose(segments[0][:, 1], np.array([0.9, 1.1]) * scale)

    assert axis.get_title() == "Sample"

    assert axis.get_ylabel() == "Differential cross section (nm² sr⁻¹)"

    np.testing.assert_array_equal(result.differential.magnitude, [[1, 3]])


def test_cross_section_isolated_terms_and_validation():
    volume = Volume(
        delta_refractive_index=np.full((2, 2, 2), 0.01),
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(np.full((2, 2, 2), 0.01)),
            spacing=2e-08 * ureg.meter,
        ),
    )

    result = Solver(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        order=2,
    ).solve_cut(
        target=volume,
        angles=[0, np.pi] * ureg.radian,
    )

    figure = result.plot_cross_section(volume=volume, terms=True, area_unit="meter**2")

    for curve, expected in zip(figure.axes[0].lines, result.term_differential.magnitude):
        np.testing.assert_allclose(curve.get_ydata(), expected * volume.volume)

    with pytest.raises(ValueError, match="area unit"):
        result.plot_cross_section(volume=volume, area_unit="meter")

    with pytest.raises(TypeError, match="Volume"):
        result.plot_cross_section(volume=1)

    with pytest.raises(ValueError, match="recorded sample"):
        result.plot_cross_section(
            volume=Volume(
                delta_refractive_index=volume.delta_refractive_index,
                background_refractive_index=1.33,
                grid=Grid(
                    shape=np.shape(volume.delta_refractive_index),
                    spacing=3e-08 * ureg.meter,
                ),
            )
        )


@pytest.mark.parametrize("kind", ["generated", "manual", "ensemble", "one-realization", "zero"])
def test_numerical_archives_round_trip(tmp_path, kind):
    medium = GaussianMedium(
        refractive_index_std=0 if kind == "zero" else 0.01,
        background_refractive_index=1.33,
        correlation_length=1e-07 * ureg.meter,
    )

    solver = Solver(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        order=3,
    )

    if kind in ("ensemble", "one-realization", "zero"):
        result = solver.ensemble(
            medium=medium,
            grid=Grid(
                shape=(2, 3, 2),
                spacing=30 * ureg.nanometer,
            ),
            sampling=AngularSampling(
                angles=[0, 90, 180] * ureg.degree,
                polar_samples=16,
                azimuth_samples=4,
            ),
            ensemble_sampling=EnsembleSampling(
                seed=42,
                realizations=1 if kind == "one-realization" else 3,
            ),
        )

        assert result.provenance["seed"] == 42

        assert result.provenance["polar_samples"] == 16

        assert result.provenance["azimuth_samples"] == 4

        assert result.provenance["realizations"] == result.realizations
    else:
        generated = medium.to_volume(
            seed=42,
            grid=Grid(
                shape=(2, 3, 2),
                spacing=30 * ureg.nanometer,
            ),
        )

        volume = (
            generated
            if kind == "generated"
            else Volume(
                delta_refractive_index=generated.delta_refractive_index,
                background_refractive_index=1.33,
                grid=Grid(
                    shape=np.shape(generated.delta_refractive_index),
                    spacing=generated.spacing.to("meter"),
                ),
            )
        )

        result = solver.solve_cut(
            target=volume,
            angles=[0, 90, 180] * ureg.degree,
        )

        assert result.provenance["seed"] == (42 if kind == "generated" else None)

        assert result.provenance["medium"] is not None if kind == "generated" else result.provenance["medium"] is None

        expected_hash = hashlib.sha256(np.asarray(volume.delta_refractive_index, dtype="<f8").tobytes()).hexdigest()

        assert result.provenance["grid"]["delta_refractive_index_sha256"] == expected_hash

        if kind == "generated":
            recorded = result.provenance

            reconstructed = GaussianMedium(
                background_refractive_index=recorded["medium"]["background_refractive_index"],
                refractive_index_std=recorded["medium"]["refractive_index_std"],
                correlation_length=recorded["medium"]["correlation_length_m"] * ureg.meter,
            ).to_volume(
                seed=recorded["seed"],
                grid=Grid(
                    shape=recorded["grid"]["shape"],
                    spacing=recorded["grid"]["spacing_m"] * ureg.meter,
                ),
            )

            np.testing.assert_array_equal(reconstructed.delta_refractive_index, volume.delta_refractive_index)

    assert result.provenance["order"] == 3

    assert result.provenance["grid"]["shape"] == [2, 3, 2]

    assert result.provenance["grid"]["spacing_m"] == pytest.approx(3e-08)

    assert result.provenance["coefficient_scope"] == "finite-sample"

    restored = Result.load(path=result.save(path=tmp_path / f"{kind}.npz"))

    for name in (
        "differential",
        "angles",
        "azimuths",
        "differential",
        "directions",
        "amplitudes",
        "term_differential",
        "stderr",
        "field_norms",
        "mu_s",
        "g",
        "mu_s_prime",
    ):
        expected = getattr(result, name)

        actual = getattr(restored, name)

        if expected is None:
            assert actual is None
        else:
            assert actual.is_compatible_with(expected.units)

            np.testing.assert_allclose(
                actual.to(expected.units).magnitude, expected.magnitude, rtol=1e-14, equal_nan=True
            )

    assert restored.warnings == result.warnings

    assert restored.provenance == result.provenance

    assert restored.realizations == result.realizations

    if result.amplitudes is not None:
        assert np.iscomplexobj(restored.amplitudes.magnitude)

        np.testing.assert_allclose(
            restored.differential.magnitude,
            np.sum(np.abs(np.cumsum(restored.amplitudes.magnitude, axis=0)) ** 2, axis=(-1, -2))
            / (2 * volume.volume.to("meter**3").magnitude),
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"directions": [[0, 0, 2], [0, 0, -1]]}, "unit vectors"),
        ({"angles": [1, 2] * ureg.radian}, "angles.*match"),
        ({"amplitudes": np.ones((2, 2, 3)) * ureg.meter}, "amplitudes.*shape"),
        ({"amplitudes": np.full((2, 2, 2, 3), complex(np.inf, 0)) * ureg.meter}, "amplitudes.*finite"),
        ({"field_norms": [[1, 1]]}, "field_norms.*shape"),
        ({"term_differential": [[1, 1]] * (1 / ureg.meter / ureg.steradian)}, "term_differential.*shape"),
        ({"mu_s": [1, 1] * (1 / ureg.meter)}, "integrated coefficients"),
    ],
)
def test_invalid_volume_results(changes, message):
    options = dict(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        kind="volume",
        differential=np.ones((2, 2)) * (1 / ureg.meter / ureg.steradian),
        directions=[[0, 0, 1], [0, 0, -1]],
    )

    options.update(changes)

    with pytest.raises(ValueError, match=message):
        Result(**options)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"realizations": None}, "realizations"),
        ({"realizations": True}, "realizations"),
        ({"realizations": 0}, "realizations"),
        ({"stderr": [[-1, 1]] * (1 / ureg.meter / ureg.steradian)}, "nonnegative"),
        ({"stderr": [[np.nan, 1]] * (1 / ureg.meter / ureg.steradian)}, "finite"),
        ({"stderr": [1, 1] * (1 / ureg.meter / ureg.steradian)}, "stderr.*shape"),
        ({"field_norms": [[1], [1]]}, "field_norms.*shape"),
        ({"realizations": 1, "stderr": [[1, 1]] * (1 / ureg.meter / ureg.steradian)}, "stderr.*NaN"),
    ],
)
def test_invalid_ensemble_results(changes, message):
    options = dict(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        kind="ensemble",
        differential=[[1, 1]] * (1 / ureg.meter / ureg.steradian),
        angles=[0, 1] * ureg.radian,
        realizations=3,
    )

    options.update(changes)

    with pytest.raises(ValueError, match=message):
        Result(**options)


def test_volume_generation_metadata_validation():
    with pytest.raises(ValueError, match="together"):
        Volume(
            delta_refractive_index=np.zeros((2,) * 3),
            seed=42,
            background_refractive_index=1.33,
            grid=Grid(
                shape=np.shape(np.zeros((2,) * 3)),
                spacing=3e-08 * ureg.meter,
            ),
        )

    with pytest.raises(ValueError, match="background_refractive_index"):
        Volume(
            delta_refractive_index=np.zeros((2,) * 3),
            medium=GaussianMedium(
                background_refractive_index=1.5,
                refractive_index_std=0.01,
                correlation_length=1e-07 * ureg.meter,
            ),
            seed=42,
            background_refractive_index=1.33,
            grid=Grid(
                shape=np.shape(np.zeros((2,) * 3)),
                spacing=3e-08 * ureg.meter,
            ),
        )
