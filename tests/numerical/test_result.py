"""Preserve numerical provenance, interference and uncertainty in archives."""

import hashlib

import numpy as np
import pytest

from bornsim import RandomMedium, Result, Solver, Source, Volume
from bornsim.media import random_volume
from bornsim.units import ureg


def test_cross_section_curves_and_errors_preserve_units_and_sample_normalization():
    volume = Volume(
        delta_index=np.zeros((2, 3, 4)),
        spacing=20e-9,
    )
    result = Result(
        source=Source(),
        kind="ensemble",
        differential=[[1, 3]],
        stderr=[[0.1, 0.2]],
        angles=[0, np.pi],
        realizations=3,
    )
    figure = result.plot_cross_section(
        volume=volume,
        area_unit="nanometer**2",
        title="Sample",
    )
    axis = figure.axes[0]
    scale = 24 * (20e-9) ** 3 * 1e18
    np.testing.assert_allclose(axis.lines[0].get_ydata(), np.array([1, 3]) * scale)
    segments = axis.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose(segments[0][:, 1], np.array([0.9, 1.1]) * scale)
    assert axis.get_title() == "Sample"
    assert axis.get_ylabel() == "Differential cross section (nm² sr⁻¹)"
    np.testing.assert_array_equal(result.differential.magnitude, [[1, 3]])


def test_cross_section_isolated_terms_and_validation():
    volume = Volume(
        delta_index=np.full((2, 2, 2), 0.01),
        spacing=20e-9,
    )
    result = Solver(
        source=Source(),
        order=2,
    ).solve_cut(
        target=volume,
        angles=[0, np.pi],
    )
    figure = result.plot_cross_section(
        volume=volume,
        terms=True,
        area_unit="meter**2",
    )
    for curve, expected in zip(figure.axes[0].lines, result.term_differential.magnitude):
        np.testing.assert_allclose(curve.get_ydata(), expected * volume.volume)
    with pytest.raises(ValueError, match="area unit"):
        result.plot_cross_section(
            volume=volume,
            area_unit="meter",
        )
    with pytest.raises(TypeError, match="Volume"):
        result.plot_cross_section(volume=1)
    with pytest.raises(ValueError, match="recorded sample"):
        result.plot_cross_section(
            volume=Volume(
                delta_index=volume.delta_index,
                spacing=30e-9,
            ),
        )


@pytest.mark.parametrize("kind", ["generated", "manual", "ensemble", "one-realization", "zero"])
def test_numerical_archives_round_trip(tmp_path, kind):
    medium = RandomMedium(
        correlation="gaussian",
        index_std=0 if kind == "zero" else 0.01,
    )
    solver = Solver(
        source=Source(),
        order=3,
    )
    if kind in ("ensemble", "one-realization", "zero"):
        result = solver.ensemble(
            medium=medium,
            shape=(2, 3, 2),
            spacing=30 * ureg.nanometer,
            seed=42,
            realizations=1 if kind == "one-realization" else 3,
            angles=[0, 90, 180] * ureg.degree,
            polar_samples=16,
            azimuth_samples=4,
        )
        assert result.provenance["seed"] == 42
        assert result.provenance["polar_samples"] == 16
        assert result.provenance["azimuth_samples"] == 4
        assert result.provenance["realizations"] == result.realizations
    else:
        generated = random_volume(
            medium=medium,
            shape=(2, 3, 2),
            spacing=30 * ureg.nanometer,
            seed=42,
        )
        volume = (
            generated
            if kind == "generated"
            else Volume(
                delta_index=generated.delta_index,
                spacing=generated.spacing,
            )
        )
        result = solver.solve_cut(
            target=volume,
            angles=[0, 90, 180] * ureg.degree,
        )
        assert result.provenance["seed"] == (42 if kind == "generated" else None)
        assert result.provenance["medium"] is not None if kind == "generated" else result.provenance["medium"] is None
        expected_hash = hashlib.sha256(np.asarray(volume.delta_index, dtype="<f8").tobytes()).hexdigest()
        assert result.provenance["grid"]["delta_index_sha256"] == expected_hash
        if kind == "generated":
            recorded = result.provenance
            reconstructed = random_volume(
                medium=RandomMedium(
                    **{
                        **{key: value for key, value in recorded["medium"].items() if key != "correlation_length_m"},
                        "correlation_length": recorded["medium"]["correlation_length_m"],
                    }
                ),
                shape=recorded["grid"]["shape"],
                spacing=recorded["grid"]["spacing_m"],
                seed=recorded["seed"],
            )
            np.testing.assert_array_equal(reconstructed.delta_index, volume.delta_index)
    assert result.provenance["order"] == 3
    assert result.provenance["grid"]["shape"] == [2, 3, 2]
    assert result.provenance["grid"]["spacing_m"] == pytest.approx(30e-9)
    assert result.provenance["coefficient_scope"] == "finite-sample"
    restored = Result.load(path=result.save(path=tmp_path / f"{kind}.npz"))
    for name in (
        "differential",
        "angles",
        "azimuths",
        "directional_differential",
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
            assert actual.units == expected.units
            np.testing.assert_array_equal(actual.magnitude, expected.magnitude)
    assert restored.warnings == result.warnings
    assert restored.provenance == result.provenance
    assert restored.realizations == result.realizations
    if result.amplitudes is not None:
        assert np.iscomplexobj(restored.amplitudes.magnitude)
        np.testing.assert_allclose(
            restored.differential.magnitude,
            np.sum(np.abs(np.cumsum(restored.amplitudes.magnitude, axis=0)) ** 2, axis=(-1, -2)) / (2 * volume.volume),
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"directions": [[0, 0, 2], [0, 0, -1]]}, "unit vectors"),
        ({"angles": [1, 2]}, "angles.*match"),
        ({"amplitudes": np.ones((2, 2, 3))}, "amplitudes.*shape"),
        ({"amplitudes": np.full((2, 2, 2, 3), complex(np.inf, 0))}, "amplitudes.*finite"),
        ({"field_norms": [[1, 1]]}, "field_norms.*shape"),
        ({"term_differential": [[1, 1]]}, "term_differential.*shape"),
        ({"mu_s": [1, 1]}, "integrated coefficients"),
    ],
)
def test_invalid_volume_results(changes, message):
    options = dict(source=Source(), kind="volume", differential=np.ones((2, 2)), directions=[[0, 0, 1], [0, 0, -1]])
    options.update(changes)
    with pytest.raises(ValueError, match=message):
        Result(**options)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"realizations": None}, "realizations"),
        ({"realizations": True}, "realizations"),
        ({"realizations": 0}, "realizations"),
        ({"stderr": [[-1, 1]]}, "nonnegative"),
        ({"stderr": [[np.nan, 1]]}, "finite"),
        ({"stderr": [1, 1]}, "stderr.*shape"),
        ({"field_norms": [[1], [1]]}, "field_norms.*shape"),
        ({"realizations": 1, "stderr": [[1, 1]]}, "stderr.*NaN"),
    ],
)
def test_invalid_ensemble_results(changes, message):
    options = dict(source=Source(), kind="ensemble", differential=[[1, 1]], angles=[0, 1], realizations=3)
    options.update(changes)
    with pytest.raises(ValueError, match=message):
        Result(**options)


def test_volume_generation_metadata_validation():
    with pytest.raises(ValueError, match="together"):
        Volume(
            delta_index=np.zeros((2,) * 3),
            spacing=30e-9,
            seed=42,
        )
    with pytest.raises(ValueError, match="background_index"):
        Volume(
            delta_index=np.zeros((2,) * 3),
            spacing=30e-9,
            medium=RandomMedium(
                correlation="gaussian",
                background_index=1.5,
            ),
            seed=42,
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"azimuths": [0, np.pi / 2, np.pi, 3 * np.pi / 2]},
        {"directional_differential": np.ones((1, 3, 4))},
        {"azimuths": [[0, 1, 2, 3]], "directional_differential": np.ones((1, 3, 4))},
        {"azimuths": [0, 1, 2, 3], "directional_differential": np.ones((1, 3, 4))},
        {"azimuths": [0, 1, 2, np.nan], "directional_differential": np.ones((1, 3, 4))},
        {"azimuths": np.arange(4, dtype=complex), "directional_differential": np.ones((1, 3, 4))},
        {"azimuths": np.arange(4) * np.pi / 2, "directional_differential": np.ones((1, 3, 3))},
        {"azimuths": np.arange(4) * np.pi / 2, "directional_differential": -np.ones((1, 3, 4))},
        {"azimuths": np.arange(4) * np.pi / 2, "directional_differential": 2 * np.ones((1, 3, 4))},
    ],
)
def test_directional_archive_fields_require_consistent_angular_grid_and_intensities(fields):
    with pytest.raises(ValueError):
        Result(
            source=Source(),
            kind="ensemble",
            realizations=1,
            differential=np.ones((1, 3)),
            angles=[0, np.pi / 2, np.pi],
            **fields,
        )


def test_legacy_ensemble_3d_requires_directional_data_instead_of_assuming_symmetry():
    result = Result(
        source=Source(),
        kind="ensemble",
        realizations=1,
        differential=np.ones((1, 3)),
        angles=[0, np.pi / 2, np.pi],
        mu_s=[4 * np.pi],
    )
    with pytest.raises(ValueError, match="recompute with Solver.solve"):
        result.plot_phase_function(view="3d")
    # Existing angular data remain available without manufacturing azimuths.
    np.testing.assert_allclose(result.phase_function.magnitude, 1 / (4 * np.pi))
